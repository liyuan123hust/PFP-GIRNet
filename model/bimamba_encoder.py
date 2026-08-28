from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """A lightweight MLP for generating channel-attention weights."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalMambaBlock(nn.Module):
    """
    A pure PyTorch Mamba-like temporal block.

    This block includes input projection, gating, depthwise temporal convolution, stepwise state updates, output projection, residual connections, and LayerNorm. Both input and output have shape [B, T, C].
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_state < 1:
            raise ValueError("d_state must be >= 1 for LocalMambaBlock.")
        if d_conv < 1:
            raise ValueError("d_conv must be >= 1 for LocalMambaBlock.")
        if expand < 1:
            raise ValueError("expand must be >= 1 for LocalMambaBlock.")

        self.d_model = d_model
        self.d_state = d_state
        self.inner_dim = d_model * expand

        self.in_proj = nn.Linear(d_model, self.inner_dim * 2)
        self.depthwise_conv = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            kernel_size=d_conv,
            groups=self.inner_dim,
            padding=d_conv - 1,
        )
        self.delta_proj = nn.Linear(self.inner_dim, self.inner_dim)
        self.b_proj = nn.Linear(self.inner_dim, d_state)
        self.c_proj = nn.Linear(self.inner_dim, d_state)
        self.a_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(self.inner_dim, 1))
        self.d_skip = nn.Parameter(torch.ones(self.inner_dim))
        self.out_proj = nn.Linear(self.inner_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def _selective_scan(self, u: torch.Tensor) -> torch.Tensor:
        """
        Stepwise state update.

        u: [B, T, H], where H is the expanded channel dimension.
        return: [B, T, H]
        """
        batch_size, seq_len, hidden_dim = u.shape
        state = u.new_zeros(batch_size, hidden_dim, self.d_state)
        a = -torch.exp(self.a_log).unsqueeze(0)
        d_skip = self.d_skip.view(1, hidden_dim)

        delta = F.softplus(self.delta_proj(u))
        b_t = self.b_proj(u)
        c_t = self.c_proj(u)

        outputs = []
        for step in range(seq_len):
            delta_step = delta[:, step, :].unsqueeze(-1)
            u_step = u[:, step, :]
            b_step = b_t[:, step, :].unsqueeze(1)
            c_step = c_t[:, step, :].unsqueeze(1)

            decay = torch.exp(delta_step * a)
            state = decay * state + delta_step * u_step.unsqueeze(-1) * b_step
            y_step = (state * c_step).sum(dim=-1) + d_skip * u_step
            outputs.append(y_step)

        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"LocalMambaBlock expects [B, T, C], got shape {tuple(x.shape)}")

        residual = x
        x_proj, gate = self.in_proj(x).chunk(2, dim=-1)

        conv_input = x_proj.transpose(1, 2)
        conv_out = self.depthwise_conv(conv_input)[..., : x.shape[1]]
        conv_out = F.silu(conv_out.transpose(1, 2))

        ssm_out = self._selective_scan(conv_out)
        gated = ssm_out * F.silu(gate)
        out = self.out_proj(gated)
        return self.norm(residual + self.dropout(out))


class BiMambaBlock(nn.Module):
    """
    A single-layer bidirectional Mamba-like encoder block.

    Both input and output have shape [B, T, C]; the reverse branch is implemented by flipping the time dimension.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mamba_fwd = LocalMambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        self.mamba_rev = LocalMambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        self.channel_fc = FeedForward(
            input_dim=d_model * 4,
            hidden_dim=d_model,
            output_dim=d_model * 2,
            dropout=dropout,
        )
        self.out_proj = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"BiMambaBlock expects [B, T, C], got shape {tuple(x.shape)}")

        x_fwd = x
        y_fwd = self.mamba_fwd(x_fwd)

        x_rev = torch.flip(x, dims=[1])
        y_rev = self.mamba_rev(x_rev)
        y_rev = torch.flip(y_rev, dims=[1])

        y_cat = torch.cat([y_fwd, y_rev], dim=-1)
        avg_pool = y_cat.mean(dim=1)
        max_pool = y_cat.max(dim=1).values
        pool_cat = torch.cat([avg_pool, max_pool], dim=-1)
        omega = torch.sigmoid(self.channel_fc(pool_cat)).unsqueeze(1)
        y_att = y_cat * omega

        out = self.out_proj(y_att)
        return self.norm(x + self.dropout(out))


class BiMambaEncoder(nn.Module):
    """A multi-layer bidirectional Mamba temporal encoder."""

    def __init__(
        self,
        d_model: int,
        num_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1 for BiMambaEncoder.")

        self.layers = nn.ModuleList(
            [
                BiMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
