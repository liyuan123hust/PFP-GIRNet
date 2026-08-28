import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from model.bimamba_encoder import BiMambaEncoder

# =========================================================================
# Graph Convolution Layer (Single Graph)
# =========================================================================
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)  # [B, N, D_out]
        output = torch.bmm(adj, support)  # [B, N, D_out]
        if self.bias is not None:
            output = output + self.bias
        return F.relu(output)

# =========================================================================
# Multi-Graph Convolution Module (Spatial Branch)
# Fuses Social, EA, and TCPA graphs seamlessly
# =========================================================================
class MultiGraphConv(nn.Module):
    def __init__(self, d_model, num_graphs=3, num_gcn_layers=2, dropout=0.0):
        super().__init__()
        self.num_graphs = num_graphs
        self.d_model = d_model
        
        self.gcn_branches = nn.ModuleList([
            nn.Sequential(
                GraphConvLayer(d_model, d_model),
                GraphConvLayer(d_model, d_model)
            ) for _ in range(num_graphs)
        ])
        
        self.attention_proj = nn.Sequential(
            nn.Linear(d_model * num_graphs, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_graphs),
            nn.Softmax(dim=-1)
        )
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.output_dropout = nn.Dropout(dropout)
    
    def forward(self, x, graphs):
        B, N, D = x.shape
        branch_outputs = []
        for i, (gcn, adj) in enumerate(zip(self.gcn_branches, graphs)):
            h = x
            for layer in gcn:
                h = layer(h, adj)
            branch_outputs.append(h)  
        
        stacked = torch.stack(branch_outputs, dim=2)
        concat_features = torch.cat(branch_outputs, dim=-1)
        attn_weights = self.attention_proj(concat_features)
        
        fused = torch.sum(stacked * attn_weights.unsqueeze(-1), dim=2)
        output = self.out_proj(fused)
        output = self.norm(output + x)
        
        return self.output_dropout(output)

    def forward_sequence(self, x, graphs):
        """
        Batched temporal variant for inference/training.
        x: [B, T, N, D]
        graphs: list[[B, T, N, N], ...]
        """
        B, T, N, D = x.shape
        x_flat = x.reshape(B * T, N, D)
        branch_outputs = []

        for gcn, adj in zip(self.gcn_branches, graphs):
            adj_flat = adj.reshape(B * T, N, N)
            h = x_flat
            for layer in gcn:
                h = layer(h, adj_flat)
            branch_outputs.append(h)

        stacked = torch.stack(branch_outputs, dim=2)
        concat_features = torch.cat(branch_outputs, dim=-1)
        attn_weights = self.attention_proj(concat_features)

        fused = torch.sum(stacked * attn_weights.unsqueeze(-1), dim=2)
        output = self.out_proj(fused)
        output = self.norm(output + x_flat)
        output = self.output_dropout(output)
        return output.reshape(B, T, N, D)

# =========================================================================
# Frequency Domain Branch (1D Discrete Wavelet Transform)
# =========================================================================
class FrequencyDomainBranch(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.register_buffer('h0', torch.tensor([0.70710678, 0.70710678]).view(1, 1, 2))
        self.register_buffer('h1', torch.tensor([-0.70710678, 0.70710678]).view(1, 1, 2))
        self.register_buffer('ih0', torch.tensor([0.70710678, 0.70710678]).view(1, 1, 2))
        self.register_buffer('ih1', torch.tensor([0.70710678, -0.70710678]).view(1, 1, 2))
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        x_trans = x.transpose(1, 2).reshape(B*D, 1, T)
        
        if T % 2 != 0:
            x_trans = F.pad(x_trans, (0, 1), mode='replicate')
            T_pad = T + 1
        else:
            T_pad = T
            
        approx = F.conv1d(x_trans, self.h0, stride=2)  
        detail = F.conv1d(x_trans, self.h1, stride=2)  
        
        approx = F.avg_pool1d(approx, kernel_size=3, stride=1, padding=1)
        detail = F.max_pool1d(detail, kernel_size=3, stride=1, padding=1)
        
        x0 = F.conv_transpose1d(approx, self.ih0, stride=2)
        x1 = F.conv_transpose1d(detail, self.ih1, stride=2)
        
        out = x0 + x1 
        if T % 2 != 0:
            out = out[..., :T]
            
        out = out.reshape(B, D, T).transpose(1, 2)  
        return self.proj(out)

# =========================================================================
# Spatial-Spectral Joint Feature Fusion
# =========================================================================
class SpatialSpectralFusion(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.freq_branch = FrequencyDomainBranch(d_model)
        self.fusion_linear = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        self.fusion_dropout = nn.Dropout(dropout)
    
    def forward(self, x_spatial):
        x_freq = self.freq_branch(x_spatial)
        concat_feat = torch.cat([x_spatial, x_freq], dim=-1)
        fused = self.fusion_linear(concat_feat)
        fused = self.fusion_dropout(fused)
        return self.norm(fused + x_spatial) 

# =========================================================================
# Temporal Extraction (BiMamba)
# =========================================================================
class TemporalExtractor(nn.Module):
    def __init__(
        self,
        d_model,
        mamba_layers=2,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_dropout=0.1,
    ):
        super().__init__()
        self.encoder = BiMambaEncoder(
            d_model=d_model,
            num_layers=mamba_layers,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dropout=mamba_dropout,
        )
    
    def forward(self, x):
        return self.encoder(x)

# =========================================================================
# Intent Generation
# =========================================================================
class IntentGenerator(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.intent_dropout = nn.Dropout(dropout)

    def forward(self, H_hist):
        H_fut = self.intent_dropout(self.mlp(H_hist))
        return H_fut

# =========================================================================
# Entropy-Regularized Polarization Gate Fusion
# =========================================================================
# =========================================================================
# Residual Polarization Gate Fusion (Updated to Section 4.4)
# =========================================================================
class EntropyPolarizationGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_z = nn.Linear(d_model * 2, d_model)
        
        # Linear transformation for intent correction increment \Delta h
        self.phi = nn.Linear(d_model, d_model)
        
    def forward(self, H_hist, H_fut):
        # 1. Generate polarization gating vector z
        concat_feat = torch.cat([H_hist, H_fut], dim=-1)
        z = torch.sigmoid(self.W_z(concat_feat))  # z = \sigma(W_z[h_hist || h_fut] + b_z) 
        
        # 2. Extract intent correction increment \Delta h
        delta_h = self.phi(H_fut - H_hist) # \Delta h = \phi(h_fut - h_hist)
        
        # 3. Dynamic residual fusion
        H_final = H_hist + z * delta_h # h_final = h_hist + z \odot \Delta h
        
        return H_final, z

# =========================================================================
# Dual-domain architecture
# =========================================================================
class ShipTrajectoryRefiner(nn.Module):
    def __init__(
        self, 
        num_ships, 
        input_dim=2,
        d_model=64, 
        hist_len=30,
        pred_len=24,
        num_layers=2,
        use_multi_graph=True,
        ship_length=200.0,  
        time_interval=5.0,
        spatial_dropout=0.0,
        fusion_dropout=0.0,
        intent_dropout=0.0,
        decoder_dropout=0.0,
        mamba_layers=2,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_dropout=0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.use_multi_graph = use_multi_graph
        self.ship_length = ship_length
        self.time_interval = time_interval
        
        self.min_turn_radius = 1.5 * ship_length  
        self.max_turn_radius = 2.5 * ship_length  
        
        self.embedding = nn.Linear(input_dim, d_model)
        
        if use_multi_graph:
            self.spatial_branch = MultiGraphConv(d_model, num_graphs=3, dropout=spatial_dropout)
        
        self.spatial_spectral_fusion = SpatialSpectralFusion(d_model, dropout=fusion_dropout)
        self.temporal_extractor = TemporalExtractor(
            d_model,
            mamba_layers=mamba_layers,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            mamba_dropout=mamba_dropout,
        )
        self.intent_generator = IntentGenerator(d_model, dropout=intent_dropout)
        self.polarization_gate = EntropyPolarizationGate(d_model)
        self.history_feature_dropout = nn.Dropout(fusion_dropout)
        self.final_decoder_dropout = nn.Dropout(decoder_dropout)
        self.base_decoder_dropout = nn.Dropout(decoder_dropout)
        
        # The model should only predict 2D coordinates (dx, dy) for the trajectory
        self.final_decoder = nn.Linear(self.hist_len * self.d_model, 2 * pred_len)
        
        # Base decoder for L_IG (Generates \hat{Y}_{base} from H_hist)
        self.base_decoder = nn.Linear(self.hist_len * self.d_model, 2 * pred_len)
        
        # Learnable parameters for homoscedastic uncertainty (\sigma_1, \sigma_2, \sigma_3)
        # We initialize their log variances (log(\sigma^2)) to 0 for numerical stability
        self.log_vars = nn.Parameter(torch.zeros(3))
        
    def forward(self, x_history, graphs=None, return_aux=True):
        B, N, T_hist, F = x_history.shape
        T_pred = self.pred_len
        
        x = self.embedding(x_history)  # [B, N, T, D]
        
        if self.use_multi_graph and graphs is not None:
            x_seq = x.permute(0, 2, 1, 3).contiguous()
            graph_seq = [
                graphs['social'].contiguous(),
                graphs['ea'].contiguous(),
                graphs['tcpa'].contiguous(),
            ]
            x_spatial = self.spatial_branch.forward_sequence(x_seq, graph_seq)
            x_spatial = x_spatial.permute(0, 2, 1, 3).contiguous()
        else:
            x_spatial = x
            
        x_spatial_flat = rearrange(x_spatial, 'b n t d -> (b n) t d')
        
        H_fusion = self.spatial_spectral_fusion(x_spatial_flat) 
        H_hist = self.temporal_extractor(H_fusion) 
        H_hist = self.history_feature_dropout(H_hist)
        
        H_fut = self.intent_generator(H_hist) 
        H_final, z_gate = self.polarization_gate(H_hist, H_fut) 
        
        H_final_flat = H_final.reshape(B * N, T_hist * self.d_model)
        y_pred = self.final_decoder(self.final_decoder_dropout(H_final_flat))
        y_pred = rearrange(y_pred, 'bn (t f) -> bn t f', t=T_pred, f=2) # Only 2D output
        
        y_pred = rearrange(y_pred, '(b n) t d -> b n t d', b=B)

        if not return_aux:
            return y_pred

        z_gate = rearrange(z_gate, '(b n) t d -> b n t d', b=B)

        # Base prediction \hat{Y}_{base} using only historical state
        H_hist_flat = H_hist.reshape(B * N, T_hist * self.d_model)
        y_base = self.base_decoder(self.base_decoder_dropout(H_hist_flat))
        y_base = rearrange(y_base, 'bn (t f) -> bn t f', t=T_pred, f=2) # Only 2D output
        y_base = rearrange(y_base, '(b n) t d -> b n t d', b=B)

        return y_pred, y_base, z_gate, self.log_vars
