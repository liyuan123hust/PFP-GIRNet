# Commands

```bash
python train_and_test/train.py
python train_and_test/test.py
```

The training command creates a new dx/dy checkpoint in `checkpoints/pfp_girnet_skagen_region/<obs_len>-<pred_len>/`; the test command uses its `best_model.pth` by default.