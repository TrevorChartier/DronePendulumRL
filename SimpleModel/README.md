# Simple Starling dynamics model

Train an MLP to predict the next drone and pendulum state over one 0.1 s control interval. The Starling flies in 3D and both pendulum hinges rotate about body X, so the links swing in the drone's Y–Z plane. The links are 100 mm and 75 mm long with endpoint masses of 79 g and 49 g and rod masses of 38 g and 19 g.

## Run from the repository root

The generator uses the root `requirements.txt`. The trainer additionally needs PyTorch. The dataset is a NumPy file, so generation and training can run under different Python interpreters. On this workstation the existing `rob831` environment provides CUDA-enabled PyTorch.

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python SimpleModel/generate_data.py --train 500000 --workers 12
python SimpleModel/train.py --epochs 120
```

If PyTorch is installed only in `rob831`, use its interpreter for the last command.

```bash
~/miniconda3/envs/rob831/bin/python SimpleModel/train.py --epochs 120
```

Training selects CUDA automatically when available. Use `--device cpu` to override it, `--trials` to run specific configurations, or `--help` to see the available flags.

## Dataset and evaluation

- Start each sample at `[0, 0, 5]` m and independently randomize the attitude, world velocity, and pendulum angles and rates within the root README's bounds. Hold one random command for ten MuJoCo ticks to generate the target.
- The model sees `[roll, pitch, cos(yaw), sin(yaw), vx, vy, vz, theta1, theta2, omega1, omega2]` and `[T, p_des, q_des, r_des]`. Angles and rates use radians and radians per second. Velocity is in m/s and thrust in newtons. The ideal controller makes the next body rates equal to the commanded rates, so they can be filled in directly.
- Fit normalized state changes with mean squared error. Validation and test each include independently reset one-step samples and separate 10-step trajectories. Rollout predictions feed back into the model for the full second, using the same action sequence as MuJoCo.
- Select the checkpoint using validation one-step normalized RMSE plus 0.1 times 10-step normalized RMSE. Normalized errors use the standard deviation of training changes for each component. Per-component RMSE is reported in the state vector's original units.
- `SimpleModel/runs/sweep/best.pt` stores the weights, normalization statistics, architecture, and state/action ordering. `trials.json` contains validation metrics and convergence histories. `test.json` contains test metrics for the selected checkpoint. Local datasets and training outputs under `runs/` are ignored by Git.

Load the checkpoint and call `predict` with float32 tensors to obtain the next state in physical units.

```python
import torch
from SimpleModel.train import DynamicsMLP, Trial

checkpoint = torch.load("SimpleModel/runs/sweep/best.pt", map_location="cpu")
model = DynamicsMLP(len(checkpoint["state_names"]), len(checkpoint["action_names"]),
                    Trial(**checkpoint["trial"]))
model.load_state_dict(checkpoint["state_dict"])
model.eval()
```
