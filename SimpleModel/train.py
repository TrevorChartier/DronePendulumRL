"""Train and compare normalized one-step MLPs on Starling transitions."""

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class Trial:
    name: str
    width: int
    depth: int
    activation: str
    learning_rate: float = 1e-3
    batch_size: int = 2048
    weight_decay: float = 1e-5


TRIALS = (
    Trial("silu_128x2", 128, 2, "silu"),
    Trial("silu_256x3", 256, 3, "silu"),
    Trial("silu_384x4", 384, 4, "silu", 7e-4),
    Trial("tanh_256x3", 256, 3, "tanh"),
    Trial("gelu_256x3", 256, 3, "gelu"),
    Trial("relu_256x3", 256, 3, "relu"),
    Trial("silu_256x3_small_batch", 256, 3, "silu", 5e-4, 1024, 1e-4),
)


class DynamicsMLP(nn.Module):
    """Predict normalized state changes and decode them to physical units."""

    def __init__(self, state_size, action_size, trial):
        super().__init__()
        activation = {"silu": nn.SiLU, "tanh": nn.Tanh, "gelu": nn.GELU, "relu": nn.ReLU}
        layers = []
        size = state_size + action_size
        for _ in range(trial.depth):
            layers.extend((nn.Linear(size, trial.width), activation[trial.activation]()))
            size = trial.width
        layers.append(nn.Linear(size, state_size))
        self.network = nn.Sequential(*layers)
        self.register_buffer("input_mean", torch.zeros(state_size + action_size))
        self.register_buffer("input_std", torch.ones(state_size + action_size))
        self.register_buffer("delta_mean", torch.zeros(state_size))
        self.register_buffer("delta_std", torch.ones(state_size))

    def forward(self, states, actions):
        inputs = torch.cat((states, actions), dim=-1)
        return self.network((inputs - self.input_mean) / self.input_std)

    def predict(self, states, actions):
        predicted = states + self.delta_mean + self.delta_std * self(states, actions)
        yaw = nn.functional.normalize(predicted[..., 2:4], dim=-1)
        return torch.cat((predicted[..., :2], yaw, predicted[..., 4:]), dim=-1)


@torch.no_grad()
def evaluate(model, transitions, trajectories):
    states, actions = transitions
    target = states[:, 1]
    next_state = model.predict(states[:, 0], actions[:, 0])
    error = next_state - target
    normalized_mse = (error / model.delta_std).square().mean()
    persistence_mse = ((states[:, 0] - target) / model.delta_std).square().mean()

    states, actions = trajectories
    prediction = states[:, 0]
    rollout_errors = []
    for step in range(actions.shape[1]):
        prediction = model.predict(prediction, actions[:, step])
        rollout_errors.append(prediction - states[:, step + 1])
    rollout_error = torch.stack(rollout_errors, dim=1)
    return {
        "one_step_normalized_rmse": normalized_mse.sqrt().item(),
        "persistence_normalized_rmse": persistence_mse.sqrt().item(),
        "one_step_rmse": error.square().mean(dim=0).sqrt().tolist(),
        "rollout_final_rmse": rollout_error[:, -1].square().mean(dim=0).sqrt().tolist(),
        "rollout_normalized_rmse_by_step": (
            (rollout_error / model.delta_std).square().mean(dim=(0, 2)).sqrt().tolist()
        ),
    }


def train_trial(trial, data, epochs, patience, seed, device):
    torch.manual_seed(seed)
    states, actions = data["train"]
    states = states[:, 0].contiguous()
    actions = actions[:, 0].contiguous()
    deltas = data["train"][0][:, 1] - states
    model = DynamicsMLP(states.shape[-1], actions.shape[-1], trial).to(device)
    inputs = torch.cat((states, actions), dim=-1)
    model.input_mean.copy_(inputs.mean(0))
    model.input_std.copy_(inputs.std(0, unbiased=False).clamp_min(1e-6))
    model.delta_mean.copy_(deltas.mean(0))
    model.delta_std.copy_(deltas.std(0, unbiased=False).clamp_min(1e-6))
    targets = (deltas - model.delta_mean) / model.delta_std

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=trial.learning_rate, weight_decay=trial.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=trial.learning_rate * 0.05
    )
    history = []
    best_score = math.inf
    best_weights = None
    best_epoch = 0
    start = time.monotonic()
    for epoch in range(1, epochs + 1):
        for indices in torch.randperm(len(states), device=device).split(trial.batch_size):
            optimizer.zero_grad(set_to_none=True)
            predicted = model(states[indices], actions[indices])
            loss = nn.functional.mse_loss(predicted, targets[indices])
            loss.backward()
            optimizer.step()
        scheduler.step()

        validation = evaluate(model, data["validation"], data["validation_rollout"])
        one_step = validation["one_step_normalized_rmse"]
        rollout = validation["rollout_normalized_rmse_by_step"][-1]
        # Validate on both the training objective and one-second open-loop accuracy.
        score = one_step + 0.1 * rollout
        history.append({"epoch": epoch, "score": score, "one_step": one_step, "rollout": rollout})
        if math.isfinite(score) and score < best_score:
            best_score = score
            best_epoch = epoch
            best_weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"{trial.name} epoch {epoch:3d}: one step {one_step:.5f}, rollout {rollout:.5f}", flush=True)
        if epoch - best_epoch >= patience:
            break

    if best_weights is None:
        raise RuntimeError(f"{trial.name} did not produce a finite validation score")
    model.load_state_dict(best_weights)
    return model, {
        "trial": asdict(trial),
        "best_epoch": best_epoch,
        "score": best_score,
        "seconds": time.monotonic() - start,
        "validation": evaluate(model, data["validation"], data["validation_rollout"]),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("SimpleModel/runs/data.npz"))
    parser.add_argument("--output", type=Path, default=Path("SimpleModel/runs/sweep"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--trials", nargs="+", choices=[trial.name for trial in TRIALS])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if min(args.epochs, args.patience) < 1:
        parser.error("Epochs and patience must be positive.")

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    print(f"Training on {device}", flush=True)
    with np.load(args.data, allow_pickle=False) as dataset:
        state_names = dataset["state_names"].tolist()
        control_dt = float(dataset["control_dt"])
        data = {
            split: tuple(torch.from_numpy(dataset[f"{split}_{kind}"]).to(device) for kind in ("states", "actions"))
            for split in ("train", "validation", "test", "validation_rollout", "test_rollout")
        }

    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    best_score = math.inf
    for trial in TRIALS:
        if args.trials and trial.name not in args.trials:
            continue
        model, report = train_trial(trial, data, args.epochs, args.patience, args.seed, device)
        reports.append(report)
        (args.output / "trials.json").write_text(json.dumps(reports, indent=2) + "\n")
        if report["score"] < best_score:
            best_score = report["score"]
            best_model, best_report = model, report
            torch.save({
                "trial": asdict(trial),
                "state_names": state_names,
                "action_names": ["thrust", "p_des", "q_des", "r_des"],
                "control_dt": control_dt,
                "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            }, args.output / "best.pt")

    test = evaluate(best_model, data["test"], data["test_rollout"])
    summary = {
        "selected_trial": best_report["trial"]["name"],
        "best_epoch": best_report["best_epoch"],
        "state_names": state_names,
        "control_dt": control_dt,
        "test": test,
    }
    (args.output / "test.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
