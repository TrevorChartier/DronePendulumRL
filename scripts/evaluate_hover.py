# This compares a saved PPO policy with LQR and can save a video and plot over SSH.

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from drone_pendulum.controllers.lqr_2d import LQRController
from drone_pendulum.controllers.ppo_hover import PPOHoverController
from drone_pendulum.envs.hover_env import HoverConfig, HoverEnv
from run_planar import plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--controller", choices=["ppo", "lqr"], default="ppo")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--push", type=float, default=0.0, help="force on link2 at 1 s, for 0.1 s")
    parser.add_argument("--video", type=Path, help="record the first episode")
    parser.add_argument("--plot-file", type=Path, help="plot the first episode")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.render and args.video:
        parser.error("choose either --render or --video")
    saved = json.loads((args.run_dir / "config.json").read_text())
    cfg = HoverConfig(**saved["environment"])
    if args.seconds is not None:
        cfg = replace(cfg, max_episode_seconds=args.seconds)
    mode = "human" if args.render else ("rgb_array" if args.video else None)
    env = HoverEnv(cfg, render_mode=mode)
    controller = None
    writer = None
    results = []
    try:
        controller = (LQRController.from_env(env) if args.controller == "lqr" else
                      PPOHoverController(args.run_dir / "policy.zip",
                                         args.run_dir / "vecnormalize.pkl", cfg))
        if args.video:
            import imageio.v2 as imageio
            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(args.video, fps=round(1 / env.control_dt))
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            controller.reset()
            positions, angles = [], []
            states, actions, rewards, times = [info["state"]], [], [], [info["time"]]
            term_totals = {}
            pushed = False
            started = time.perf_counter()
            if writer is not None and episode == 0:
                writer.append_data(env.render())
            while True:
                if args.push and not pushed and info["time"] >= 1.0:
                    env.sim.apply_disturbance("link2", [args.push, 0.0], 0.1)
                    pushed = True
                action = controller.act(obs, info)
                obs, reward, terminated, truncated, info = env.step(action)
                positions.append(obs[:2].copy())
                angles.append(info["absolute_link_angles"])
                states.append(info["state"])
                times.append(info["time"])
                actions.append(action)
                rewards.append(reward)
                for name, value in info["reward_terms"].items():
                    term_totals[name] = term_totals.get(name, 0.0) + value
                if writer is not None and episode == 0:
                    writer.append_data(env.render())
                if args.render:
                    time.sleep(max(0, started + env.step_count * env.control_dt - time.perf_counter()))
                if terminated or truncated:
                    break
            results.append({
                "seed": args.seed + episode, "return": float(sum(rewards)),
                "seconds": info["time"], "failed": terminated,
                "failure_reason": info["failure_reason"],
                "position_rmse_m": float(np.sqrt(np.mean(np.square(positions)))),
                "link_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(np.square(angles))))),
                "action_rms": float(np.sqrt(np.mean(np.square(actions)))),
                "mean_reward_terms": {k: v / len(rewards) for k, v in term_totals.items()},
            })
            if episode == 0 and args.plot_file:
                from drone_pendulum.evaluation.rollout import Trajectory
                args.plot_file.parent.mkdir(parents=True, exist_ok=True)
                trajectory = Trajectory(t=np.array(times), states=np.array(states),
                                        actions=np.array(actions), rewards=np.array(rewards),
                                        terminated=terminated)
                plot(trajectory, args.plot_file)
    finally:
        if writer is not None:
            writer.close()
        if isinstance(controller, PPOHoverController):
            controller.close()
        env.close()
    report = {"controller": args.controller, "push_newtons": args.push,
              "episodes": results,
              "failure_rate": float(np.mean([r["failed"] for r in results])),
              "mean_return": float(np.mean([r["return"] for r in results]))}
    output = args.run_dir / f"evaluation_{args.controller}_{args.seed}_{time.time_ns()}.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
