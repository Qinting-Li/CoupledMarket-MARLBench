"""Command-line entry point for CoupledMarket-MARLBench."""

from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np

from market_env import list_scenarios, make_env
from train import run_algorithm_comparison, run_all_scenarios
from visualize import generate_all_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CoupledMarket-MARLBench runner")
    parser.add_argument("--scenario-name", default="electricity_carbon_green", choices=list_scenarios())
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="run 25 episodes for smoke testing")
    parser.add_argument("--describe", action="store_true", help="print scenario metadata and exit")
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.describe:
        env = make_env(args.scenario_name, seed=args.seed)
        desc = env.scenario_description() if hasattr(env, "scenario_description") else {
            "scenario": args.scenario_name,
            "num_agents": env.num_agents,
            "state_dim": env.state_dim,
            "action_dim": env.action_dim,
        }
        print(json.dumps(desc, indent=2, ensure_ascii=False))
        return

    if args.scenario_name != "electricity_carbon_green":
        env = make_env(args.scenario_name, seed=args.seed)
        print(json.dumps(env.scenario_description(), indent=2, ensure_ascii=False))
        print("Template scenarios expose the benchmark interface; training baselines are implemented for electricity_carbon_green.")
        return

    episodes = 25 if args.quick else int(args.episodes)
    save_dir = os.path.abspath(args.output_dir)
    os.makedirs(save_dir, exist_ok=True)
    scenario_results = run_all_scenarios("MA-CETD3", episodes, args.seed)
    algo_results = run_algorithm_comparison(episodes, args.seed)
    with open(os.path.join(save_dir, "scenario_algo_results.pkl"), "wb") as f:
        pickle.dump({"scenario_results": scenario_results, "algo_results": algo_results}, f)

    generate_all_plots(
        scenario_results,
        algo_results,
        quota_results=None,
        penalty_results=None,
        save_dir=save_dir,
    )

    summary = {"episodes": episodes, "seed": args.seed, "scenario_revenues": {}, "algorithm_comparison": {}}
    for sc in [1, 2, 3]:
        rev = scenario_results[sc]["revenues"]
        summary["scenario_revenues"][f"scenario_{sc}"] = {
            name: float(rev[i]) for i, name in enumerate(["T1", "T2", "WT", "PV", "V1", "V2"])
        }
        summary["scenario_revenues"][f"scenario_{sc}"]["total"] = float(np.sum(rev))
    for algo in ["MATD3", "MA-CETD3"]:
        rev = algo_results[algo]["revenues"]
        summary["algorithm_comparison"][algo] = {
            name: float(rev[i]) for i, name in enumerate(["T1", "T2", "WT", "PV", "V1", "V2"])
        }
        summary["algorithm_comparison"][algo]["total"] = float(np.sum(rev))
    with open(os.path.join(save_dir, "results_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved benchmark outputs to {save_dir}")


if __name__ == "__main__":
    main()
