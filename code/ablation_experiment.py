"""Run, and summarize R_PMNN ablations. / 运行并汇总 R_PMNN 消融实验。"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


ROOT_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = CODE_DIR / "configs"
CHECKPOINT_DIR = CODE_DIR / "checkpoints"
RESULTS_DIR = CODE_DIR / "results"
LOG_DIR = CODE_DIR / "logs"
SOURCE_CONFIG = CONFIG_DIR / "full_template.json"
SLOPE_EXPERIMENT = CODE_DIR / "slope_experiment.py"

BASE_FEATURE_COLUMNS = [
    "INITIAL_WATER_LEVEL",
    "DIRECTION",
    "RATE",
    "DURATION",
]
MECHANISM_FEATURE_COLUMNS = [
    "INITIAL_WATER_LEVEL",
    "DIRECTION",
    "RATE",
    "DURATION",
    "SIGNED_RATE",
    "DELTA_WATER_LEVEL",
    "FINAL_WATER_LEVEL",
]


@dataclass(frozen=True)
class AblationCase:
    """One M/S/L component combination. / 一种 M/S/L 组件组合。"""
    case_id: str
    display_name: str
    mechanism_features: bool
    physical_structure: bool
    weak_physics_loss: bool
    removal_role: str = ""
    incremental_role: str = ""


CASES: Dict[str, AblationCase] = {
    "base": AblationCase(
        case_id="base",
        display_name="Base",
        mechanism_features=False,
        physical_structure=False,
        weak_physics_loss=False,
        incremental_role="Base",
    ),
    "base_m": AblationCase(
        case_id="base_m",
        display_name="Base + M",
        mechanism_features=True,
        physical_structure=False,
        weak_physics_loss=False,
        incremental_role="Base + M",
    ),
    "wo_l": AblationCase(
        case_id="wo_l",
        display_name="Base + M + S / w/o L",
        mechanism_features=True,
        physical_structure=True,
        weak_physics_loss=False,
        removal_role="w/o L",
        incremental_role="Base + M + S",
    ),
    "full": AblationCase(
        case_id="full",
        display_name="Full",
        mechanism_features=True,
        physical_structure=True,
        weak_physics_loss=True,
        removal_role="Full",
        incremental_role="Full",
    ),
    "wo_m": AblationCase(
        case_id="wo_m",
        display_name="w/o M",
        mechanism_features=False,
        physical_structure=True,
        weak_physics_loss=True,
        removal_role="w/o M",
    ),
    "wo_s": AblationCase(
        case_id="wo_s",
        display_name="w/o S",
        mechanism_features=True,
        physical_structure=False,
        weak_physics_loss=True,
        removal_role="w/o S",
    ),
}

REMOVAL_ORDER = ["full", "wo_m", "wo_s", "wo_l"]
INCREMENTAL_ORDER = ["base", "base_m", "wo_l", "full"]
DEFAULT_RUN_ORDER = ["base", "base_m", "wo_l", "full", "wo_m", "wo_s"]


def rel_to_code(path: Path) -> str:
    """Return a portable path relative to ``code``. / 返回相对于 ``code`` 的可移植路径。"""

    return os.path.relpath(path, CODE_DIR).replace(os.sep, "/")


def load_source_config() -> dict:
    with SOURCE_CONFIG.open("r", encoding="utf-8") as f:
        return json.load(f)


def case_config_path(case_id: str) -> Path:
    return CONFIG_DIR / f"{case_id}.json"


def case_output_dir(case_id: str) -> Path:
    return RESULTS_DIR / case_id


def case_checkpoint_dir(case_id: str) -> Path:
    return CHECKPOINT_DIR / case_id


def case_checkpoint_path(case_id: str) -> Path:
    return case_checkpoint_dir(case_id) / f"R_PMNN_{case_id}_epoch_100.pth"


def build_case_config(case: AblationCase) -> dict:
    """Build one isolated experiment configuration. / 构建单个独立实验配置。"""

    config = load_source_config()
    config["action"] = "train,predict"
    config["model_types"] = ["r_pmnn"]
    config["checkpoint_prefix"] = f"R_PMNN_{case.case_id}"
    config["seed"] = 60
    config["feature_columns"] = (
        MECHANISM_FEATURE_COLUMNS if case.mechanism_features else BASE_FEATURE_COLUMNS
    )

    config["field_branch_skip_enabled"] = case.physical_structure
    config["field_residual_enabled"] = case.physical_structure
    config["pore_residual_scale"] = 0.25
    config["stress_residual_scale"] = 0.25

    if case.weak_physics_loss:
        config["delta_physics_loss_weight"] = 0.05
        config["pore_diffusion_loss_weight"] = 0.0001
    else:
        config["delta_physics_loss_weight"] = 0.0
        config["pore_diffusion_loss_weight"] = 0.0
    config["feature_normalization"] = {
        "enabled": True,
        "stats_path": rel_to_code(case_checkpoint_dir(case.case_id) / "reservoir_feature_stats.json"),
    }
    config["label_normalization"] = {
        "enabled": False,
        "stats_path": rel_to_code(case_checkpoint_dir(case.case_id) / "reservoir_label_stats.json"),
    }

    train_config = dict(config["train"])
    train_config["model_save_dir"] = rel_to_code(case_checkpoint_dir(case.case_id))
    config["train"] = train_config

    predict_config = dict(config["predict"])
    predict_config["output_dir"] = rel_to_code(case_output_dir(case.case_id))
    predict_config["output_prefix"] = case.case_id
    predict_config["model_path"] = rel_to_code(case_checkpoint_path(case.case_id))
    predict_config["save_sample_metrics"] = True
    predict_config["save_explainability_outputs"] = True
    config["predict"] = predict_config

    config["ablation"] = {
        "case_id": case.case_id,
        "display_name": case.display_name,
        "mechanism_features": case.mechanism_features,
        "physical_structure": case.physical_structure,
        "weak_physics_loss": case.weak_physics_loss,
        "definition": {
            "M": "mechanism-aware reservoir features",
            "S": "field residual prediction and pore/stress branch skip",
            "L": "delta physics loss and weak pore diffusion residual loss",
        },
    }
    return config


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def make_configs(case_ids: Iterable[str]) -> None:
    """Generate case configs and a manifest. / 生成各实验配置及清单。"""
    case_ids = list(case_ids)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source_config": str(SOURCE_CONFIG.relative_to(ROOT_DIR)),
        "case_order": DEFAULT_RUN_ORDER,
        "removal_order": REMOVAL_ORDER,
        "incremental_order": INCREMENTAL_ORDER,
        "cases": {},
    }
    for case_id in case_ids:
        case = CASES[case_id]
        config = build_case_config(case)
        path = case_config_path(case_id)
        write_json(path, config)
        manifest["cases"][case_id] = {
            "display_name": case.display_name,
            "mechanism_features": case.mechanism_features,
            "physical_structure": case.physical_structure,
            "weak_physics_loss": case.weak_physics_loss,
            "config": str(path.relative_to(ROOT_DIR)),
            "checkpoint": str(case_checkpoint_path(case_id).relative_to(ROOT_DIR)),
            "output_dir": str(case_output_dir(case_id).relative_to(ROOT_DIR)),
            "removal_role": case.removal_role,
            "incremental_role": case.incremental_role,
        }
    write_json(CONFIG_DIR / "manifest.json", manifest)
    print(f"Wrote {len(case_ids)} ablation configs to {CONFIG_DIR}")


def parse_cases(value: str) -> List[str]:
    if value == "all":
        return list(DEFAULT_RUN_ORDER)
    case_ids = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [case_id for case_id in case_ids if case_id not in CASES]
    if unknown:
        known = ", ".join(DEFAULT_RUN_ORDER)
        raise SystemExit(f"Unknown cases: {unknown}. Known cases: {known}")
    return case_ids


def run_command(command: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(" ".join(command))
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n" + "=" * 100 + "\n")
        log_file.write(" ".join(command) + "\n")
        process = subprocess.Popen(
            command,
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def run_cases(case_ids: Iterable[str], action: str, python_executable: str, skip_trained: bool) -> None:
    """Run selected cases sequentially. / 按顺序运行所选实验。"""
    make_configs(case_ids)
    for case_id in case_ids:
        checkpoint = case_checkpoint_path(case_id)
        effective_action = action
        if skip_trained and "train" in action and checkpoint.exists():
            effective_action = "predict" if "predict" in action else ""
        if not effective_action:
            print(f"Skipping {case_id}: checkpoint already exists at {checkpoint}")
            continue
        command = [
            python_executable,
            str(SLOPE_EXPERIMENT),
            "--config",
            str(case_config_path(case_id)),
            "--action",
            effective_action,
        ]
        return_code = run_command(command, LOG_DIR / f"{case_id}.log")
        if return_code != 0:
            raise SystemExit(f"Case {case_id} failed with return code {return_code}")


def read_metric_file(path: Path) -> dict:
    import pandas as pd

    frame = pd.read_excel(path)
    return {str(row["metric"]): float(row["mean"]) for _, row in frame.iterrows()}


def add_sample_metric_summary(row: dict, case_id: str, prefix: str, long_duration_threshold: float) -> None:
    import pandas as pd

    path = case_output_dir(case_id) / f"{prefix}_sample_metrics_results.xlsx"
    if not path.exists():
        return
    frame = pd.read_excel(path)
    if "DURATION" in frame.columns:
        long_frame = frame[frame["DURATION"] >= long_duration_threshold]
        row["long_sample_count"] = int(len(long_frame))
        for metric in ("task1_psnr", "task2_psnr", "task3_psnr", "task5_mae"):
            if metric in long_frame.columns and len(long_frame) > 0:
                row[f"long_{metric}"] = float(long_frame[metric].mean())
    for metric in ("task1_psnr", "task2_psnr", "task3_psnr", "task5_mae"):
        if metric in frame.columns:
            row[f"sample_std_{metric}"] = float(frame[metric].std(ddof=0))


def add_explainability_summary(row: dict, case_id: str, prefix: str, long_duration_threshold: float) -> None:
    import pandas as pd

    path = case_output_dir(case_id) / f"{prefix}_explainability_results.xlsx"
    if not path.exists():
        return
    frame = pd.read_excel(path)
    for metric in (
        "pore_delta_mae",
        "stress_delta_mae",
        "pore_delta_mean_error",
        "stress_delta_mean_error",
        "pore_rgb_delta_mae",
        "stress_rgb_delta_mae",
    ):
        if metric in frame.columns:
            row[metric] = float(frame[metric].mean())
    for metric in ("pore_delta_mae", "stress_delta_mae"):
        if metric in frame.columns and "DURATION" in frame.columns:
            long_frame = frame[frame["DURATION"] >= long_duration_threshold]
            if len(long_frame) > 0:
                row[f"long_{metric}"] = float(long_frame[metric].mean())


def summarize_results(long_duration_threshold: float = 30.0) -> None:
    """Aggregate completed metrics into comparison tables. / 汇总已完成指标为对比表。"""
    import pandas as pd

    rows = []
    for case_id in DEFAULT_RUN_ORDER:
        case = CASES[case_id]
        prefix = case_id
        metrics_path = case_output_dir(case_id) / f"{prefix}_metrics_results_summary.xlsx"
        row = {
            "case_id": case_id,
            "display_name": case.display_name,
            "M_mechanism_features": case.mechanism_features,
            "S_physical_structure": case.physical_structure,
            "L_weak_physics_loss": case.weak_physics_loss,
            "removal_role": case.removal_role,
            "incremental_role": case.incremental_role,
            "status": "missing",
        }
        if metrics_path.exists():
            row.update(read_metric_file(metrics_path))
            row["status"] = "complete"
            add_sample_metric_summary(row, case_id, prefix, long_duration_threshold)
            add_explainability_summary(row, case_id, prefix, long_duration_threshold)
        rows.append(row)

    summary = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_csv = RESULTS_DIR / "ablation_all_cases_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    removal = summary.set_index("case_id").loc[REMOVAL_ORDER].reset_index()
    incremental = summary.set_index("case_id").loc[INCREMENTAL_ORDER].reset_index()
    removal.to_csv(RESULTS_DIR / "ablation_removal_table.csv", index=False, encoding="utf-8-sig")
    incremental.to_csv(RESULTS_DIR / "ablation_incremental_table.csv", index=False, encoding="utf-8-sig")

    try:
        with pd.ExcelWriter(RESULTS_DIR / "ablation_summary.xlsx") as writer:
            summary.to_excel(writer, sheet_name="all_cases", index=False)
            removal.to_excel(writer, sheet_name="removal", index=False)
            incremental.to_excel(writer, sheet_name="incremental", index=False)
    except Exception as exc:
        print(f"Excel summary was not written: {exc}")

    print(f"Wrote summary to {summary_csv}")
    print(f"Wrote removal table to {RESULTS_DIR / 'ablation_removal_table.csv'}")
    print(f"Wrote incremental table to {RESULTS_DIR / 'ablation_incremental_table.csv'}")


def list_cases() -> None:
    print("Unique ablation cases:")
    for case_id in DEFAULT_RUN_ORDER:
        case = CASES[case_id]
        print(
            f"{case_id:8s} M={int(case.mechanism_features)} "
            f"S={int(case.physical_structure)} L={int(case.weak_physics_loss)} "
            f"{case.display_name}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reservoir fluctuation ablation experiment runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List ablation cases.")

    make_parser = subparsers.add_parser("make-configs", help="Generate isolated config files.")
    make_parser.add_argument("--cases", default="all", help="Comma-separated case ids or 'all'.")

    run_parser = subparsers.add_parser("run", help="Run training and/or prediction for selected cases.")
    run_parser.add_argument("--cases", default="all", help="Comma-separated case ids or 'all'.")
    run_parser.add_argument(
        "--action",
        default="train,predict",
        choices=["train", "predict", "train,predict"],
        help="Action passed to code/slope_experiment.py. / 传给训练入口的动作。",
    )
    run_parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    run_parser.add_argument(
        "--skip-trained",
        action="store_true",
        help="If checkpoint exists, skip the train stage and run predict only.",
    )

    summarize_parser = subparsers.add_parser("summarize", help="Aggregate completed ablation results.")
    summarize_parser.add_argument(
        "--long-duration-threshold",
        type=float,
        default=30.0,
        help="DURATION threshold for long-term subset metrics.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "list":
        list_cases()
    elif args.command == "make-configs":
        make_configs(parse_cases(args.cases))
    elif args.command == "run":
        run_cases(parse_cases(args.cases), args.action, args.python, args.skip_trained)
    elif args.command == "summarize":
        summarize_results(args.long_duration_threshold)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
