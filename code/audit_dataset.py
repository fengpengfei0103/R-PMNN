"""Audit the active reservoir datasets. / 审核当前使用的库水波动数据集。

The script is read-only: it checks metadata/image alignment, duplicated metadata,
image readability, and files that are not referenced by the active metadata.
本脚本只读：检查元数据与图像对齐、重复元数据、图像可读性以及未被引用的文件。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CODE_DIR.parent
DATASET_DIR = WORKSPACE_DIR / "dataset"

IMAGE_DIRS = {
    "x1": "SaturabilityTraincopy",
    "x3": "PorePressureTraincopy",
    "x4": "EffectiveStressTraincopy",
    "y1": "Saturabilitycopy",
    "y2": "PorePressurecopy",
    "y3": "EffectiveStresscopy",
}

DATA_SPLITS = {
    "train": {
        "metadata": DATASET_DIR / "H15-JS5-T2-50.xlsx",
        "image_root": DATASET_DIR,
    },
    "test": {
        "metadata": DATASET_DIR / "testfull" / "H15-JS5-T4-40.xlsx",
        "image_root": DATASET_DIR / "testfull",
    },
}


def clean_image_name(value: object) -> str:
    """Remove spreadsheet quoting around a filename. / 去除表格文件名外层引号。"""

    return str(value).strip().strip("'").strip('"')


def image_name_for_input(sample_name: str, key: str) -> str:
    """Map saturation names to pore/stress names. / 将饱和度名称映射到孔压或应力名称。"""

    if key in {"x3", "y2"}:
        return sample_name.replace("S_", "P_")
    if key in {"x4", "y3"}:
        return sample_name.replace("S_", "E_")
    return sample_name


def audit_split(split_name: str, verify_images: bool) -> dict[str, Any]:
    """Audit one configured split. / 审核一个已配置的数据划分。"""

    split = DATA_SPLITS[split_name]
    metadata_path = Path(split["metadata"])
    image_root = Path(split["image_root"])
    frame = pd.read_excel(metadata_path, sheet_name="Sheet2")
    names = [clean_image_name(value) for value in frame["name"]]

    result: dict[str, Any] = {
        "metadata": str(metadata_path.relative_to(WORKSPACE_DIR)),
        "rows": int(len(frame)),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_names": int(pd.Series(names).duplicated().sum()),
        "null_counts": {str(key): int(value) for key, value in frame.isna().sum().items()},
        "image_directories": {},
    }

    for key, directory_name in IMAGE_DIRS.items():
        directory = image_root / directory_name
        expected_names = {image_name_for_input(name, key) for name in names}
        actual_paths = {path.name: path for path in directory.glob("*.png")}
        missing = sorted(expected_names - actual_paths.keys())
        extra = sorted(actual_paths.keys() - expected_names)
        unreadable: list[dict[str, str]] = []
        image_shapes: Counter[str] = Counter()

        if verify_images:
            for name in sorted(expected_names & actual_paths.keys()):
                try:
                    with Image.open(actual_paths[name]) as image:
                        image_shapes[f"{image.width}x{image.height}/{image.mode}"] += 1
                        image.verify()
                except Exception as exc:  # pragma: no cover - depends on source files
                    unreadable.append({"name": name, "error": str(exc)})

        result["image_directories"][directory_name] = {
            "expected": len(expected_names),
            "actual": len(actual_paths),
            "missing": missing,
            "extra": extra,
            "unreadable": unreadable,
            "shapes": dict(sorted(image_shapes.items())),
        }
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments. / 构建命令行参数。"""

    parser = argparse.ArgumentParser(
        description="Audit metadata and images used by R_PMNN. / 审核 R_PMNN 使用的元数据和图像。"
    )
    parser.add_argument(
        "--skip-image-verify",
        action="store_true",
        help="Skip PNG decoding checks. / 跳过 PNG 解码检查。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path. / 可选的 JSON 报告路径。",
    )
    return parser


def main() -> None:
    """Run the audit and print JSON. / 运行审核并输出 JSON。"""

    args = build_parser().parse_args()
    report = {
        name: audit_split(name, verify_images=not args.skip_image_verify)
        for name in DATA_SPLITS
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
