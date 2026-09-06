"""Config-driven R_PMNN training and prediction. / 基于配置的 R_PMNN 训练与预测。"""

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CODE_DIR.parent

MODEL_REGISTRY = {
    "r_pmnn": {
        "script": "R_PMNN.py",
        "checkpoint": "R_PMNN_full_epoch_100.pth",
        "prefix": "R_PMNN",
    },
}

DEFAULT_IMAGE_DIRS = {
    "x1": "SaturabilityTraincopy",
    "x3": "PorePressureTraincopy",
    "x4": "EffectiveStressTraincopy",
    "y1": "Saturabilitycopy",
    "y2": "PorePressurecopy",
    "y3": "EffectiveStresscopy",
}

RESERVOIR_BASE_FEATURES = ("INITIAL_WATER_LEVEL", "DIRECTION", "RATE", "DURATION")
RESERVOIR_DERIVED_FEATURES = ("SIGNED_RATE", "DELTA_WATER_LEVEL", "FINAL_WATER_LEVEL")

TASK_OUTPUT_DIRS = {
    "task1": "saturation",
    "task2": "pore_pressure",
    "task3": "effective_stress",
}

def resolve_path(value, base_dir=CODE_DIR):
    """Resolve paths relative to the code directory. / 将相对路径解析到代码目录。"""
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_config(config_path):
    path = resolve_path(config_path, Path.cwd())
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config["_config_path"] = str(path)
    return config


def configured_seed(config, data_config=None):
    seed = None
    if data_config:
        seed = data_config.get("seed")
    if seed is None:
        seed = config.get("seed")
    return None if seed is None else int(seed)


def set_random_seed(seed, module=None):
    """Configure reproducible random states. / 配置可复现的随机状态。"""
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    if module is None:
        return
    torch = module.torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except Exception:
        pass


def resolve_model_type(config):
    explicit = config.get("model_type")
    if explicit:
        if explicit not in MODEL_REGISTRY:
            known = ", ".join(sorted(MODEL_REGISTRY))
            raise ValueError(f"Unknown model_type '{explicit}'. Known values: {known}")
        return explicit
    return "r_pmnn"


def load_model_module(model_type):
    """Load the selected model implementation. / 加载所选模型实现。"""
    model_info = MODEL_REGISTRY[model_type]
    module_path = CODE_DIR / model_info["script"]
    module_name = f"_slope_model_{model_type}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load legacy module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def patch_feature_modules(module, feature_dim):
    """Match feature layers to each ablation case. / 使特征层适配各消融实验。"""
    if feature_dim == getattr(module, "SLOPE_FEATURE_DIM", 2):
        return

    nn = module.nn

    class FeatureModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(feature_dim, 20 * 35 * 8)
            self.relu = nn.ReLU()
            self.reshape = lambda x: x.view(-1, 8, 20, 35)
            self.ConvNorm1 = module.ConvNorm(
                in_channels=8,
                out_channels=8,
                kernel_size=3,
                stride=1,
                leaky=False,
                norm="INSTANCE",
                activation=True,
            )
            self.encoder_level = nn.Sequential(
                *[
                    module.TransformerBlock(
                        dim=8,
                        num_heads=1,
                        ffn_expansion_factor=2.66,
                        bias=False,
                        LayerNorm_type="WithBias",
                    )
                    for _ in range(4)
                ]
            )
            self.conv_blocks = nn.Sequential(
                nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False),
                self.encoder_level,
                nn.InstanceNorm2d(8),
                nn.ReLU(),
            )
            self.ConvNorm2 = module.ConvNorm(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                stride=1,
                leaky=False,
                norm="INSTANCE",
                activation=True,
            )

        def forward(self, value):
            x = self.fc(value)
            x = self.relu(x)
            x = self.reshape(x)
            x = self.ConvNorm1(x)
            x = self.conv_blocks(x)
            return self.ConvNorm2(x)

    class FeatureModuleStab(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(feature_dim, 40 * 70 * 2)
            self.relu = nn.ReLU()
            self.reshape = lambda x: x.view(-1, 2, 40, 70)
            self.ConvNorm1 = module.ConvNorm(
                in_channels=2,
                out_channels=8,
                kernel_size=3,
                stride=1,
                leaky=False,
                norm="INSTANCE",
                activation=True,
            )
            self.encoder_level = nn.Sequential(
                *[
                    module.TransformerBlock(
                        dim=8,
                        num_heads=1,
                        ffn_expansion_factor=2.66,
                        bias=False,
                        LayerNorm_type="WithBias",
                    )
                    for _ in range(4)
                ]
            )
            self.conv_blocks = nn.Sequential(
                nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False),
                self.encoder_level,
                nn.InstanceNorm2d(8),
                nn.ReLU(),
            )
            self.ConvNorm2 = module.ConvNorm(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                stride=1,
                leaky=False,
                norm="INSTANCE",
                activation=True,
            )
            self.fc2 = nn.Sequential(
                nn.Linear(feature_dim, 40 * 70 * 2),
                nn.ReLU(),
                nn.Linear(40 * 70 * 2, 2),
            )

        def forward(self, value):
            x = self.fc(value)
            x = self.relu(x)
            x = self.reshape(x)
            x = self.ConvNorm1(x)
            x1 = self.conv_blocks(x)
            x1 = self.ConvNorm2(x1)
            x2 = self.fc2(value)
            return x1, x2

    module.FeatureModule = FeatureModule
    module.FeatureModuleStab = FeatureModuleStab


def read_metadata(module, data_config):
    """Read and concatenate metadata workbooks. / 读取并合并元数据工作簿。"""
    sheet_name = data_config.get("sheet_name", "Sheet2")
    files = data_config.get("metadata_files")
    if files is None:
        files = [data_config["metadata_file"]]
    elif isinstance(files, str):
        files = [files]
    frames = []
    for file_name in files:
        path = resolve_path(file_name)
        frame = module.pd.read_excel(path, sheet_name=sheet_name)
        frame["_metadata_file"] = str(path)
        frames.append(frame)
    return module.pd.concat(frames, ignore_index=True)


def apply_column_aliases(frame, config, data_config):
    aliases = {}
    aliases.update(config.get("column_aliases", {}))
    aliases.update(data_config.get("column_aliases", {}))
    for target, source in aliases.items():
        if target not in frame.columns and source in frame.columns:
            frame[target] = frame[source]
    return frame


def add_reservoir_feature_columns(frame):
    """Add mechanism-aware features. / 添加机理感知的库水运行特征。"""
    if {"DIRECTION", "RATE"}.issubset(frame.columns):
        frame["SIGNED_RATE"] = frame["DIRECTION"].astype(float) * frame["RATE"].astype(float)
    if {"SIGNED_RATE", "DURATION"}.issubset(frame.columns):
        frame["DELTA_WATER_LEVEL"] = frame["SIGNED_RATE"].astype(float) * frame["DURATION"].astype(float)
    if {"INITIAL_WATER_LEVEL", "DELTA_WATER_LEVEL"}.issubset(frame.columns):
        frame["FINAL_WATER_LEVEL"] = (
            frame["INITIAL_WATER_LEVEL"].astype(float) + frame["DELTA_WATER_LEVEL"].astype(float)
        )
    return frame


def feature_normalization_enabled(config):
    return bool(config.get("feature_normalization", {}).get("enabled", False))


def label_normalization_enabled(config):
    if config.get("numeric_output_mode") == "legacy_fr_index":
        return False
    return bool(config.get("label_normalization", {}).get("enabled", False))


def feature_stats_path(config, data_config=None):
    norm_config = config.get("feature_normalization", {})
    configured = norm_config.get("stats_path")
    if configured:
        return resolve_path(configured)
    train_config = config.get("train", {})
    model_save_dir = train_config.get("model_save_dir", "model_snapshot")
    return resolve_path(model_save_dir) / "reservoir_feature_stats.json"


def label_stats_path(config, data_config=None):
    norm_config = config.get("label_normalization", {})
    configured = norm_config.get("stats_path")
    if configured:
        return resolve_path(configured)
    train_config = config.get("train", {})
    model_save_dir = train_config.get("model_save_dir", "model_snapshot")
    return resolve_path(model_save_dir) / "reservoir_label_stats.json"


def compute_feature_stats(frame, feature_columns):
    values = frame[feature_columns].astype(float)
    mean = values.mean()
    std = values.std(ddof=0).replace(0, 1.0)
    return {
        "feature_columns": list(feature_columns),
        "mean": {column: float(mean[column]) for column in feature_columns},
        "std": {column: float(std[column]) for column in feature_columns},
    }


def compute_label_stats(frame, label_column):
    values = frame[label_column].astype(float)
    std = float(values.std(ddof=0)) or 1.0
    return {
        "label_column": label_column,
        "mean": float(values.mean()),
        "std": std,
    }


def save_feature_stats(stats, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def load_feature_stats(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_label_stats(stats, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def load_label_stats(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def apply_feature_normalization(frame, feature_columns, stats):
    normalized = frame.copy()
    for column in feature_columns:
        mean = float(stats["mean"][column])
        std = float(stats["std"][column]) or 1.0
        normalized[column] = (normalized[column].astype(float) - mean) / std
    return normalized


def apply_label_normalization(values, stats):
    return (values.astype(float) - float(stats["mean"])) / (float(stats["std"]) or 1.0)


def inverse_label_normalization(values, stats):
    return values * (float(stats["std"]) or 1.0) + float(stats["mean"])


def clean_image_name(value):
    return str(value).strip().strip("'").strip('"')


def transform_image_name(name, key, config, data_config):
    name = clean_image_name(name)
    replacements = data_config.get("name_replacements", {})
    replacements = {
        **config.get("name_replacements", {}),
        **replacements,
    }
    for old, new in replacements.get(key, []):
        name = name.replace(old, new)
    return name


def validate_columns(frame, columns, context):
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        available = ", ".join(map(str, frame.columns))
        raise ValueError(f"{context} missing columns {missing}. Available columns: {available}")


def make_dataset(module, frame, feature_frame, label_values, config, data_config, feature_columns, label_column, name_column):
    """Create a metadata-aligned image dataset. / 创建与元数据严格对齐的图像数据集。"""
    image_root = resolve_path(data_config["image_root"])
    image_dirs = dict(DEFAULT_IMAGE_DIRS)
    image_dirs.update(data_config.get("image_dirs", {}))
    transform = getattr(module, "transform", None)

    class AlignedSlopeDataset(module.Dataset):
        def __init__(self):
            self.frame = frame.reset_index(drop=True)
            self.feature_values = feature_frame.reset_index(drop=True)[feature_columns].astype(float).to_numpy()
            self.label_values = np.asarray(label_values, dtype=float)
            self.paths = {
                key: self._resolve_image_paths(key, image_root / rel_dir)
                for key, rel_dir in image_dirs.items()
            }

        def _resolve_image_paths(self, key, directory):
            if name_column and name_column in self.frame.columns:
                paths = [
                    directory / transform_image_name(name, key, config, data_config)
                    for name in self.frame[name_column]
                ]
            else:
                paths = sorted(directory.glob("*.png"), key=lambda item: item.name)
            missing = [str(path) for path in paths if not path.exists()]
            if missing:
                sample = "\n".join(missing[:5])
                raise FileNotFoundError(f"Missing image files in {directory}:\n{sample}")
            if len(paths) != len(self.frame):
                raise ValueError(
                    f"Image count mismatch for {directory}: {len(paths)} images, "
                    f"{len(self.frame)} metadata rows"
                )
            return paths

        def __len__(self):
            return len(self.frame)

        def __getitem__(self, idx):
            image = module.Image
            with image.open(self.paths["x1"][idx]) as img:
                x1 = img.convert("RGB")
            with image.open(self.paths["x3"][idx]) as img:
                x3 = img.convert("RGB")
            with image.open(self.paths["x4"][idx]) as img:
                x4 = img.convert("RGB")
            with image.open(self.paths["y1"][idx]) as img:
                y1 = img.convert("RGB")
            with image.open(self.paths["y2"][idx]) as img:
                y2 = img.convert("RGB")
            with image.open(self.paths["y3"][idx]) as img:
                y3 = img.convert("RGB")
            if transform:
                x1 = transform(x1)
                x3 = transform(x3)
                x4 = transform(x4)
                y1 = transform(y1)
                y2 = transform(y2)
                y3 = transform(y3)
            return (
                x1,
                self.feature_values[idx],
                x3,
                x4,
                y1,
                y2,
                y3,
                self.label_values[idx],
            )

    return AlignedSlopeDataset()


def build_dataset(module, config, data_config, feature_columns, label_column, name_column, fit_feature_stats=False):
    """Validate and normalize one dataset split. / 校验并归一化一个数据划分。"""
    frame = read_metadata(module, data_config)
    frame = apply_column_aliases(frame, config, data_config)
    frame = add_reservoir_feature_columns(frame)
    validate_columns(frame, feature_columns + [label_column], "Metadata")
    if name_column:
        validate_columns(frame, [name_column], "Metadata")
    feature_frame = frame
    if feature_normalization_enabled(config):
        stats_path = feature_stats_path(config, data_config)
        if fit_feature_stats:
            stats = compute_feature_stats(frame, feature_columns)
            save_feature_stats(stats, stats_path)
            print(f"Saved feature normalization stats to {stats_path}")
        elif stats_path.exists():
            stats = load_feature_stats(stats_path)
        else:
            stats = compute_feature_stats(frame, feature_columns)
            print(f"Warning: feature stats not found at {stats_path}; using current metadata statistics.")
        if stats.get("feature_columns") != list(feature_columns):
            raise ValueError(
                f"Feature stats columns {stats.get('feature_columns')} do not match config columns {feature_columns}"
            )
        module.FEATURE_NORMALIZATION_STATS = stats
        feature_frame = apply_feature_normalization(frame, feature_columns, stats)
    else:
        module.FEATURE_NORMALIZATION_STATS = None
    label_values = frame[label_column].astype(float).to_numpy()
    if label_normalization_enabled(config):
        stats_path = label_stats_path(config, data_config)
        if fit_feature_stats:
            label_stats = compute_label_stats(frame, label_column)
            save_label_stats(label_stats, stats_path)
            print(f"Saved label normalization stats to {stats_path}")
        elif stats_path.exists():
            label_stats = load_label_stats(stats_path)
        else:
            label_stats = compute_label_stats(frame, label_column)
            print(f"Warning: label stats not found at {stats_path}; using current metadata statistics.")
        if label_stats.get("label_column") != label_column:
            raise ValueError(
                f"Label stats column {label_stats.get('label_column')} does not match config label column {label_column}"
            )
        module.LABEL_NORMALIZATION_STATS = label_stats
        label_values = apply_label_normalization(frame[label_column], label_stats).to_numpy()
    else:
        module.LABEL_NORMALIZATION_STATS = None
    dataset = make_dataset(
        module,
        frame,
        feature_frame,
        label_values,
        config,
        data_config,
        feature_columns,
        label_column,
        name_column,
    )
    return frame, dataset


def run_train(module, config):
    """Train one configured ablation case. / 训练一个已配置的消融实验。"""
    train_config = config["train"]
    seed = configured_seed(config, train_config)
    feature_columns = config["feature_columns"]
    label_column = config.get("label_column", "StabilizationFactor")
    name_column = config.get("name_column", "name")
    _, dataset = build_dataset(
        module,
        config,
        train_config,
        feature_columns,
        label_column,
        name_column,
        fit_feature_stats=True,
    )
    model_save_dir = resolve_path(train_config.get("model_save_dir", "model_snapshot"))
    module.STABILITY_LOSS_WEIGHT = float(config.get("stability_loss_weight", 1.0))
    module.IMAGE_LOSS_WEIGHTS = config.get(
        "image_loss_weights",
        {"saturation": 1.0, "pore_pressure": 1.5, "effective_stress": 2.0},
    )
    module.GRADIENT_LOSS_WEIGHT = float(config.get("gradient_loss_weight", 0.2))
    module.TV_LOSS_WEIGHT = float(config.get("tv_loss_weight", 0.05))
    module.SOIL_MASK_PATH = str(resolve_path(config.get("soil_mask_path", "../dataset/mask.png")))
    module.PHYSICS_WARMUP_EPOCHS = int(config.get("physics_warmup_epochs", 20))
    module.DELTA_PHYSICS_LOSS_WEIGHT = float(config.get("delta_physics_loss_weight", 0.0))
    module.PORE_DIFFUSION_LOSS_WEIGHT = float(config.get("pore_diffusion_loss_weight", 0.0))
    module.PORE_DIFFUSIVITY = float(config.get("pore_diffusivity", 0.01))
    module.FIELD_BRANCH_SKIP_ENABLED = bool(config.get("field_branch_skip_enabled", True))
    module.FIELD_RESIDUAL_ENABLED = bool(config.get("field_residual_enabled", True))
    module.PORE_RESIDUAL_SCALE = float(config.get("pore_residual_scale", 0.25))
    module.STRESS_RESIDUAL_SCALE = float(config.get("stress_residual_scale", 0.25))
    module.CHECKPOINT_PREFIX = str(config.get("checkpoint_prefix", "R_PMNN"))
    set_random_seed(seed, module)
    if seed is not None:
        print(f"Using random seed: {seed}")
    module.train_model(
        datasetTrain=dataset,
        model_save_dir=str(model_save_dir),
        batch_size=int(train_config.get("batch_size", 4)),
        epochs=int(train_config.get("epochs", 100)),
        lr=float(train_config.get("lr", 0.001)),
        checkpoint_prefix=str(config.get("checkpoint_prefix", "R_PMNN")),
    )


def image_to_uint8(module, tensor):
    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = np.clip(image, 0.0, 1.0)
    return (image * 255).astype(np.uint8)


def scalar_field_numpy(tensor):
    field = tensor.detach().cpu().float().numpy()
    if field.ndim == 3 and field.shape[0] >= 3:
        return 0.299 * field[0] + 0.587 * field[1] + 0.114 * field[2]
    if field.ndim == 3:
        return field.mean(axis=0)
    return field


def save_signed_field_map(module, tensor, path, limit):
    values = scalar_field_numpy(tensor)
    limit = max(float(limit), 1e-6)
    module.plt.imsave(str(path), values, cmap="coolwarm", vmin=-limit, vmax=limit)


def save_abs_field_map(module, tensor, path, limit):
    values = np.abs(scalar_field_numpy(tensor))
    limit = max(float(limit), 1e-6)
    module.plt.imsave(str(path), values, cmap="inferno", vmin=0.0, vmax=limit)


def delta_field_stats(prefix, pred_delta, true_delta):
    diff = pred_delta - true_delta
    diff_rgb = diff.detach().cpu().float().numpy()
    pred_scalar = scalar_field_numpy(pred_delta)
    true_scalar = scalar_field_numpy(true_delta)
    diff_scalar = pred_scalar - true_scalar
    return {
        f"{prefix}_pred_delta_mean": float(pred_scalar.mean()),
        f"{prefix}_true_delta_mean": float(true_scalar.mean()),
        f"{prefix}_delta_mean_error": float(diff_scalar.mean()),
        f"{prefix}_delta_mae": float(np.mean(np.abs(diff_scalar))),
        f"{prefix}_delta_mse": float(np.mean(diff_scalar ** 2)),
        f"{prefix}_delta_max_abs_error": float(np.max(np.abs(diff_scalar))),
        f"{prefix}_pred_delta_min": float(pred_scalar.min()),
        f"{prefix}_pred_delta_max": float(pred_scalar.max()),
        f"{prefix}_true_delta_min": float(true_scalar.min()),
        f"{prefix}_true_delta_max": float(true_scalar.max()),
        f"{prefix}_rgb_delta_mae": float(np.mean(np.abs(diff_rgb))),
    }


def numeric_values(module, predictions, truths, mode):
    if mode == "legacy_fr_index":
        pred = np.round(module.value_to_Fr(predictions), 5)
        true = np.round(module.value_to_Fr(truths), 5)
    elif mode == "direct":
        pred = predictions.detach().cpu().numpy()
        true = truths.detach().cpu().numpy()
        label_stats = getattr(module, "LABEL_NORMALIZATION_STATS", None)
        if label_stats:
            pred = inverse_label_normalization(pred, label_stats)
            true = inverse_label_normalization(true, label_stats)
    else:
        raise ValueError("numeric_output_mode must be 'direct' or 'legacy_fr_index'")
    return np.asarray(pred, dtype=float).reshape(-1), np.asarray(true, dtype=float).reshape(-1)


def adapt_state_dict_feature_inputs(module, model, state_dict):
    """Adapt compatible historical checkpoints. / 适配结构兼容的历史检查点。"""
    model_state = model.state_dict()
    adapted_keys = []
    if "log_var_img" in state_dict and "log_var1" in model_state and "log_var1" not in state_dict:
        state_dict["log_var1"] = state_dict["log_var_img"].clone()
        adapted_keys.append("log_var1")
    if "log_var_stability" in state_dict and "log_var2" in model_state and "log_var2" not in state_dict:
        state_dict["log_var2"] = state_dict["log_var_stability"].clone()
        adapted_keys.append("log_var2")
    if "log_var_sat" in state_dict and "log_var_img" in model_state and "log_var_img" not in state_dict:
        state_dict["log_var_img"] = state_dict["log_var_sat"].clone()
        adapted_keys.append("log_var_img")
    if "log_var1" in state_dict:
        legacy_targets = ("log_var_img", "log_var_pore", "log_var_stress")
        for target_key in legacy_targets:
            if target_key in model_state and target_key not in state_dict:
                state_dict[target_key] = state_dict["log_var1"].clone()
                adapted_keys.append(target_key)
    if "log_var2" in state_dict and "log_var_stability" in model_state and "log_var_stability" not in state_dict:
        state_dict["log_var_stability"] = state_dict["log_var2"].clone()
        adapted_keys.append("log_var_stability")
    if "final_conv.weight" in state_dict and "final_conv_sat.weight" in model_state:
        for suffix in ("weight", "bias"):
            source_key = f"final_conv.{suffix}"
            if source_key not in state_dict:
                continue
            for target_name in ("final_conv_sat", "final_conv_pore", "final_conv_stress"):
                target_key = f"{target_name}.{suffix}"
                if target_key in model_state and target_key not in state_dict:
                    state_dict[target_key] = state_dict[source_key].clone()
                    adapted_keys.append(target_key)
            state_dict.pop(source_key, None)
    if "decoderConvTrans.0.weight" in state_dict and "decoderConvTrans.1.weight" in model_state:
        state_dict["decoderConvTrans.1.weight"] = state_dict.pop("decoderConvTrans.0.weight")
        adapted_keys.append("decoderConvTrans.1.weight")
    if "decoderConvTrans.0.bias" not in model_state:
        state_dict.pop("decoderConvTrans.0.bias", None)
    if "decoderConvTrans.1.weight" in state_dict and "decoderConvTrans.0.weight" in model_state:
        state_dict["decoderConvTrans.0.weight"] = state_dict.pop("decoderConvTrans.1.weight")
        adapted_keys.append("decoderConvTrans.0.weight")
        if "decoderConvTrans.0.bias" in model_state and "decoderConvTrans.0.bias" not in state_dict:
            state_dict["decoderConvTrans.0.bias"] = model_state["decoderConvTrans.0.bias"].clone()
            adapted_keys.append("decoderConvTrans.0.bias")
    for key, value in list(state_dict.items()):
        if key not in model_state:
            state_dict.pop(key, None)
            continue
        target = model_state[key]
        if value.shape == target.shape:
            continue
        can_reduce_output_channels = (
            value.ndim >= 1
            and target.ndim == value.ndim
            and value.shape[0] > target.shape[0]
            and target.shape[0] == 1
            and value.shape[1:] == target.shape[1:]
        )
        if can_reduce_output_channels:
            state_dict[key] = value.mean(dim=0, keepdim=True)
            adapted_keys.append(key)
            continue
        can_expand_output_channels = (
            value.ndim >= 1
            and target.ndim == value.ndim
            and value.shape[0] == 1
            and target.shape[0] > 1
            and value.shape[1:] == target.shape[1:]
        )
        if can_expand_output_channels:
            state_dict[key] = value.expand_as(target).clone()
            adapted_keys.append(key)
            continue
        can_pad_feature_axis = (
            value.ndim == 2
            and target.ndim == 2
            and value.shape[0] == target.shape[0]
            and value.shape[1] < target.shape[1]
        )
        if can_pad_feature_axis:
            padded = target.clone()
            padded[:, : value.shape[1]] = value
            padded[:, value.shape[1] :] = 0
            state_dict[key] = padded
            adapted_keys.append(key)
    if adapted_keys:
        print(
            "Adapted checkpoint feature-input weights for expanded reservoir features: "
            + ", ".join(adapted_keys)
        )
    return state_dict


def run_predict(module, config, model_type):
    """Predict fields/stability and write metrics. / 预测物理场与稳定性并写出指标。"""
    predict_config = config["predict"]
    feature_columns = config["feature_columns"]
    label_column = config.get("label_column", "StabilizationFactor")
    name_column = config.get("name_column", "name")
    frame, dataset = build_dataset(module, config, predict_config, feature_columns, label_column, name_column)

    torch = module.torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.FIELD_BRANCH_SKIP_ENABLED = bool(config.get("field_branch_skip_enabled", True))
    module.FIELD_RESIDUAL_ENABLED = bool(config.get("field_residual_enabled", True))
    module.PORE_RESIDUAL_SCALE = float(config.get("pore_residual_scale", 0.25))
    module.STRESS_RESIDUAL_SCALE = float(config.get("stress_residual_scale", 0.25))
    model = module.R_PMNN().to(device)
    default_checkpoint = CODE_DIR / "model_snapshot" / MODEL_REGISTRY[model_type]["checkpoint"]
    train_epochs = int(config.get("train", {}).get("epochs", 100))
    final_checkpoint_name = MODEL_REGISTRY[model_type]["checkpoint"].replace(
        "_epoch_100.pth",
        f"_epoch_{train_epochs}.pth",
    )
    final_checkpoint = CODE_DIR / "model_snapshot" / final_checkpoint_name
    if train_epochs != 100 and final_checkpoint.exists():
        default_checkpoint = final_checkpoint
    model_path = resolve_path(predict_config.get("model_path", str(default_checkpoint)))
    try:
        state_dict = torch.load(str(model_path), weights_only=True, map_location=device)
    except TypeError:
        state_dict = torch.load(str(model_path), map_location=device)
    state_dict = adapt_state_dict_feature_inputs(module, model, state_dict)
    model.load_state_dict(state_dict)
    model.eval()

    dataloader = module.DataLoader(
        dataset,
        batch_size=int(predict_config.get("batch_size", 1)),
        shuffle=False,
    )
    psnr_metric = module.PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = module.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    output_dir = resolve_path(predict_config.get("output_dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = predict_config.get("output_prefix", MODEL_REGISTRY[model_type]["prefix"])
    numeric_mode = predict_config.get("numeric_output_mode", config.get("numeric_output_mode", "direct"))
    save_sample_metrics = bool(predict_config.get("save_sample_metrics", True))
    save_explainability = bool(predict_config.get("save_explainability_outputs", True))
    pore_delta_limit = float(predict_config.get("pore_delta_visual_limit", config.get("pore_residual_scale", 0.25)))
    stress_delta_limit = float(
        predict_config.get("stress_delta_visual_limit", config.get("stress_residual_scale", 0.25))
    )

    metrics = {
        "task1_psnr": [],
        "task1_ssim": [],
        "task2_psnr": [],
        "task2_ssim": [],
        "task3_psnr": [],
        "task3_ssim": [],
        "task5_mse": [],
        "task5_mae": [],
    }
    prediction_rows = []
    sample_metric_rows = []
    explainability_rows = []
    sample_index = 0

    with torch.no_grad():
        for batch in dataloader:
            x1, x2, x3, x4, y1, y2, y3, y5 = batch
            x1 = x1.to(device)
            x2 = x2.unsqueeze(1).to(device)
            x3 = x3.to(device)
            x4 = x4.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)
            y3 = y3.to(device)
            y5 = y5.to(device)

            outputs1, outputs2, outputs3, outputs5 = model(x1, x2, x3, x4)

            pred_values, true_values = numeric_values(module, outputs5, y5, numeric_mode)

            batch_size = x1.size(0)
            for offset in range(batch_size):
                row = frame.iloc[sample_index + offset]
                if name_column and name_column in frame.columns:
                    stem = Path(clean_image_name(row[name_column])).stem
                else:
                    stem = f"{sample_index + offset + 1:04d}"

                for task, output in (
                    ("task1", outputs1[offset]),
                    ("task2", outputs2[offset]),
                    ("task3", outputs3[offset]),
                ):
                    task_dir = output_dir / f"{prefix}_{TASK_OUTPUT_DIRS[task]}"
                    task_dir.mkdir(parents=True, exist_ok=True)
                    module.plt.imsave(str(task_dir / f"prediction_{stem}.png"), image_to_uint8(module, output))

                sample_metrics = {}
                for task, output, target in (
                    ("task1", outputs1[offset], y1[offset]),
                    ("task2", outputs2[offset], y2[offset]),
                    ("task3", outputs3[offset], y3[offset]),
                ):
                    psnr_value = psnr_metric(output.unsqueeze(0), target.unsqueeze(0)).detach().cpu().item()
                    ssim_value = ssim_metric(output.unsqueeze(0), target.unsqueeze(0)).detach().cpu().item()
                    metrics[f"{task}_psnr"].append(psnr_value)
                    metrics[f"{task}_ssim"].append(ssim_value)
                    sample_metrics[f"{task}_psnr"] = psnr_value
                    sample_metrics[f"{task}_ssim"] = ssim_value

                task5_mse = float((pred_values[offset] - true_values[offset]) ** 2)
                task5_mae = float(abs(pred_values[offset] - true_values[offset]))
                metrics["task5_mse"].append(task5_mse)
                metrics["task5_mae"].append(task5_mae)
                sample_metrics["task5_mse"] = task5_mse
                sample_metrics["task5_mae"] = task5_mae
                sample_metrics["task5_error"] = float(pred_values[offset] - true_values[offset])

                pore_pred_delta = outputs2[offset] - x3[offset]
                pore_true_delta = y2[offset] - x3[offset]
                stress_pred_delta = outputs3[offset] - x4[offset]
                stress_true_delta = y3[offset] - x4[offset]
                explainability_stats = {}
                explainability_stats.update(delta_field_stats("pore", pore_pred_delta, pore_true_delta))
                explainability_stats.update(delta_field_stats("stress", stress_pred_delta, stress_true_delta))

                if save_explainability:
                    explainability_dir = output_dir / f"{prefix}_explainability"
                    explainability_paths = (
                        ("pore_delta_pred", pore_pred_delta, pore_delta_limit, "signed"),
                        ("pore_delta_true", pore_true_delta, pore_delta_limit, "signed"),
                        ("pore_delta_abs_error", pore_pred_delta - pore_true_delta, pore_delta_limit, "abs"),
                        ("stress_delta_pred", stress_pred_delta, stress_delta_limit, "signed"),
                        ("stress_delta_true", stress_true_delta, stress_delta_limit, "signed"),
                        ("stress_delta_abs_error", stress_pred_delta - stress_true_delta, stress_delta_limit, "abs"),
                    )
                    for subdir, tensor, limit, mode in explainability_paths:
                        target_dir = explainability_dir / subdir
                        target_dir.mkdir(parents=True, exist_ok=True)
                        target_path = target_dir / f"{subdir}_{stem}.png"
                        if mode == "signed":
                            save_signed_field_map(module, tensor, target_path, limit)
                        else:
                            save_abs_field_map(module, tensor, target_path, limit)

                result = {column: row[column] for column in feature_columns}
                result[label_column] = row[label_column]
                if name_column and name_column in frame.columns:
                    result[name_column] = row[name_column]
                result["prediction"] = pred_values[offset]
                result["ground_truth"] = true_values[offset]
                result["numeric_output_mode"] = numeric_mode
                if numeric_mode == "legacy_fr_index":
                    result["prediction_index_raw"] = float(outputs5[offset].detach().cpu().reshape(-1)[0])
                    result["ground_truth_index_raw"] = float(y5[offset].detach().cpu().reshape(-1)[0])
                prediction_rows.append(result)

                sample_metric_row = {column: row[column] for column in feature_columns}
                sample_metric_row[label_column] = row[label_column]
                if name_column and name_column in frame.columns:
                    sample_metric_row[name_column] = row[name_column]
                sample_metric_row.update(sample_metrics)
                sample_metric_rows.append(sample_metric_row)

                explainability_row = {column: row[column] for column in feature_columns}
                explainability_row[label_column] = row[label_column]
                if name_column and name_column in frame.columns:
                    explainability_row[name_column] = row[name_column]
                explainability_row.update(explainability_stats)
                explainability_rows.append(explainability_row)

            sample_index += batch_size

    all_preds = np.asarray([row["prediction"] for row in prediction_rows], dtype=float)
    all_trues = np.asarray([row["ground_truth"] for row in prediction_rows], dtype=float)
    if len(all_preds) >= 2:
        ss_res = np.sum((all_trues - all_preds) ** 2)
        ss_tot = np.sum((all_trues - all_trues.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
    else:
        r2 = np.nan

    summary_rows = []
    for key, values in metrics.items():
        summary_rows.append(
            {
                "metric": key,
                "mean": float(np.mean(values)) if values else np.nan,
                "std": float(np.std(values)) if values else np.nan,
                "min": float(np.min(values)) if values else np.nan,
                "max": float(np.max(values)) if values else np.nan,
            }
        )
    summary_rows.append({"metric": "task5_r2", "mean": float(r2), "std": np.nan, "min": np.nan, "max": np.nan})

    results_path = output_dir / f"{prefix}_prediction_results.xlsx"
    sample_metrics_path = output_dir / f"{prefix}_sample_metrics_results.xlsx"
    explainability_path = output_dir / f"{prefix}_explainability_results.xlsx"
    metrics_path = output_dir / f"{prefix}_metrics_results_summary.xlsx"
    text_path = output_dir / f"{prefix}Results.txt"
    module.pd.DataFrame(prediction_rows).to_excel(results_path, index=False)
    if save_sample_metrics:
        module.pd.DataFrame(sample_metric_rows).to_excel(sample_metrics_path, index=False)
    if save_explainability:
        module.pd.DataFrame(explainability_rows).to_excel(explainability_path, index=False)
    module.pd.DataFrame(summary_rows).to_excel(metrics_path, index=False)
    with text_path.open("w", encoding="utf-8") as f:
        for row in summary_rows:
            f.write(f"{row['metric']}: {row['mean']:.6f}\n")

    print(f"Saved predictions to {results_path}")
    if save_sample_metrics:
        print(f"Saved sample metrics to {sample_metrics_path}")
    if save_explainability:
        print(f"Saved explainability outputs to {explainability_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved text summary to {text_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Unified slope training and prediction entrypoint.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    parser.add_argument(
        "--action",
        choices=["train", "predict", "train,predict"],
        help="Override config action/actions.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    model_types = config.get("model_types")
    if model_types is None:
        model_types = [resolve_model_type(config)]
    elif isinstance(model_types, str):
        model_types = [model_types]

    feature_columns = config["feature_columns"]
    feature_dim = len(feature_columns)
    set_random_seed(configured_seed(config))

    os.chdir(str(CODE_DIR))
    if args.action:
        actions = args.action.split(",")
    else:
        actions = config.get("actions")
        if actions is None:
            actions = [config.get("action", "predict")]
        if isinstance(actions, str):
            actions = actions.split(",")

    for model_type in model_types:
        if model_type not in MODEL_REGISTRY:
            known = ", ".join(sorted(MODEL_REGISTRY))
            raise ValueError(f"Unknown model_type '{model_type}'. Known values: {known}")
        module = load_model_module(model_type)
        patch_feature_modules(module, feature_dim)
        set_random_seed(configured_seed(config), module)

        for action in actions:
            if action == "train":
                run_train(module, config)
            elif action == "predict":
                run_predict(module, config, model_type)
            else:
                raise ValueError(f"Unknown action '{action}'")


if __name__ == "__main__":
    main()
