
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SOURCE_CODE_DIR = PROJECT_ROOT / "Source Code"

sys.path.insert(0, str(SOURCE_CODE_DIR))
sys.path.insert(1, str(SCRIPT_DIR))

np = None
plt = None
cm = None
colors = None
ticker = None
measure = None
Poly3DCollection = None
torch = None
RegularGridInterpolator = None
gaussian_filter = None
uniform_filter = None
label = None
sobel = None
build_model_input = None
get_input_channel_names = None
get_target_scale = None
LocalizationAwareHybridUNet = None


def _load_project_modules() -> None:
    """Load model dependencies only when figures are actually generated."""
    global np, plt, cm, colors, ticker, measure, Poly3DCollection
    global torch, RegularGridInterpolator, gaussian_filter, uniform_filter, label, sobel
    global build_model_input, get_input_channel_names, get_target_scale, LocalizationAwareHybridUNet
    if np is not None:
        return
    np = importlib.import_module("numpy")
    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    cm = importlib.import_module("matplotlib.cm")
    colors = importlib.import_module("matplotlib.colors")
    ticker = importlib.import_module("matplotlib.ticker")
    measure = importlib.import_module("skimage.measure")
    Poly3DCollection = importlib.import_module("mpl_toolkits.mplot3d.art3d").Poly3DCollection
    torch = importlib.import_module("torch")
    RegularGridInterpolator = importlib.import_module("scipy.interpolate").RegularGridInterpolator
    scipy_ndimage = importlib.import_module("scipy.ndimage")
    gaussian_filter = scipy_ndimage.gaussian_filter
    uniform_filter = scipy_ndimage.uniform_filter
    label = scipy_ndimage.label
    sobel = scipy_ndimage.sobel
    data_generator = importlib.import_module("data_generator")
    build_model_input = data_generator.build_model_input
    get_input_channel_names = data_generator.get_input_channel_names
    get_target_scale = data_generator.get_target_scale
    LocalizationAwareHybridUNet = importlib.import_module("network").LocalizationAwareHybridUNet


CASE_CATEGORY_FOLDERS = {
    "synthetic_prism_model": "synthetic_prism_model",
    "synthetic_two_prisms_model": "synthetic_two_prisms_model",
    "synthetic_irregular_model": "synthetic_irregular_model",
    "vinton_field_data": "vinton_field_data",
}

DATA_CASE_FOLDERS = {
    "synthetic_prism_model": "synthetic_prism_model",
    "synthetic_two_prisms_model": "synthetic_two_prisms_model",
    "synthetic_irregular_model": "synthetic_irregular_model",
}

DATA_FILE_PREFIXES = {
    "synthetic_prism_model": "synthetic_prism_model",
    "synthetic_two_prisms_model": "synthetic_two_prisms_model",
    "synthetic_irregular_model": "synthetic_irregular_model",
}

MAIN_PAGE_EXPORT_NAMES = (
    "01_primary_input",
    "02_aux_input",
    "03_prediction_xy",
    "04_prediction_xz",
    "05_prediction_yz",
    "06_source_projection_xy",
    "07_source_extent_xz",
    "08_source_depth_profile",
)

GUI_BACKEND_INTERMEDIATE_FILENAMES = (
    "real_data_inference_visualization.png",
    "real_data_derivative_isosurfaces.png",
    "real_data_inference_outputs.npz",
    "real_data_derivative_fields.npz",
)

STANDARD_2D_SLICE_FIGSIZE = (6.0, 5.0)
STANDARD_2D_SLICE_DPI = 160
THREE_D_FIGSIZE = (9.2, 6.8)
THREE_D_DPI = 110
THREE_D_VIEW_ELEV = 25.0
THREE_D_VIEW_AZIM = 225.0
THREE_D_GRID_NUM = 15
THREE_D_GRID_COLOR = (0.5, 0.5, 0.5, 0.3)
THREE_D_PANE_FACE_COLOR = (0.95, 0.95, 0.95, 0.3)
THREE_D_PANE_EDGE_COLOR = (0.8, 0.8, 0.8, 0.3)
THREE_D_COLORBAR_LAYOUT = {
    "fraction": 0.024,
    "pad": 0.018,
    "shrink": 0.56,
    "aspect": 36,
    "subplot_right": 0.89,
}
PAPER_CLIP_PERCENTILE = 99.5
PAPER_LINEAR_LOWER_PERCENTILE = 0.5
SYNTHETIC_CASES = {
    "synthetic_prism_model": {
        "label": "Synthetic prism model",
        "fig_input": "Fig04_synthetic_prism_model_and_observed_gravity_gradient.png",
        "fig_prediction": "Fig05_synthetic_prism_model_predicted_3d_gravity_field_and_gradient_magnitude.png",
        "fig_slices": "Fig06_synthetic_prism_model_gravity_field_slices.png",
        "slices": {"z": 1550.0, "y": 1600.0, "x": 1650.0},
    },
    "synthetic_two_prisms_model": {
        "label": "Synthetic two-prism model",
        "fig_input": "Fig07_synthetic_two_prisms_model_and_observed_gravity_gradient.png",
        "fig_prediction": "Fig08_synthetic_two_prisms_model_predicted_3d_gravity_field_and_gradient_magnitude.png",
        "fig_slices": "Fig09_synthetic_two_prisms_model_gravity_field_slices.png",
        "slices": {"z": 650.0, "y": 1550.0, "x": 2150.0},
    },
    "synthetic_irregular_model": {
        "label": "Synthetic irregular model",
        "fig_input": "Fig10_synthetic_irregular_model_and_observed_gravity_gradient.png",
        "fig_prediction": "Fig11_synthetic_irregular_model_predicted_3d_gravity_field_and_gradient_magnitude.png",
        "fig_slices": "Fig12_synthetic_irregular_model_gravity_field_slices.png",
        "slices": {"z": 1250.0, "y": 1600.0, "x": 1450.0},
    },
}

REAL_FIGURES = {
    "fig_input": "Fig13_vinton_field_data_observed_gravity_gradient.png",
    "fig_prediction": "Fig14_vinton_field_data_predicted_3d_gravity_field_and_gradient_magnitude.png",
    "fig_slices": "Fig15_vinton_field_data_gravity_field_slices.png",
    "slices": {"z": 600.0, "y": 1714.0, "x": 2031.0},
}


@dataclass
class VolumeBundle:
    volume: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


@dataclass
class InferenceResult:
    case_name: str
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    surface_channels: dict[str, np.ndarray]
    prediction: np.ndarray
    gradient_magnitude: np.ndarray
    source_response: np.ndarray
    source_response_raw: np.ndarray | None = None
    true_gravity: np.ndarray | None = None
    true_density: np.ndarray | None = None
    normalization_mode: str = ""
    target_scale: float = 1.0


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _case_output_dir(output_root: Path, case_name: str) -> Path:
    """Return the figure subfolder that mirrors the packaged data categories."""
    return output_root / CASE_CATEGORY_FOLDERS.get(case_name, case_name)


def _case_data_dir(data_root: Path, case_name: str) -> Path:
    return data_root / DATA_CASE_FOLDERS.get(case_name, case_name)


def _case_data_prefix(case_name: str) -> str:
    return DATA_FILE_PREFIXES.get(case_name, case_name)


def _real_data_gzz_path(data_root: Path) -> Path:
    return data_root / "vinton_field_data" / "vinton_observed_gzz.txt"


def _expected_reference_export_paths(case_root: Path, synthetic: bool) -> list[Path]:
    """List the GUI-style files expected under one categorized result folder."""
    paths = [
        case_root / "observed_gravity_gradient" / "observed_gravity_gradient_gzz.png",
        case_root / "predicted_gravity_field_slices_2d" / "predicted_gravity_field_slice_xy.png",
        case_root / "predicted_gravity_field_slices_2d" / "predicted_gravity_field_slice_xz.png",
        case_root / "predicted_gravity_field_slices_2d" / "predicted_gravity_field_slice_yz.png",
        case_root / "predicted_3d_gravity_field" / "predicted_3d_gravity_field.png",
        case_root / "predicted_gravity_field_slices_3d" / "predicted_gravity_field_3d_slice_xy.png",
        case_root / "predicted_gravity_field_slices_3d" / "predicted_gravity_field_3d_slice_xz.png",
        case_root / "predicted_gravity_field_slices_3d" / "predicted_gravity_field_3d_slice_yz.png",
        case_root / "gradient_magnitude_slices_2d" / "gradient_magnitude_slice_xy.png",
        case_root / "gradient_magnitude_slices_2d" / "gradient_magnitude_slice_xz.png",
        case_root / "gradient_magnitude_slices_2d" / "gradient_magnitude_slice_yz.png",
        case_root / "gradient_magnitude_3d_field" / "gradient_magnitude_3d_field.png",
        case_root / "gradient_magnitude_slices_3d" / "gradient_magnitude_3d_slice_xy.png",
        case_root / "gradient_magnitude_slices_3d" / "gradient_magnitude_3d_slice_xz.png",
        case_root / "gradient_magnitude_slices_3d" / "gradient_magnitude_3d_slice_yz.png",
    ]
    if synthetic:
        paths.extend(
            [
                case_root / "synthetic_model" / "synthetic_model_3d_isosurface.png",
                case_root
                / "synthetic_model"
                / "synthetic_model_slices"
                / "2d_slices"
                / "synthetic_model_2d_slice_xy.png",
                case_root
                / "synthetic_model"
                / "synthetic_model_slices"
                / "2d_slices"
                / "synthetic_model_2d_slice_xz.png",
                case_root
                / "synthetic_model"
                / "synthetic_model_slices"
                / "2d_slices"
                / "synthetic_model_2d_slice_yz.png",
                case_root
                / "synthetic_model"
                / "synthetic_model_slices"
                / "3d_slices"
                / "synthetic_model_3d_slice_xy.png",
                case_root
                / "synthetic_model"
                / "synthetic_model_slices"
                / "3d_slices"
                / "synthetic_model_3d_slice_xz.png",
                case_root
                / "synthetic_model"
                / "synthetic_model_slices"
                / "3d_slices"
                / "synthetic_model_3d_slice_yz.png",
                case_root / "synthetic_model" / "synthetic_model_slices" / "position_info.json",
            ]
        )
    return paths


def _load_npz_volume(path: Path, key: str) -> VolumeBundle:
    if not path.exists():
        raise FileNotFoundError(f"Missing NPZ file: {path}")
    with np.load(path, allow_pickle=True) as data:
        if key not in data.files:
            raise KeyError(f"File {path} does not contain key '{key}'. Available keys: {data.files}")
        volume = np.asarray(data[key], dtype=np.float32)
        x = np.asarray(data["x_coords"], dtype=np.float64) if "x_coords" in data.files else np.arange(volume.shape[0]) * 50.0
        y = np.asarray(data["y_coords"], dtype=np.float64) if "y_coords" in data.files else np.arange(volume.shape[1]) * 50.0
        z = np.asarray(data["z_coords"], dtype=np.float64) if "z_coords" in data.files else np.arange(volume.shape[2]) * 50.0
    return VolumeBundle(volume=volume, x=x, y=y, z=z)


def _coord_index(coords: np.ndarray, value: float) -> int:
    coords = np.asarray(coords, dtype=np.float64).reshape(-1)
    if coords.size == 0:
        return 0
    return int(np.argmin(np.abs(coords - float(value))))


def _axis_step(coords: np.ndarray, fallback: float = 50.0) -> float:
    coords = np.asarray(coords, dtype=np.float64).reshape(-1)
    if coords.size < 2:
        return fallback
    return float(np.median(np.diff(coords)))


def _display_z_coords(z_coords: np.ndarray) -> np.ndarray:
    z_values = np.asarray(z_coords, dtype=np.float64).reshape(-1)
    if z_values.size == 0:
        return z_values
    finite = z_values[np.isfinite(z_values)]
    if finite.size == 0:
        return z_values
    return z_values - float(np.min(finite))


def _npz_scalar_to_str(value) -> str:
    array = np.asarray(value)
    if array.shape == ():
        return str(array.item())
    if array.size == 1:
        return str(array.reshape(-1)[0])
    return str(array)


def _optional_saved_grid(data, key: str) -> np.ndarray | None:
    if key not in data.files:
        return None
    value = np.asarray(data[key], dtype=np.float32)
    if value.ndim != 2 or value.size == 0:
        return None
    return value


def _safe_torch_load(checkpoint_path: Path, device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)
    except Exception as exc:
        if "Weights only load failed" in str(exc):
            return torch.load(checkpoint_path, map_location=device, weights_only=False)
        raise


def _load_model_from_checkpoint(checkpoint_path: Path, device, model_capacity: str | None, input_mode: str | None):
    checkpoint = _safe_torch_load(checkpoint_path, device)
    checkpoint_capacity = checkpoint.get("model_capacity")
    checkpoint_input_mode = checkpoint.get("input_mode")
    state_dict = checkpoint.get("model_state_dict", {})

    if checkpoint_capacity is not None and model_capacity is not None and model_capacity != checkpoint_capacity:
        warnings.warn(f"Checkpoint capacity is {checkpoint_capacity!r}; ignoring requested {model_capacity!r}.")
    if checkpoint_input_mode is not None and input_mode is not None and input_mode != checkpoint_input_mode:
        warnings.warn(f"Checkpoint input_mode is {checkpoint_input_mode!r}; ignoring requested {input_mode!r}.")
    if checkpoint_input_mode is None and input_mode not in (None, "gz_amp"):
        warnings.warn("Legacy checkpoint lacks input_mode metadata; falling back to gz_amp compatibility.")

    resolved_capacity = checkpoint_capacity or model_capacity or "small"
    resolved_input_mode = checkpoint_input_mode or "gz_amp"
    input_channels = checkpoint.get("input_channels", len(get_input_channel_names(resolved_input_mode)))
    use_multimodal_stem = checkpoint.get(
        "use_multimodal_stem",
        any(key.startswith("input_adapter.") for key in state_dict),
    )
    model = LocalizationAwareHybridUNet(
        capacity=resolved_capacity,
        input_channels=input_channels,
        input_mode=resolved_input_mode,
        use_multimodal_stem=use_multimodal_stem,
    ).to(device)
    try:
        incompatible = model.load_state_dict(state_dict, strict=False)
        allowed_missing_prefixes = (
            "decoder_3d.body_head.",
            "decoder_3d.center_head.",
            "decoder_3d.positive_body_head.",
            "decoder_3d.negative_body_head.",
            "decoder_3d.axis_head.",
        )
        disallowed_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        if disallowed_missing or incompatible.unexpected_keys:
            raise RuntimeError("Checkpoint architecture mismatch.")
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint architecture mismatch. Retrain the checkpoint with the current network before inference."
        ) from exc

    aux_head_prefixes = (
        "decoder_3d.body_head.",
        "decoder_3d.center_head.",
        "decoder_3d.positive_body_head.",
        "decoder_3d.negative_body_head.",
        "decoder_3d.axis_head.",
    )
    has_aux_heads = not incompatible.missing_keys or any(key.startswith(aux_head_prefixes) for key in state_dict)
    return model, checkpoint, resolved_input_mode, has_aux_heads


def _select_aux_channel(input_mode: str) -> str:
    aux_channels = [name for name in get_input_channel_names(input_mode) if name != "gz"]
    return aux_channels[0] if aux_channels else "gz"


def _select_primary_channel(input_mode: str) -> str:
    return get_input_channel_names(input_mode)[0]


def _resolve_input_normalization_mode(checkpoint) -> str:
    if "input_normalization_mode" in checkpoint:
        return str(checkpoint["input_normalization_mode"])
    if "input_mode" in checkpoint:
        return "gz_ref_aux_separate"
    return "legacy_shared"


def _build_inference_input(surface_channels: dict[str, np.ndarray], input_mode: str, checkpoint):
    normalization_mode = _resolve_input_normalization_mode(checkpoint)
    channel_names = get_input_channel_names(input_mode)
    if normalization_mode == "legacy_shared":
        scale = max(
            max(float(np.max(np.abs(surface_channels[channel_name]))), 1e-12)
            for channel_name in channel_names
        )
        input_array = np.stack(
            [(surface_channels[channel_name] / scale).astype(np.float32) for channel_name in channel_names],
            axis=0,
        )
        return input_array, scale, normalization_mode

    target_scale = get_target_scale(surface_channels)
    input_array, _, target_scale = build_model_input(
        surface_channels,
        input_mode=input_mode,
        target_scale=target_scale,
    )
    return input_array, target_scale, normalization_mode


def _load_real_surface_grid(txt_path: Path):
    arr = np.loadtxt(txt_path, delimiter=",")
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected 3 columns (x, y, value) in {txt_path}, got shape {arr.shape}")
    xs = np.unique(arr[:, 0])
    ys = np.unique(arr[:, 1])
    grid = np.full((xs.size, ys.size), np.nan, dtype=np.float64)
    x_to_idx = {value: idx for idx, value in enumerate(xs)}
    y_to_idx = {value: idx for idx, value in enumerate(ys)}
    for x_value, y_value, field_value in arr:
        grid[x_to_idx[x_value], y_to_idx[y_value]] = field_value
    if np.isnan(grid).any():
        raise ValueError(f"Incomplete grid detected in {txt_path}")
    return xs, ys, grid


def _resample_grid_to_model_size(xs, ys, grid, target_size: int = 64):
    interpolator = RegularGridInterpolator((xs, ys), grid, method="linear", bounds_error=False, fill_value=None)
    x_new = np.linspace(xs.min(), xs.max(), target_size)
    y_new = np.linspace(ys.min(), ys.max(), target_size)
    x_mesh, y_mesh = np.meshgrid(x_new, y_new, indexing="ij")
    samples = np.stack([x_mesh.ravel(), y_mesh.ravel()], axis=1)
    return x_new, y_new, interpolator(samples).reshape(target_size, target_size)


def _compute_surface_analytic_amplitude(field_2d: np.ndarray, dx: float, dy: float) -> np.ndarray:
    nx, ny = field_2d.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    radial_k = np.sqrt(kx_grid**2 + ky_grid**2)
    field_fft = np.fft.fft2(field_2d)
    dfdx = np.fft.ifft2(1j * kx_grid * field_fft).real
    dfdy = np.fft.ifft2(1j * ky_grid * field_fft).real
    dfdz = np.fft.ifft2(radial_k * field_fft).real
    return np.sqrt(dfdx**2 + dfdy**2 + dfdz**2).astype(np.float32)


def _convert_real_channel_to_training_units(channel_name: str, grid: np.ndarray):
    grid = np.asarray(grid, dtype=np.float32)
    max_abs = float(np.max(np.abs(grid))) if grid.size else 0.0
    scale = 1.0
    assumed_unit = "training_native"
    if channel_name == "gzz" and 0.0 < max_abs < 1e-5:
        scale = 1e5
        assumed_unit = "SI_to_mGal_per_m"
    elif channel_name == "gz" and 0.0 < max_abs < 1e-2:
        scale = 1e5
        assumed_unit = "SI_to_mGal"
    return (grid * scale).astype(np.float32), {
        "channel": channel_name,
        "input_abs_max": max_abs,
        "applied_scale": scale,
        "assumed_unit": assumed_unit,
    }


def _surface_fit_metrics(pred_surface: np.ndarray, obs_surface: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred_surface, dtype=np.float64).reshape(-1)
    obs = np.asarray(obs_surface, dtype=np.float64).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(obs)
    if not np.any(mask):
        return {"corr": np.nan, "rmse": np.nan, "mae": np.nan}
    pred = pred[mask]
    obs = obs[mask]
    return {
        "corr": float(np.corrcoef(pred, obs)[0, 1]) if pred.size > 1 else np.nan,
        "rmse": float(np.sqrt(np.mean((pred - obs) ** 2))),
        "mae": float(np.mean(np.abs(pred - obs))),
    }


def _compute_surface_vertical_gradient_from_volume(field_3d: np.ndarray, z_coords: np.ndarray) -> np.ndarray:
    volume = np.asarray(field_3d, dtype=np.float32)
    if volume.ndim != 3 or volume.shape[2] < 2:
        return np.zeros(volume.shape[:2], dtype=np.float32)
    z_coords = np.asarray(z_coords, dtype=np.float64).reshape(-1)
    dz = max(abs(float(z_coords[1] - z_coords[0])) if z_coords.size > 1 else 1.0, 1e-12)
    if volume.shape[2] >= 3:
        gradient = (-3.0 * volume[:, :, 0] + 4.0 * volume[:, :, 1] - volume[:, :, 2]) / (2.0 * dz)
    else:
        gradient = (volume[:, :, 1] - volume[:, :, 0]) / dz
    return np.asarray(gradient, dtype=np.float32)


def _apply_post_inference_surface_calibration(
    field_3d: np.ndarray,
    z_coords: np.ndarray,
    observed_gz: np.ndarray | None = None,
    observed_gzz: np.ndarray | None = None,
    gz_weight: float = 5.0,
    gzz_weight: float = 5.0,
    prior_weights=(1.0, 1.0, 1.0),
):
    volume = np.asarray(field_3d, dtype=np.float32)
    corrected = volume.copy()
    if volume.ndim != 3 or volume.shape[2] < 2:
        return corrected, {"applied": False, "reason": "insufficient_depth_layers"}

    z_coords = np.asarray(z_coords, dtype=np.float64).reshape(-1)
    dz = max(abs(float(z_coords[1] - z_coords[0])) if z_coords.size > 1 else 1.0, 1e-12)
    n_layers = 3 if volume.shape[2] >= 3 else 2
    prior_weights = np.asarray(prior_weights, dtype=np.float64).reshape(-1)
    if prior_weights.size < n_layers:
        prior_weights = np.pad(prior_weights, (0, n_layers - prior_weights.size), mode="edge")
    prior_weights = np.maximum(prior_weights[:n_layers], 1e-8)

    deriv = np.array([-3.0, 4.0, -1.0], dtype=np.float64) / (2.0 * dz) if n_layers == 3 else np.array([-1.0, 1.0], dtype=np.float64) / dz
    observed_gz_arr = None if observed_gz is None else np.asarray(observed_gz, dtype=np.float64)
    observed_gzz_arr = None if observed_gzz is None else np.asarray(observed_gzz, dtype=np.float64)
    effective_gz_weight = float(gz_weight) if observed_gz_arr is not None else 0.0
    effective_gzz_weight = float(gzz_weight) if observed_gzz_arr is not None else 0.0
    if effective_gz_weight <= 0.0 and effective_gzz_weight <= 0.0:
        return corrected, {"applied": False, "reason": "no_surface_targets"}

    surface_selector = np.zeros((n_layers,), dtype=np.float64)
    surface_selector[0] = 1.0
    system = np.diag(prior_weights)
    if effective_gz_weight > 0.0:
        system += effective_gz_weight * np.outer(surface_selector, surface_selector)
    if effective_gzz_weight > 0.0:
        system += effective_gzz_weight * np.outer(deriv, deriv)

    rhs = volume[:, :, :n_layers].astype(np.float64) * prior_weights.reshape(1, 1, -1)
    if effective_gz_weight > 0.0:
        rhs += effective_gz_weight * observed_gz_arr[..., None] * surface_selector.reshape(1, 1, -1)
    if effective_gzz_weight > 0.0:
        rhs += effective_gzz_weight * observed_gzz_arr[..., None] * deriv.reshape(1, 1, -1)
    corrected[:, :, :n_layers] = np.matmul(rhs, np.linalg.inv(system).T).astype(np.float32)

    diagnostics = {"applied": True}
    if observed_gz_arr is not None:
        diagnostics["surface_gz_before"] = _surface_fit_metrics(volume[:, :, 0], observed_gz_arr)
        diagnostics["surface_gz_after"] = _surface_fit_metrics(corrected[:, :, 0], observed_gz_arr)
    if observed_gzz_arr is not None:
        diagnostics["surface_gzz_before"] = _surface_fit_metrics(
            _compute_surface_vertical_gradient_from_volume(volume, z_coords),
            observed_gzz_arr,
        )
        diagnostics["surface_gzz_after"] = _surface_fit_metrics(
            _compute_surface_vertical_gradient_from_volume(corrected, z_coords),
            observed_gzz_arr,
        )
    return corrected, diagnostics


def _build_boundary_taper(shape: tuple[int, int, int], border: int = 4) -> np.ndarray:
    border = max(0, int(border))
    if border == 0:
        return np.ones(shape, dtype=np.float32)

    def axis_window(length: int) -> np.ndarray:
        if length <= 2:
            return np.ones(length, dtype=np.float32)
        current_border = min(border, max(length // 2 - 1, 1))
        window = np.ones(length, dtype=np.float32)
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, current_border, dtype=np.float32))
        window[:current_border] = ramp
        window[-current_border:] = ramp[::-1]
        return window

    wx = axis_window(shape[0])
    wy = axis_window(shape[1])
    wz = axis_window(shape[2])
    return wx[:, None, None] * wy[None, :, None] * wz[None, None, :]


def _zero_boundary_shell(volume: np.ndarray, shell_width: int = 4, fill_value: float = 0.0) -> np.ndarray:
    shell_width = max(0, int(shell_width))
    masked = np.array(volume, copy=True)
    if shell_width == 0 or min(masked.shape) <= 2 * shell_width:
        return masked
    masked[:shell_width, :, :] = fill_value
    masked[-shell_width:, :, :] = fill_value
    masked[:, :shell_width, :] = fill_value
    masked[:, -shell_width:, :] = fill_value
    masked[:, :, :shell_width] = fill_value
    masked[:, :, -shell_width:] = fill_value
    return masked


def _interior_volume_view(volume: np.ndarray, edge_ignore: int = 4) -> np.ndarray:
    edge_ignore = max(0, int(edge_ignore))
    if edge_ignore == 0 or min(volume.shape) <= 2 * edge_ignore:
        return volume
    return volume[edge_ignore:-edge_ignore, edge_ignore:-edge_ignore, edge_ignore:-edge_ignore]


def _build_interior_weight_2d(shape: tuple[int, int], edge_ignore: int = 4) -> np.ndarray:
    edge_ignore = max(0, int(edge_ignore))
    weight = np.ones(shape, dtype=np.float32)
    if edge_ignore == 0 or min(shape) <= 2 * edge_ignore:
        return weight
    weight[:edge_ignore, :] = 0.0
    weight[-edge_ignore:, :] = 0.0
    weight[:, :edge_ignore] = 0.0
    weight[:, -edge_ignore:] = 0.0
    return weight


def _build_interior_weight(shape: tuple[int, int, int], edge_ignore: int = 4) -> np.ndarray:
    edge_ignore = max(0, int(edge_ignore))
    weight = np.ones(shape, dtype=np.float32)
    if edge_ignore == 0 or min(shape) <= 2 * edge_ignore:
        return weight
    weight[:edge_ignore, :, :] = 0.0
    weight[-edge_ignore:, :, :] = 0.0
    weight[:, :edge_ignore, :] = 0.0
    weight[:, -edge_ignore:, :] = 0.0
    weight[:, :, :edge_ignore] = 0.0
    weight[:, :, -edge_ignore:] = 0.0
    return weight


def _normalize_source_response(response: np.ndarray, edge_ignore: int = 4) -> np.ndarray:
    response = np.abs(np.asarray(response, dtype=np.float32))
    interior_weight = _build_interior_weight(response.shape, edge_ignore=edge_ignore)
    interior_values = response[interior_weight > 0.5]
    if interior_values.size == 0:
        interior_values = response.reshape(-1)
    low_q = float(np.quantile(interior_values, 0.50))
    high_q = float(np.quantile(interior_values, 0.995))
    normalized = np.clip((response - low_q) / max(high_q - low_q, 1e-6), 0.0, 1.0)
    return (normalized * (0.25 + 0.75 * interior_weight)).astype(np.float32)


def _normalize_surface_response(response: np.ndarray, edge_ignore: int = 4) -> np.ndarray:
    response = np.abs(np.asarray(response, dtype=np.float32))
    interior_weight = _build_interior_weight_2d(response.shape, edge_ignore=edge_ignore)
    interior_values = response[interior_weight > 0.5]
    if interior_values.size == 0:
        interior_values = response.reshape(-1)
    low_q = float(np.quantile(interior_values, 0.50))
    high_q = float(np.quantile(interior_values, 0.995))
    normalized = np.clip((response - low_q) / max(high_q - low_q, 1e-6), 0.0, 1.0)
    return (normalized * (0.25 + 0.75 * interior_weight)).astype(np.float32)


def _compute_volume_derivative_fields(field_3d: np.ndarray, dx: float, dy: float, dz: float, edge_taper_width: int = 4, smooth_sigma: float = 0.8):
    centered_field = np.asarray(field_3d, dtype=np.float32) - np.float32(np.median(field_3d))
    tapered_field = centered_field * _build_boundary_taper(centered_field.shape, border=edge_taper_width)
    dfdx, dfdy, dfdz = np.gradient(tapered_field, dx, dy, dz, edge_order=1)
    thdr = np.sqrt(dfdx**2 + dfdy**2)
    asa3d = np.sqrt(dfdx**2 + dfdy**2 + dfdz**2)
    if smooth_sigma and smooth_sigma > 0:
        thdr = gaussian_filter(thdr, sigma=smooth_sigma)
        asa3d = gaussian_filter(asa3d, sigma=smooth_sigma)
    return {
        "dfdx": _zero_boundary_shell(dfdx, shell_width=edge_taper_width),
        "dfdy": _zero_boundary_shell(dfdy, shell_width=edge_taper_width),
        "dfdz": _zero_boundary_shell(dfdz, shell_width=edge_taper_width),
        "thdr": _zero_boundary_shell(thdr, shell_width=edge_taper_width),
        "asa3d": _zero_boundary_shell(asa3d, shell_width=edge_taper_width),
        "tapered_prediction": tapered_field,
        "edge_taper_width": edge_taper_width,
    }


def _compute_source_localization_response(field_3d: np.ndarray, dx: float, dy: float, dz: float, edge_ignore: int = 4, smooth_sigma: float = 0.8):
    centered_field = np.asarray(field_3d, dtype=np.float32) - np.float32(np.median(field_3d))
    tapered_field = centered_field * _build_boundary_taper(centered_field.shape, border=edge_ignore)
    grad_x = sobel(tapered_field, axis=0, mode="nearest") / max(float(dx), 1e-6)
    grad_y = sobel(tapered_field, axis=1, mode="nearest") / max(float(dy), 1e-6)
    grad_z = sobel(tapered_field, axis=2, mode="nearest") / max(float(dz), 1e-6)
    gradient_response = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    envelope_response = uniform_filter(np.abs(tapered_field), size=5, mode="nearest")
    if smooth_sigma and smooth_sigma > 0:
        gradient_response = gaussian_filter(gradient_response, sigma=smooth_sigma)
        envelope_response = gaussian_filter(envelope_response, sigma=smooth_sigma)
    gradient_response = _normalize_source_response(gradient_response, edge_ignore=edge_ignore)
    envelope_response = _normalize_source_response(envelope_response, edge_ignore=edge_ignore)
    source_response = _normalize_source_response(0.78 * gradient_response + 0.22 * envelope_response, edge_ignore=edge_ignore)
    return {
        "source_response": source_response,
        "gradient_response": gradient_response,
        "envelope_response": envelope_response,
        "tapered_prediction": tapered_field,
        "edge_ignore": max(0, int(edge_ignore)),
    }


def _compute_surface_focus_prior(surface_map: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, edge_ignore: int = 4):
    surface_map = np.asarray(surface_map, dtype=np.float32)
    detrended = surface_map - gaussian_filter(surface_map, sigma=4.2)
    anomaly_response = _normalize_surface_response(gaussian_filter(detrended, sigma=1.1), edge_ignore=edge_ignore)
    peak_index = np.unravel_index(np.argmax(anomaly_response), anomaly_response.shape)
    peak_x = float(x_coords[peak_index[0]])
    peak_y = float(y_coords[peak_index[1]])
    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords, indexing="ij")
    sigma_x = max(650.0, float(abs(x_coords[1] - x_coords[0]) * 6)) if len(x_coords) > 1 else 650.0
    sigma_y = max(650.0, float(abs(y_coords[1] - y_coords[0]) * 6)) if len(y_coords) > 1 else 650.0
    gaussian_prior = np.exp(-0.5 * (((x_mesh - peak_x) / sigma_x) ** 2 + ((y_mesh - peak_y) / sigma_y) ** 2)).astype(np.float32)
    return {
        "response": anomaly_response,
        "center": (peak_x, peak_y),
        "bbox": None,
        "centers": np.asarray([(peak_x, peak_y)], dtype=np.float64),
        "bboxes": [],
        "weight": (0.22 + 0.78 * gaussian_prior).astype(np.float32),
        "threshold": float(np.max(anomaly_response)) if anomaly_response.size else 0.0,
    }


def _assess_aux_response_confidence(volume: np.ndarray, edge_ignore: int = 4, min_peak: float = 0.08, min_dynamic_range: float = 0.03):
    volume = np.asarray(volume, dtype=np.float32)
    interior_values = _interior_volume_view(volume, edge_ignore=edge_ignore).reshape(-1)
    interior_values = interior_values[np.isfinite(interior_values)]
    if interior_values.size == 0:
        return False, {"peak": 0.0, "median": 0.0, "dynamic_range": 0.0}
    peak = float(np.max(interior_values))
    median = float(np.median(interior_values))
    dynamic_range = float(np.quantile(interior_values, 0.995)) - median
    return peak >= min_peak and dynamic_range >= min_dynamic_range, {
        "peak": peak,
        "median": median,
        "dynamic_range": dynamic_range,
    }


def _combine_body_support_response(source_response: np.ndarray, pred_body_mask=None, pred_center_heatmap=None, edge_ignore: int = 4):
    combined = _normalize_source_response(source_response, edge_ignore=edge_ignore)
    diagnostics = {
        "body_mask_used": False,
        "center_heatmap_used": False,
        "body_mask_stats": None,
        "center_heatmap_stats": None,
    }
    if pred_body_mask is not None:
        body_ok, body_stats = _assess_aux_response_confidence(pred_body_mask, edge_ignore=edge_ignore, min_peak=0.08, min_dynamic_range=0.025)
        diagnostics["body_mask_stats"] = body_stats
        if body_ok:
            combined = 0.58 * _normalize_source_response(pred_body_mask, edge_ignore=edge_ignore) + 0.42 * combined
            diagnostics["body_mask_used"] = True
    if pred_center_heatmap is not None:
        center_ok, center_stats = _assess_aux_response_confidence(pred_center_heatmap, edge_ignore=edge_ignore, min_peak=0.12, min_dynamic_range=0.04)
        diagnostics["center_heatmap_stats"] = center_stats
        if center_ok:
            combined = 0.90 * combined + 0.10 * _normalize_source_response(pred_center_heatmap, edge_ignore=edge_ignore)
            diagnostics["center_heatmap_used"] = True
    return _normalize_source_response(combined, edge_ignore=edge_ignore), diagnostics


def _build_trusted_body_response(source_response: np.ndarray, surface_prior_weight: np.ndarray, edge_ignore: int = 4, min_depth_idx: int = 5) -> np.ndarray:
    trusted = np.asarray(source_response, dtype=np.float32) * np.asarray(surface_prior_weight, dtype=np.float32)[..., None]
    trusted = _zero_boundary_shell(trusted, shell_width=edge_ignore, fill_value=0.0)
    if 0 < min_depth_idx < trusted.shape[2]:
        trusted[:, :, :min_depth_idx] = 0.0
    return _normalize_source_response(trusted, edge_ignore=edge_ignore)


def _run_gui_real_data_inference(
    output_dir: Path,
    *,
    checkpoint_path: Path,
    model_capacity: str | None,
    input_mode: str | None,
    real_gz_path: Path | None,
    real_gzz_path: Path | None,
    surface_calibration: bool = False,
) -> None:
    """Run the real-data inference flow needed by the manuscript exporter."""
    _load_project_modules()
    _ensure_dir(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, resolved_input_mode, has_aux_heads = _load_model_from_checkpoint(
        checkpoint_path,
        device,
        model_capacity=model_capacity,
        input_mode=input_mode,
    )
    required_channels = get_input_channel_names(resolved_input_mode)

    primary_grid_path = real_gz_path if "gz" in required_channels else real_gzz_path
    if primary_grid_path is None:
        raise ValueError(f"Real-data inference for input_mode={resolved_input_mode!r} requires gz or gzz input.")

    primary_xs, primary_ys, primary_grid_raw = _load_real_surface_grid(primary_grid_path)
    x_model, y_model, primary_grid_model = _resample_grid_to_model_size(primary_xs, primary_ys, primary_grid_raw)
    primary_grid_model = primary_grid_model - np.mean(primary_grid_model)
    dx = float(x_model[1] - x_model[0]) if x_model.size > 1 else 1.0
    dy = float(y_model[1] - y_model[0]) if y_model.size > 1 else 1.0

    surface_channels: dict[str, np.ndarray] = {}
    unit_conversions = []
    gz_model = None
    gz_model_centered = None
    if real_gz_path is not None:
        gz_xs, gz_ys, gz_raw = _load_real_surface_grid(real_gz_path)
        _, _, gz_model = _resample_grid_to_model_size(gz_xs, gz_ys, gz_raw)
        gz_model, gz_unit_info = _convert_real_channel_to_training_units("gz", gz_model)
        unit_conversions.append(gz_unit_info)
        gz_model_centered = gz_model - np.mean(gz_model)
    if "gz" in required_channels:
        if gz_model_centered is None:
            raise ValueError("This checkpoint expects gz input. Provide a gz grid.")
        surface_channels["gz"] = gz_model_centered

    gzz_model = None
    if "gzz" in required_channels:
        if real_gzz_path is None:
            raise ValueError("This checkpoint expects gzz input. Provide a gzz grid.")
        gzz_xs, gzz_ys, gzz_raw = _load_real_surface_grid(real_gzz_path)
        _, _, gzz_model = _resample_grid_to_model_size(gzz_xs, gzz_ys, gzz_raw)
        gzz_model, gzz_unit_info = _convert_real_channel_to_training_units("gzz", gzz_model)
        unit_conversions.append(gzz_unit_info)
        surface_channels["gzz"] = gzz_model - np.mean(gzz_model)

    if "amp" in required_channels:
        if "gz" not in surface_channels:
            raise ValueError("Analytic-signal input mode requires gz input.")
        surface_channels["amp"] = _compute_surface_analytic_amplitude(surface_channels["gz"], dx=dx, dy=dy)

    primary_channel_name = _select_primary_channel(resolved_input_mode)
    aux_channel_name = _select_aux_channel(resolved_input_mode)
    aux_surface = surface_channels[aux_channel_name]
    input_array, target_scale, normalization_mode = _build_inference_input(
        surface_channels,
        input_mode=resolved_input_mode,
        checkpoint=checkpoint,
    )

    model.eval()
    with torch.no_grad():
        input_tensor = torch.from_numpy(input_array[np.newaxis, :, :, :]).float().to(device)
        output_dict = model(input_tensor)
        y_pred_norm = output_dict["output"].squeeze().cpu().numpy()
        pred_body_mask = output_dict.get("body_mask")
        pred_center_heatmap = output_dict.get("center_heatmap")

    pred_body_mask_np = pred_body_mask.squeeze().cpu().numpy().astype(np.float32) if has_aux_heads and pred_body_mask is not None else None
    pred_center_heatmap_np = pred_center_heatmap.squeeze().cpu().numpy().astype(np.float32) if has_aux_heads and pred_center_heatmap is not None else None
    y_pred = y_pred_norm * target_scale
    raw_y_pred = y_pred.copy()
    z_coords = np.arange(y_pred.shape[2], dtype=np.float64) * 50.0
    surface_calibration_diagnostics = {"applied": False, "reason": "disabled"}
    if surface_calibration:
        y_pred, surface_calibration_diagnostics = _apply_post_inference_surface_calibration(
            y_pred,
            z_coords=z_coords,
            observed_gz=gz_model_centered,
            observed_gzz=surface_channels.get("gzz"),
        )

    edge_ignore = 4
    min_depth_idx = int(np.searchsorted(z_coords, 250.0, side="left"))
    focus_surface_name = "gzz" if "gzz" in surface_channels else primary_channel_name
    surface_focus = _compute_surface_focus_prior(surface_channels[focus_surface_name], x_model, y_model, edge_ignore=edge_ignore)
    source_fields = _compute_source_localization_response(y_pred, dx=dx, dy=dy, dz=50.0, edge_ignore=edge_ignore)
    source_response_raw, aux_diagnostics = _combine_body_support_response(
        source_fields["source_response"],
        pred_body_mask=pred_body_mask_np,
        pred_center_heatmap=pred_center_heatmap_np,
        edge_ignore=edge_ignore,
    )
    source_response = _build_trusted_body_response(
        source_response_raw,
        surface_focus["weight"],
        edge_ignore=edge_ignore,
        min_depth_idx=min_depth_idx,
    )
    derivative_fields = _compute_volume_derivative_fields(y_pred, dx=dx, dy=dy, dz=50.0)

    np.savez(
        output_dir / "real_data_inference_outputs.npz",
        primary_input=surface_channels[primary_channel_name],
        primary_input_name=np.array(primary_channel_name),
        gz_input_raw=gz_model.astype(np.float32) if gz_model is not None else np.array([], dtype=np.float32),
        gz_input=surface_channels.get("gz", gz_model_centered if gz_model_centered is not None else np.array([], dtype=np.float32)),
        aux_input=aux_surface,
        input_channel_names=np.array(required_channels, dtype=object),
        aux_input_name=np.array(aux_channel_name),
        prediction=y_pred,
        prediction_raw=raw_y_pred,
        prediction_asa3d=derivative_fields["asa3d"],
        prediction_thdr=derivative_fields["thdr"],
        prediction_source_response=source_response,
        prediction_source_response_raw=source_response_raw,
        prediction_source_gradient=source_fields["gradient_response"],
        prediction_source_envelope=source_fields["envelope_response"],
        prediction_body_mask=pred_body_mask_np if pred_body_mask_np is not None else np.array([], dtype=np.float32),
        prediction_center_heatmap=pred_center_heatmap_np if pred_center_heatmap_np is not None else np.array([], dtype=np.float32),
        real_input_unit_scales=np.array([info["applied_scale"] for info in unit_conversions], dtype=np.float64),
        real_input_unit_channels=np.array([info["channel"] for info in unit_conversions], dtype=object),
        surface_calibration_applied=np.array(int(surface_calibration_diagnostics.get("applied", False)), dtype=np.int8),
        prediction_surface_prior_response=surface_focus["response"],
        prediction_surface_prior_weight=surface_focus["weight"],
        surface_prior_center=np.array(surface_focus["center"], dtype=np.float64),
        surface_prior_centers=np.asarray(surface_focus["centers"], dtype=np.float64),
        surface_prior_bboxes=np.array([], dtype=np.float64),
        body_mask_used=np.array(int(aux_diagnostics["body_mask_used"]), dtype=np.int8),
        center_heatmap_used=np.array(int(aux_diagnostics["center_heatmap_used"]), dtype=np.int8),
        normalization_mode=np.array(normalization_mode),
        target_scale=np.array(target_scale, dtype=np.float64),
        x_coords=x_model,
        y_coords=y_model,
        z_coords=z_coords,
    )


def _load_gui_real_data_result(
    case_name: str,
    output_dir: Path,
    *,
    true_gravity: np.ndarray | None = None,
    true_density: np.ndarray | None = None,
) -> tuple[InferenceResult, np.ndarray]:
    npz_path = output_dir / "real_data_inference_outputs.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"GUI real-data inference did not create {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        primary_input = np.asarray(data["primary_input"], dtype=np.float32)
        primary_name = _npz_scalar_to_str(data["primary_input_name"]) if "primary_input_name" in data.files else "gzz"
        aux_input = _optional_saved_grid(data, "aux_input")
        aux_name = _npz_scalar_to_str(data["aux_input_name"]) if "aux_input_name" in data.files else primary_name

        surface_channels: dict[str, np.ndarray] = {primary_name: primary_input}
        if aux_input is not None:
            surface_channels.setdefault(aux_name, aux_input)

        gz_input = _optional_saved_grid(data, "gz_input")
        if gz_input is not None:
            surface_channels.setdefault("gz", gz_input)

        display_input = surface_channels.get("gzz", primary_input)
        normalization_mode = ""
        if "normalization_mode" in data.files:
            normalization_mode = _npz_scalar_to_str(data["normalization_mode"])

        target_scale = 1.0
        if "target_scale" in data.files:
            target_scale = float(np.asarray(data["target_scale"], dtype=np.float64))

        if "prediction_source_response_raw" in data.files:
            source_response_raw = np.asarray(data["prediction_source_response_raw"], dtype=np.float32)
        elif "source_response_raw" in data.files:
            source_response_raw = np.asarray(data["source_response_raw"], dtype=np.float32)
        elif "source_response" in data.files:
            source_response_raw = np.asarray(data["source_response"], dtype=np.float32)
        else:
            source_response_raw = np.asarray(data["prediction_source_response"], dtype=np.float32)

        if "prediction_source_response" in data.files:
            source_response = np.asarray(data["prediction_source_response"], dtype=np.float32)
        elif "source_response" in data.files:
            source_response = np.asarray(data["source_response"], dtype=np.float32)
        else:
            source_response = source_response_raw

        result = InferenceResult(
            case_name=case_name,
            x=np.asarray(data["x_coords"], dtype=np.float64).reshape(-1),
            y=np.asarray(data["y_coords"], dtype=np.float64).reshape(-1),
            z=np.asarray(data["z_coords"], dtype=np.float64).reshape(-1),
            surface_channels=surface_channels,
            prediction=np.asarray(data["prediction"], dtype=np.float32),
            gradient_magnitude=np.asarray(data["prediction_asa3d"], dtype=np.float32),
            source_response=source_response,
            source_response_raw=source_response_raw,
            true_gravity=true_gravity,
            true_density=true_density,
            normalization_mode=normalization_mode,
            target_scale=target_scale,
        )
    return result, display_input


def _cleanup_gui_backend_intermediates(output_dir: Path) -> None:
    for filename in GUI_BACKEND_INTERMEDIATE_FILENAMES:
        path = output_dir / filename
        if path.exists():
            path.unlink()

    legacy_profile_dir = "\u5730\u8868\u62df\u5408"
    for profile_dir in (output_dir / "surface_fit", output_dir / legacy_profile_dir):
        if profile_dir.exists():
            shutil.rmtree(profile_dir)


def _robust_norm(values: np.ndarray, symmetric: bool = True):
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return colors.Normalize(vmin=0.0, vmax=1.0)
    if symmetric:
        limit = float(np.percentile(np.abs(finite), 99.5))
        limit = max(limit, 1e-8)
        return colors.Normalize(vmin=-limit, vmax=limit)
    vmin = float(np.percentile(finite, 0.5))
    vmax = float(np.percentile(finite, 99.5))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return colors.Normalize(vmin=vmin, vmax=vmax)


def _plot_surface(ax, x: np.ndarray, y: np.ndarray, grid: np.ndarray, title: str, cmap_name: str = "bwr", symmetric: bool = True):
    norm = _robust_norm(grid, symmetric=symmetric)
    mesh = ax.pcolormesh(x, y, grid.T, shading="auto", cmap=cmap_name, norm=norm)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    return mesh


def _positive_isosurface_level(volume: np.ndarray, ratio: float = 0.30) -> float | None:
    values = np.asarray(volume, dtype=np.float32)
    abs_values = np.abs(values[np.isfinite(values)])
    if abs_values.size == 0:
        return None
    vmax = float(abs_values.max())
    if vmax <= 0.0:
        return None
    return ratio * vmax


def _plot_abs_isosurface(ax, volume: np.ndarray, coords: tuple[np.ndarray, np.ndarray, np.ndarray], title: str, cmap_name: str = "viridis"):
    x, y, z = coords
    dx = _axis_step(x)
    dy = _axis_step(y)
    dz = _axis_step(z)
    abs_volume = np.abs(np.asarray(volume, dtype=np.float32))
    level = _positive_isosurface_level(abs_volume, ratio=0.30)
    if level is None:
        ax.text2D(0.5, 0.5, "No isosurface", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title, fontsize=11, fontweight="bold")
        return
    if not (float(np.min(abs_volume)) < level < float(np.max(abs_volume))):
        ax.text2D(0.5, 0.5, "Invalid isosurface level", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title, fontsize=11, fontweight="bold")
        return

    verts, faces, _, _ = measure.marching_cubes(abs_volume, level=level, spacing=(dx, dy, dz))
    origin = np.array([float(x[0]), float(y[0]), float(z[0])], dtype=np.float64)
    verts = verts + origin
    face_vertices = verts[faces]
    face_depth = face_vertices[:, :, 2].mean(axis=1)
    norm = colors.Normalize(vmin=float(face_depth.min()), vmax=float(face_depth.max()) + 1e-12)
    face_colors = cm.get_cmap(cmap_name)(norm(face_depth))
    mesh = Poly3DCollection(face_vertices, facecolors=face_colors, edgecolor="none", alpha=0.68)
    ax.add_collection3d(mesh)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    ax.set_zlim(float(z.max()), float(z.min()))
    ax.set_box_aspect((float(np.ptp(x)), float(np.ptp(y)), float(np.ptp(z))))
    ax.view_init(elev=24, azim=-56)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"{title}\nIso={level:.3e}", fontsize=11, fontweight="bold")


def _plot_3d_slice(
    ax,
    volume: np.ndarray,
    coords: tuple[np.ndarray, np.ndarray, np.ndarray],
    plane: str,
    value: float,
    title: str,
    cmap_name: str = "bwr",
    symmetric: bool = True,
):
    x, y, z = coords
    norm = _robust_norm(volume, symmetric=symmetric)
    cmap = cm.get_cmap(cmap_name)
    if plane == "z":
        idx = _coord_index(z, value)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        zz = np.full_like(xx, z[idx], dtype=np.float64)
        facecolors = cmap(norm(volume[:, :, idx]))
        ax.plot_surface(xx, yy, zz, facecolors=facecolors, rstride=1, cstride=1, shade=False)
    elif plane == "y":
        idx = _coord_index(y, value)
        xx, zz = np.meshgrid(x, z, indexing="ij")
        yy = np.full_like(xx, y[idx], dtype=np.float64)
        facecolors = cmap(norm(volume[:, idx, :]))
        ax.plot_surface(xx, yy, zz, facecolors=facecolors, rstride=1, cstride=1, shade=False)
    elif plane == "x":
        idx = _coord_index(x, value)
        yy, zz = np.meshgrid(y, z, indexing="ij")
        xx = np.full_like(yy, x[idx], dtype=np.float64)
        facecolors = cmap(norm(volume[idx, :, :]))
        ax.plot_surface(xx, yy, zz, facecolors=facecolors, rstride=1, cstride=1, shade=False)
    else:
        raise ValueError(f"Unsupported plane: {plane}")
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(y.min()), float(y.max()))
    ax.set_zlim(float(z.max()), float(z.min()))
    ax.set_box_aspect((float(np.ptp(x)), float(np.ptp(y)), float(np.ptp(z))))
    ax.view_init(elev=25, azim=225)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def _extract_slice(volume: np.ndarray, coords: tuple[np.ndarray, np.ndarray, np.ndarray], plane: str, value: float):
    x, y, z = coords
    if plane == "z":
        idx = _coord_index(z, value)
        return x, y, volume[:, :, idx].T, "X (m)", "Y (m)"
    if plane == "y":
        idx = _coord_index(y, value)
        return x, z, volume[:, idx, :].T, "X (m)", "Z (m)"
    if plane == "x":
        idx = _coord_index(x, value)
        return y, z, volume[idx, :, :].T, "Y (m)", "Z (m)"
    raise ValueError(f"Unsupported plane: {plane}")


def _plot_2d_slice(
    ax,
    volume: np.ndarray,
    coords: tuple[np.ndarray, np.ndarray, np.ndarray],
    plane: str,
    value: float,
    title: str,
    cmap_name: str = "bwr",
    symmetric: bool = True,
):
    axis_a, axis_b, slice_data, xlabel, ylabel = _extract_slice(volume, coords, plane, value)
    mesh = ax.pcolormesh(axis_a, axis_b, slice_data, shading="auto", cmap=cmap_name, norm=_robust_norm(volume, symmetric=symmetric))
    if "Z" in ylabel:
        ax.invert_yaxis()
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    return mesh


def _field_unit_text(field_name: str | None) -> str:
    """Return the short unit label used above GUI colorbars."""
    unit_map = {
        "gzz": "E",
        "gz": "mGal",
        "amp": "mGal/m",
        "prediction": "mGal",
        "density_model": "kg/m^3",
        "body_response": "",
    }
    return unit_map.get(str(field_name), "")


def _display_cmap(_cmap_name: str = "jet"):
    """Match the GUI display palette for scalar volume outputs."""
    return cm.get_cmap("bwr")


def _input_display_cmap(input_name: str):
    """The GUI keeps the Gzz input in jet while using bwr elsewhere."""
    return cm.get_cmap("jet") if str(input_name) == "gzz" else _display_cmap("jet")


def _build_publication_norm(values: np.ndarray, symmetric: bool):
    """Build the same robust scale used by the GUI publication exports."""
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return colors.Normalize(vmin=0.0, vmax=1.0), "neither"

    if symmetric:
        abs_values = np.abs(finite_values)
        peak = max(float(np.max(abs_values)), 1e-12)
        robust_limit = float(np.percentile(abs_values, PAPER_CLIP_PERCENTILE))
        limit = max(robust_limit, peak * 0.15, 1e-12)
        extend = "both" if limit < peak * 0.999 else "neither"
        return colors.Normalize(vmin=-limit, vmax=limit), extend

    data_min = float(np.min(finite_values))
    data_max = float(np.max(finite_values))
    vmin = float(np.percentile(finite_values, PAPER_LINEAR_LOWER_PERCENTILE))
    vmax = float(np.percentile(finite_values, PAPER_CLIP_PERCENTILE))
    if abs(vmax - vmin) < 1e-12:
        vmin, vmax = data_min, data_max
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1e-12

    clipped = vmin > data_min + 1e-12 or vmax < data_max - 1e-12
    return colors.Normalize(vmin=vmin, vmax=vmax), ("both" if clipped else "neither")


def _build_body_response_projection_norm(values: np.ndarray):
    """Center body-response colorbars so quiet background stays near white."""
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        limit = 1.0
        return colors.Normalize(vmin=-limit, vmax=limit), "neither"

    abs_values = np.abs(finite_values)
    peak = max(float(np.max(abs_values)), 1e-12)
    robust_limit = float(np.percentile(abs_values, PAPER_CLIP_PERCENTILE))
    limit = max(robust_limit, peak * 0.15, 1e-12)
    extend = "both" if limit < peak * 0.999 else "neither"
    return colors.Normalize(vmin=-limit, vmax=limit), extend


def _format_signed_colorbar_tick(value: float, _position=None) -> str:
    """Format compact tick labels with an ASCII minus sign."""
    numeric = float(value)
    if not np.isfinite(numeric):
        return ""
    magnitude = abs(numeric)
    if magnitude < 1e-12:
        return "0"

    sign = "-" if numeric < 0 else ""
    exponent = int(np.floor(np.log10(magnitude)))
    if magnitude >= 1e3 or magnitude < 1e-2:
        coefficient = magnitude / (10**exponent)
        if np.isclose(coefficient, 1.0, rtol=1e-6, atol=1e-12):
            return f"{sign}1e{exponent}"
        return f"{sign}{coefficient:.1f}e{exponent}"
    if magnitude >= 100:
        return f"{sign}{magnitude:.0f}"
    if magnitude >= 10:
        return f"{sign}{magnitude:.1f}"
    if magnitude >= 0.1:
        return f"{sign}{magnitude:.2f}".rstrip("0").rstrip(".")
    return f"{sign}{magnitude:.3f}".rstrip("0").rstrip(".")


def _colorbar_display_scale(field_name: str | None) -> float:
    """Scale Gzz tick labels to Eotvos without changing the plotted data."""
    return 1.0e4 if str(field_name) == "gzz" else 1.0


def _format_scaled_colorbar_tick(value: float, display_scale: float, position=None) -> str:
    scaled_value = float(value) * float(display_scale)
    if np.isclose(scaled_value, round(scaled_value), rtol=1e-9, atol=1e-9):
        return str(int(round(scaled_value)))
    return _format_signed_colorbar_tick(scaled_value, position)


def _nice_linear_colorbar_ticks(vmin: float, vmax: float, display_scale: float = 1.0, nbins: int = 8) -> np.ndarray:
    """Build evenly spaced ticks in display units, matching the GUI helper."""
    raw_min = float(vmin)
    raw_max = float(vmax)
    if not np.isfinite(raw_min) or not np.isfinite(raw_max):
        return np.array([0.0], dtype=np.float64)
    if abs(raw_max - raw_min) < 1e-12:
        return np.array([raw_min], dtype=np.float64)

    scale = float(display_scale) if abs(float(display_scale)) > 1e-12 else 1.0
    display_min = raw_min * scale
    display_max = raw_max * scale
    if display_min > display_max:
        display_min, display_max = display_max, display_min

    locator = ticker.MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10], min_n_ticks=5)
    ticks_display = np.asarray(locator.tick_values(display_min, display_max), dtype=np.float64)
    tolerance = max(abs(display_max - display_min), 1.0) * 1e-9
    ticks_display = ticks_display[
        (ticks_display >= display_min - tolerance) & (ticks_display <= display_max + tolerance)
    ]
    if ticks_display.size == 0:
        ticks_display = np.array([display_min, display_max], dtype=np.float64)

    if display_min < 0.0 < display_max and not np.any(np.isclose(ticks_display, 0.0, atol=tolerance, rtol=0.0)):
        ticks_display = np.sort(np.concatenate([ticks_display, np.array([0.0], dtype=np.float64)]))

    ticks = ticks_display / scale
    ticks[np.isclose(ticks, 0.0, atol=max(abs(raw_max - raw_min), 1.0) * 1e-12, rtol=0.0)] = 0.0
    return ticks


def _apply_colorbar_display(colorbar, field_name: str | None, tick_labelsize: float | None = None) -> None:
    """Apply the GUI colorbar labels, ticks, and top unit text."""
    display_scale = _colorbar_display_scale(field_name)
    colorbar.ax.tick_params(which="both", direction="in")
    norm = getattr(colorbar, "norm", None)
    formatter = ticker.FuncFormatter(_format_signed_colorbar_tick)
    if isinstance(norm, colors.Normalize):
        vmin = float(getattr(norm, "vmin", np.nan))
        vmax = float(getattr(norm, "vmax", np.nan))
        if np.isfinite(vmin) and np.isfinite(vmax):
            colorbar.set_ticks(_nice_linear_colorbar_ticks(vmin, vmax, display_scale=display_scale))
        if not np.isclose(display_scale, 1.0):
            formatter = ticker.FuncFormatter(lambda value, position: _format_scaled_colorbar_tick(value, display_scale, position))
        colorbar.formatter = formatter
        colorbar.ax.yaxis.set_major_formatter(formatter)
        colorbar.update_ticks()
    colorbar.set_label("")
    colorbar.ax.set_ylabel("")
    if tick_labelsize is not None:
        colorbar.ax.tick_params(labelsize=tick_labelsize)
    colorbar.ax.set_title(_field_unit_text(field_name or ""), fontsize=9, pad=6)


def _fixed_spatial_ticks(coords: np.ndarray, tick_step: float = 500.0) -> np.ndarray:
    values = np.asarray(coords, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.array([], dtype=np.float64)
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    step = max(float(tick_step), 1e-12)
    first = np.ceil(lower / step) * step
    last = np.floor(upper / step) * step
    if first <= last:
        ticks = np.arange(first, last + step * 0.5, step, dtype=np.float64)
    elif np.isclose(lower, upper):
        ticks = np.array([lower], dtype=np.float64)
    else:
        ticks = np.array([lower, upper], dtype=np.float64)
    ticks[np.isclose(ticks, 0.0, atol=step * 1e-12, rtol=0.0)] = 0.0
    return ticks


def _apply_coordinate_axis_limits(ax, *, x_coords: np.ndarray, y_coords: np.ndarray) -> None:
    x_values = np.asarray(x_coords, dtype=np.float64).reshape(-1)
    y_values = np.asarray(y_coords, dtype=np.float64).reshape(-1)
    if x_values.size:
        ax.set_xlim(float(np.nanmin(x_values)), float(np.nanmax(x_values)))
    if y_values.size:
        ax.set_ylim(float(np.nanmin(y_values)), float(np.nanmax(y_values)))


def _apply_plan_view_spatial_ticks(ax, x_coords: np.ndarray, y_coords: np.ndarray, tick_step: float = 500.0) -> None:
    _apply_coordinate_axis_limits(ax, x_coords=x_coords, y_coords=y_coords)
    x_ticks = _fixed_spatial_ticks(x_coords, tick_step=tick_step)
    y_ticks = _fixed_spatial_ticks(y_coords, tick_step=tick_step)
    if x_ticks.size:
        ax.set_xticks(x_ticks)
    if y_ticks.size:
        ax.set_yticks(y_ticks)


def _apply_section_horizontal_spatial_ticks(ax, coords: np.ndarray, tick_step: float = 500.0) -> None:
    values = np.asarray(coords, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size:
        ax.set_xlim(float(np.min(finite)), float(np.max(finite)))
        ax.set_xticks(_fixed_spatial_ticks(finite, tick_step=tick_step))


def _new_standard_2d_figure():
    return plt.subplots(figsize=STANDARD_2D_SLICE_FIGSIZE, dpi=STANDARD_2D_SLICE_DPI, facecolor="white")


def _save_figure(figure, path: Path, dpi: int = STANDARD_2D_SLICE_DPI) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(path), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _set_3d_axes_style(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(float(np.min(y)), float(np.max(y)))
    ax.set_zlim(float(np.max(z)), float(np.min(z)))
    ax.set_box_aspect((float(np.ptp(x)), float(np.ptp(y)), float(np.ptp(z))))
    ax.view_init(elev=THREE_D_VIEW_ELEV, azim=THREE_D_VIEW_AZIM)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Depth (m)")
    ax.set_xticks(_fixed_spatial_ticks(x, tick_step=800.0))
    ax.set_yticks(_fixed_spatial_ticks(y, tick_step=800.0))
    ax.set_zticks(_fixed_spatial_ticks(z, tick_step=800.0))
    ax.xaxis.pane.set_facecolor(THREE_D_PANE_FACE_COLOR)
    ax.yaxis.pane.set_facecolor(THREE_D_PANE_FACE_COLOR)
    ax.zaxis.pane.set_facecolor(THREE_D_PANE_FACE_COLOR)
    ax.xaxis.pane.set_edgecolor(THREE_D_PANE_EDGE_COLOR)
    ax.yaxis.pane.set_edgecolor(THREE_D_PANE_EDGE_COLOR)
    ax.zaxis.pane.set_edgecolor(THREE_D_PANE_EDGE_COLOR)
    ax.grid(True)


def _apply_gui_dense_3d_axis_style(
    ax,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    *,
    elev: float = THREE_D_VIEW_ELEV,
    azim: float = THREE_D_VIEW_AZIM,
) -> None:
    x_values = np.asarray(x_coords, dtype=np.float64).reshape(-1)
    y_values = np.asarray(y_coords, dtype=np.float64).reshape(-1)
    z_display = _display_z_coords(z_coords)

    if x_values.size == 0:
        x_values = np.array([0.0, 1.0], dtype=np.float64)
    if y_values.size == 0:
        y_values = np.array([0.0, 1.0], dtype=np.float64)
    if z_display.size == 0:
        z_display = np.array([0.0, 1.0], dtype=np.float64)

    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    z_min = float(np.min(z_display))
    z_max = float(np.max(z_display))

    if np.isclose(x_min, x_max):
        x_max = x_min + 1.0
    if np.isclose(y_min, y_max):
        y_max = y_min + 1.0
    if np.isclose(z_min, z_max):
        z_max = z_min + 1.0

    ax.set_title("")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("Depth (m)", rotation=90)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_max, z_min)

    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor(THREE_D_PANE_FACE_COLOR)
    ax.yaxis.pane.set_facecolor(THREE_D_PANE_FACE_COLOR)
    ax.zaxis.pane.set_facecolor(THREE_D_PANE_FACE_COLOR)
    ax.xaxis.pane.set_edgecolor(THREE_D_PANE_EDGE_COLOR)
    ax.yaxis.pane.set_edgecolor(THREE_D_PANE_EDGE_COLOR)
    ax.zaxis.pane.set_edgecolor(THREE_D_PANE_EDGE_COLOR)

    ax.xaxis._axinfo["grid"]["linewidth"] = 0.5
    ax.yaxis._axinfo["grid"]["linewidth"] = 0.5
    ax.zaxis._axinfo["grid"]["linewidth"] = 0.5
    ax.xaxis._axinfo["grid"]["color"] = (0.3, 0.3, 0.3, 0.6)
    ax.yaxis._axinfo["grid"]["color"] = (0.3, 0.3, 0.3, 0.6)
    ax.zaxis._axinfo["grid"]["color"] = (0.3, 0.3, 0.3, 0.6)
    ax.grid(True, alpha=0.35)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.tick_params(axis="both", labelsize=8)
    ax.tick_params(axis="z", labelsize=8)

    x_grid = np.linspace(x_min, x_max, THREE_D_GRID_NUM)
    y_grid = np.linspace(y_min, y_max, THREE_D_GRID_NUM)
    z_grid = np.linspace(z_min, z_max, THREE_D_GRID_NUM)

    x_top, y_top = np.meshgrid(x_grid, y_grid)
    z_top = np.full_like(x_top, z_min)
    ax.plot_wireframe(x_top, y_top, z_top, color=THREE_D_GRID_COLOR, linewidth=0.3)

    x_back, z_back = np.meshgrid(x_grid, z_grid)
    y_back = np.full_like(x_back, y_min)
    ax.plot_wireframe(x_back, y_back, z_back, color=THREE_D_GRID_COLOR, linewidth=0.3)

    y_left, z_left = np.meshgrid(y_grid, z_grid)
    x_left = np.full_like(y_left, x_min)
    ax.plot_wireframe(x_left, y_left, z_left, color=THREE_D_GRID_COLOR, linewidth=0.3)

    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    z_span = max(z_max - z_min, 1.0)
    max_span = max(x_span, y_span, z_span)
    z_scale = max(1.0, max_span / z_span * 0.6) if z_span > 0 else 1.0
    ax.set_box_aspect((x_span / max_span, y_span / max_span, z_span / max_span * z_scale))
    ax.view_init(elev=elev, azim=azim)


def _add_3d_colorbar(fig, ax, mappable, field_name: str | None) -> None:
    fig.subplots_adjust(right=THREE_D_COLORBAR_LAYOUT["subplot_right"])
    colorbar = fig.colorbar(
        mappable,
        ax=ax,
        fraction=THREE_D_COLORBAR_LAYOUT["fraction"],
        pad=THREE_D_COLORBAR_LAYOUT["pad"],
        shrink=THREE_D_COLORBAR_LAYOUT["shrink"],
        aspect=THREE_D_COLORBAR_LAYOUT["aspect"],
    )
    _apply_colorbar_display(colorbar, field_name, tick_labelsize=9)


def _slice_indices_from_metadata(result: InferenceResult, metadata: dict) -> tuple[int, int, int]:
    slices = metadata.get("slices", {})
    x_idx = _coord_index(result.x, float(slices.get("x", result.x[len(result.x) // 2])))
    y_idx = _coord_index(result.y, float(slices.get("y", result.y[len(result.y) // 2])))
    z_idx = _coord_index(result.z, float(slices.get("z", result.z[len(result.z) // 2])))
    return x_idx, y_idx, z_idx


def _plane_slice(volume: np.ndarray, plane_key: str, x_idx: int, y_idx: int, z_idx: int) -> np.ndarray:
    if plane_key == "xy":
        return np.asarray(volume[:, :, z_idx], dtype=np.float32)
    if plane_key == "xz":
        return np.asarray(volume[:, y_idx, :], dtype=np.float32)
    if plane_key == "yz":
        return np.asarray(volume[x_idx, :, :], dtype=np.float32)
    raise ValueError(f"Unsupported plane key: {plane_key}")


def _binary_slice_bbox(slice_data: np.ndarray, axis_a: np.ndarray, axis_b: np.ndarray) -> tuple[float, float, float, float] | None:
    finite = np.asarray(slice_data, dtype=np.float32)
    abs_values = np.abs(finite[np.isfinite(finite)])
    if abs_values.size == 0 or float(np.max(abs_values)) <= 0.0:
        return None
    mask = np.abs(finite) >= max(float(np.max(abs_values)) * 0.3, 1e-12)
    if not np.any(mask):
        return None
    positions = np.argwhere(mask)
    a_min = float(axis_a[int(np.min(positions[:, 0]))])
    a_max = float(axis_a[int(np.max(positions[:, 0]))])
    b_min = float(axis_b[int(np.min(positions[:, 1]))])
    b_max = float(axis_b[int(np.max(positions[:, 1]))])
    return a_min, a_max, b_min, b_max


def _draw_slice_bbox(ax, bbox: tuple[float, float, float, float] | None) -> None:
    if bbox is None:
        return
    a_min, a_max, b_min, b_max = bbox
    ax.plot([a_min, a_max, a_max, a_min, a_min], [b_min, b_min, b_max, b_max, b_min], color="black", linewidth=2.2)


def _mask_to_contour_polylines(mask: np.ndarray, axis_a: np.ndarray, axis_b: np.ndarray) -> list[np.ndarray]:
    mask_values = np.asarray(mask, dtype=bool)
    if mask_values.ndim != 2 or mask_values.size == 0 or not np.any(mask_values):
        return []

    axis_a = np.asarray(axis_a, dtype=np.float64).reshape(-1)
    axis_b = np.asarray(axis_b, dtype=np.float64).reshape(-1)
    if axis_a.size != mask_values.shape[0]:
        axis_a = np.arange(mask_values.shape[0], dtype=np.float64)
    if axis_b.size != mask_values.shape[1]:
        axis_b = np.arange(mask_values.shape[1], dtype=np.float64)

    def axis_edges(axis: np.ndarray) -> np.ndarray:
        if axis.size == 1:
            return np.asarray([axis[0] - 0.5, axis[0] + 0.5], dtype=np.float64)
        mids = (axis[:-1] + axis[1:]) / 2.0
        first = axis[0] - (mids[0] - axis[0])
        last = axis[-1] + (axis[-1] - mids[-1])
        return np.concatenate(([first], mids, [last])).astype(np.float64)

    a_edges = axis_edges(axis_a)
    b_edges = axis_edges(axis_b)
    polylines: list[np.ndarray] = []
    rows, cols = mask_values.shape
    for row in range(rows):
        a_min = float(a_edges[row])
        a_max = float(a_edges[row + 1])
        for col in range(cols):
            if not mask_values[row, col]:
                continue
            b_min = float(b_edges[col])
            b_max = float(b_edges[col + 1])
            if row == 0 or not mask_values[row - 1, col]:
                polylines.append(np.asarray([[a_min, b_min], [a_min, b_max]], dtype=np.float64))
            if row == rows - 1 or not mask_values[row + 1, col]:
                polylines.append(np.asarray([[a_max, b_min], [a_max, b_max]], dtype=np.float64))
            if col == 0 or not mask_values[row, col - 1]:
                polylines.append(np.asarray([[a_min, b_min], [a_max, b_min]], dtype=np.float64))
            if col == cols - 1 or not mask_values[row, col + 1]:
                polylines.append(np.asarray([[a_min, b_max], [a_max, b_max]], dtype=np.float64))
    return polylines


def _slice_boundary_polylines(
    boundary_volume: np.ndarray | None,
    result: InferenceResult,
    plane_key: str,
    indices: tuple[int, int, int],
) -> list[np.ndarray]:
    if boundary_volume is None:
        return []
    x_idx, y_idx, z_idx = indices
    if plane_key == "xy":
        axes = (result.x, result.y)
        boundary_slice = np.asarray(boundary_volume[:, :, z_idx], dtype=np.float32)
    elif plane_key == "xz":
        axes = (result.x, result.z)
        boundary_slice = np.asarray(boundary_volume[:, y_idx, :], dtype=np.float32)
    elif plane_key == "yz":
        axes = (result.y, result.z)
        boundary_slice = np.asarray(boundary_volume[x_idx, :, :], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported plane key: {plane_key}")
    mask = np.isfinite(boundary_slice) & (np.abs(boundary_slice) > 1e-6)
    return _mask_to_contour_polylines(mask, axes[0], axes[1])


def _draw_contour_polylines(ax, polylines: list[np.ndarray], *, linewidth: float = 2.0) -> None:
    for polyline in polylines:
        points = np.asarray(polyline, dtype=np.float64)
        if points.ndim == 2 and points.shape[0] >= 2:
            ax.plot(points[:, 0], points[:, 1], color="black", linewidth=linewidth)


def _save_gui_2d_map(
    path: Path,
    axis_a: np.ndarray,
    axis_b: np.ndarray,
    data: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    cmap,
    norm=None,
    extend: str = "neither",
    field_name: str | None = None,
    invert_y: bool = False,
    equal_aspect: bool = False,
    bbox: tuple[float, float, float, float] | None = None,
    boundary_polylines: list[np.ndarray] | None = None,
) -> Path:
    fig, ax = _new_standard_2d_figure()
    pc = ax.pcolormesh(axis_a, axis_b, np.asarray(data).T, cmap=cmap, shading="auto", norm=norm)
    if boundary_polylines:
        _draw_contour_polylines(ax, boundary_polylines, linewidth=2.2)
    if invert_y:
        ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if equal_aspect:
        ax.set_aspect("equal")
        _apply_plan_view_spatial_ticks(ax, axis_a, axis_b)
    else:
        _apply_section_horizontal_spatial_ticks(ax, axis_a)
    colorbar = fig.colorbar(pc, ax=ax, extend=extend, extendrect=True)
    _apply_colorbar_display(colorbar, field_name, tick_labelsize=9)
    return _save_figure(fig, path)


def _component_bboxes(volume: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray, ratio: float = 0.30) -> list[tuple[float, float, float, float, float, float]]:
    values = np.asarray(volume, dtype=np.float32)
    abs_values = np.abs(values[np.isfinite(values)])
    if abs_values.size == 0 or float(np.max(abs_values)) <= 0.0:
        return []
    threshold = max(float(np.max(abs_values)) * float(ratio), 1e-12)
    labels = measure.label(np.abs(values) >= threshold, connectivity=1)
    bboxes: list[tuple[float, float, float, float, float, float]] = []
    for label_value in range(1, int(labels.max()) + 1):
        positions = np.argwhere(labels == label_value)
        if positions.shape[0] < 4:
            continue
        x_min = float(x[int(np.min(positions[:, 0]))])
        x_max = float(x[int(np.max(positions[:, 0]))])
        y_min = float(y[int(np.min(positions[:, 1]))])
        y_max = float(y[int(np.max(positions[:, 1]))])
        z_min = float(z[int(np.min(positions[:, 2]))])
        z_max = float(z[int(np.max(positions[:, 2]))])
        bboxes.append((x_min, x_max, y_min, y_max, z_min, z_max))
    return bboxes


def _draw_3d_bbox(ax, bbox: tuple[float, float, float, float, float, float], linewidth: float = 2.0) -> None:
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    corners = {
        "000": (x_min, y_min, z_min),
        "100": (x_max, y_min, z_min),
        "010": (x_min, y_max, z_min),
        "110": (x_max, y_max, z_min),
        "001": (x_min, y_min, z_max),
        "101": (x_max, y_min, z_max),
        "011": (x_min, y_max, z_max),
        "111": (x_max, y_max, z_max),
    }
    edges = [
        ("000", "100"),
        ("000", "010"),
        ("100", "110"),
        ("010", "110"),
        ("001", "101"),
        ("001", "011"),
        ("101", "111"),
        ("011", "111"),
        ("000", "001"),
        ("100", "101"),
        ("010", "011"),
        ("110", "111"),
    ]
    for start, end in edges:
        xs = [corners[start][0], corners[end][0]]
        ys = [corners[start][1], corners[end][1]]
        zs = [corners[start][2], corners[end][2]]
        ax.plot(xs, ys, zs, color="black", linewidth=linewidth)


def _draw_3d_slice_bbox(
    ax,
    plane_key: str,
    bbox_2d: tuple[float, float, float, float] | None,
    fixed_coord: float,
    linewidth: float = 2.4,
) -> None:
    if bbox_2d is None:
        return
    a_min, a_max, b_min, b_max = bbox_2d
    if plane_key == "xy":
        xs = [a_min, a_max, a_max, a_min, a_min]
        ys = [b_min, b_min, b_max, b_max, b_min]
        zs = [fixed_coord] * 5
    elif plane_key == "xz":
        xs = [a_min, a_max, a_max, a_min, a_min]
        ys = [fixed_coord] * 5
        zs = [b_min, b_min, b_max, b_max, b_min]
    elif plane_key == "yz":
        xs = [fixed_coord] * 5
        ys = [a_min, a_max, a_max, a_min, a_min]
        zs = [b_min, b_min, b_max, b_max, b_min]
    else:
        raise ValueError(f"Unsupported plane key: {plane_key}")
    ax.plot(xs, ys, zs, color="black", linewidth=linewidth)


def _draw_3d_boundary_polylines(
    ax,
    polylines: list[np.ndarray],
    *,
    plane_key: str,
    fixed_coord: float,
    normal_offset: float = 0.0,
    linewidth: float = 2.4,
) -> None:
    for polyline in polylines:
        points = np.asarray(polyline, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 2:
            continue
        if plane_key == "xy":
            xs = points[:, 0]
            ys = points[:, 1]
            zs = np.full(points.shape[0], float(fixed_coord) - float(normal_offset), dtype=np.float64)
        elif plane_key == "xz":
            xs = points[:, 0]
            ys = np.full(points.shape[0], float(fixed_coord) - float(normal_offset), dtype=np.float64)
            zs = points[:, 1]
        elif plane_key == "yz":
            xs = np.full(points.shape[0], float(fixed_coord) - float(normal_offset), dtype=np.float64)
            ys = points[:, 0]
            zs = points[:, 1]
        else:
            raise ValueError(f"Unsupported plane key: {plane_key}")
        ax.plot(xs, ys, zs, color="black", linewidth=linewidth, zorder=1000)


def _add_signed_isosurface(
    ax,
    volume: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    norm,
    cmap,
    level_ratio: float = 0.30,
    signed: bool = True,
    alpha: float = 0.62,
    isosurface_color_value: float | None = None,
) -> bool:
    values = np.asarray(volume, dtype=np.float32)
    finite_abs = np.abs(values[np.isfinite(values)])
    if finite_abs.size == 0:
        return False
    limit = float(np.max(finite_abs))
    if limit <= 0.0:
        return False

    spacing = (_axis_step(x), _axis_step(y), _axis_step(z))
    origin = np.array([float(x[0]), float(y[0]), float(z[0])], dtype=np.float64)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    levels = [float(level_ratio) * limit]
    if signed:
        levels.append(-float(level_ratio) * limit)

    drawn = False
    for signed_level in levels:
        if signed_level >= 0:
            surface_volume = values
            level = signed_level
        else:
            surface_volume = -values
            level = abs(signed_level)
        vmin = float(np.nanmin(surface_volume))
        vmax = float(np.nanmax(surface_volume))
        if not (vmin < level < vmax):
            continue
        verts, faces, _, _ = measure.marching_cubes(surface_volume, level=level, spacing=spacing)
        verts = verts + origin
        color_value = signed_level if isosurface_color_value is None else float(isosurface_color_value)
        if not signed and isosurface_color_value is None:
            color_value = level
        face_color = mappable.to_rgba(color_value)
        mesh = Poly3DCollection(verts[faces], facecolors=[face_color], edgecolor="none", alpha=alpha)
        ax.add_collection3d(mesh)
        drawn = True
    return drawn


def _save_gui_3d_isosurface(
    path: Path,
    volume: np.ndarray,
    result: InferenceResult,
    *,
    field_name: str,
    norm,
    cmap,
    signed: bool,
    level_ratio: float = 0.30,
    bboxes: list[tuple[float, float, float, float, float, float]] | None = None,
    boundary_volume: np.ndarray | None = None,
    isosurface_color_value: float | None = None,
) -> Path:
    fig = plt.figure(figsize=THREE_D_FIGSIZE, dpi=THREE_D_DPI, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    drawn = _add_signed_isosurface(
        ax,
        volume,
        result.x,
        result.y,
        result.z,
        norm=norm,
        cmap=cmap,
        level_ratio=level_ratio,
        signed=signed,
        alpha=0.70 if field_name == "density_model" else 0.58,
        isosurface_color_value=isosurface_color_value,
    )
    if not drawn:
        ax.text2D(0.5, 0.5, "No positive isosurface", transform=ax.transAxes, ha="center", va="center")
    _set_3d_axes_style(ax, result.x, result.y, result.z)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(np.asarray(volume, dtype=np.float32))
    _add_3d_colorbar(fig, ax, mappable, field_name)
    return _save_figure(fig, path, dpi=THREE_D_DPI)


def _save_gui_raw_body_response_isosurface(
    path: Path,
    volume: np.ndarray,
    result: InferenceResult,
    *,
    threshold_ratio: float = 0.30,
) -> Path:
    values = np.asarray(volume, dtype=np.float32)
    finite_mask = np.isfinite(values)
    fig = plt.figure(figsize=(9.4, 6.9), dpi=THREE_D_DPI, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    if not np.any(finite_mask):
        ax.text2D(0.5, 0.5, "No valid data", transform=ax.transAxes, ha="center", va="center")
        _apply_gui_dense_3d_axis_style(ax, result.x, result.y, result.z)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=THREE_D_DPI, facecolor="white")
        plt.close(fig)
        return path

    finite_values = values[finite_mask]
    vmin = float(np.min(finite_values))
    vmax = float(np.max(finite_values))
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-12

    value_range = vmax - vmin
    center = (vmax + vmin) / 2.0
    pos_level = center + value_range * float(threshold_ratio) / 2.0

    x_values = np.asarray(result.x, dtype=np.float64).reshape(-1)
    y_values = np.asarray(result.y, dtype=np.float64).reshape(-1)
    z_values = np.asarray(result.z, dtype=np.float64).reshape(-1)
    x_min = float(x_values.min()) if x_values.size else 0.0
    y_min = float(y_values.min()) if y_values.size else 0.0
    z_display_values = _display_z_coords(z_values if z_values.size else np.arange(values.shape[2], dtype=np.float64))
    z_min = float(z_display_values.min()) if z_display_values.size else 0.0

    spacing = (_axis_step(z_values), _axis_step(y_values), _axis_step(x_values))
    volume_zyx = np.transpose(np.nan_to_num(values, nan=center), (2, 1, 0))
    data_max = float(np.max(volume_zyx))
    cmap = _display_cmap("magma")
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    ax.clear()
    ax.set_facecolor("none")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
    ax.yaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))
    ax.zaxis.pane.set_edgecolor((0.8, 0.8, 0.8, 0.3))

    surfaces_drawn = 0
    if pos_level < data_max:
        try:
            verts, faces, _normals, surface_values = measure.marching_cubes(volume_zyx, level=pos_level, spacing=spacing)
            verts_plot = np.zeros_like(verts)
            verts_plot[:, 0] = verts[:, 2] + x_min
            verts_plot[:, 1] = verts[:, 1] + y_min
            verts_plot[:, 2] = verts[:, 0] + z_min

            face_values = surface_values[faces].mean(axis=1)
            face_colors = cmap(norm(face_values))
            mesh = Poly3DCollection(verts_plot[faces], alpha=0.85)
            mesh.set_facecolor(face_colors)
            mesh.set_edgecolor("none")
            ax.add_collection3d(mesh)
            surfaces_drawn += 1
        except Exception:
            pass

    if surfaces_drawn == 0:
        ax.text(
            float(np.mean(x_values)) if x_values.size else 0.0,
            float(np.mean(y_values)) if y_values.size else 0.0,
            float(np.mean(z_display_values)) if z_display_values.size else 0.0,
            "No isosurface\n(adjust threshold)",
            ha="center",
            va="center",
            fontsize=10,
        )

    color_mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    color_mapper.set_array([vmin, vmax])
    colorbar = fig.colorbar(
        color_mapper,
        ax=ax,
        fraction=THREE_D_COLORBAR_LAYOUT["fraction"],
        pad=THREE_D_COLORBAR_LAYOUT["pad"],
        shrink=THREE_D_COLORBAR_LAYOUT["shrink"],
        aspect=int(THREE_D_COLORBAR_LAYOUT["aspect"]),
    )
    _apply_colorbar_display(colorbar, "body_response", tick_labelsize=8)

    _apply_gui_dense_3d_axis_style(ax, result.x, result.y, result.z)
    fig.subplots_adjust(left=0.02, right=THREE_D_COLORBAR_LAYOUT["subplot_right"], bottom=0.02, top=0.98)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=THREE_D_DPI, facecolor="white")
    plt.close(fig)
    return path


def _save_gui_3d_slice(
    path: Path,
    volume: np.ndarray,
    result: InferenceResult,
    plane_key: str,
    indices: tuple[int, int, int],
    *,
    field_name: str,
    norm,
    cmap,
    bbox: tuple[float, float, float, float] | None = None,
    boundary_polylines: list[np.ndarray] | None = None,
) -> Path:
    x_idx, y_idx, z_idx = indices
    fig = plt.figure(figsize=THREE_D_FIGSIZE, dpi=THREE_D_DPI, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    volume = np.asarray(volume, dtype=np.float32)
    face_cmap = cmap

    if plane_key == "xy":
        xx, yy = np.meshgrid(result.x, result.y, indexing="ij")
        zz = np.full_like(xx, result.z[z_idx], dtype=np.float64)
        colors_rgba = face_cmap(norm(volume[:, :, z_idx]))
        ax.plot_surface(xx, yy, zz, facecolors=colors_rgba, rstride=1, cstride=1, shade=False)
        if boundary_polylines:
            _draw_3d_boundary_polylines(
                ax,
                boundary_polylines,
                plane_key="xy",
                fixed_coord=float(result.z[z_idx]),
                normal_offset=0.04 * _axis_step(result.z),
            )
    elif plane_key == "xz":
        xx, zz = np.meshgrid(result.x, result.z, indexing="ij")
        yy = np.full_like(xx, result.y[y_idx], dtype=np.float64)
        colors_rgba = face_cmap(norm(volume[:, y_idx, :]))
        ax.plot_surface(xx, yy, zz, facecolors=colors_rgba, rstride=1, cstride=1, shade=False)
        if boundary_polylines:
            _draw_3d_boundary_polylines(
                ax,
                boundary_polylines,
                plane_key="xz",
                fixed_coord=float(result.y[y_idx]),
                normal_offset=0.04 * _axis_step(result.y),
            )
    elif plane_key == "yz":
        yy, zz = np.meshgrid(result.y, result.z, indexing="ij")
        xx = np.full_like(yy, result.x[x_idx], dtype=np.float64)
        colors_rgba = face_cmap(norm(volume[x_idx, :, :]))
        ax.plot_surface(xx, yy, zz, facecolors=colors_rgba, rstride=1, cstride=1, shade=False)
        if boundary_polylines:
            _draw_3d_boundary_polylines(
                ax,
                boundary_polylines,
                plane_key="yz",
                fixed_coord=float(result.x[x_idx]),
                normal_offset=0.04 * _axis_step(result.x),
            )
    else:
        raise ValueError(f"Unsupported plane key: {plane_key}")

    _set_3d_axes_style(ax, result.x, result.y, result.z)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(volume)
    _add_3d_colorbar(fig, ax, mappable, field_name)
    return _save_figure(fig, path, dpi=THREE_D_DPI)


def _plane_axes_and_bbox(result: InferenceResult, density: np.ndarray | None, plane_key: str, indices: tuple[int, int, int]):
    x_idx, y_idx, z_idx = indices
    if plane_key == "xy":
        axes = (result.x, result.y)
        labels = ("X (m)", "Y (m)")
        invert_y = False
        equal_aspect = True
        density_slice = density[:, :, z_idx] if density is not None else None
    elif plane_key == "xz":
        axes = (result.x, result.z)
        labels = ("X (m)", "Depth (m)")
        invert_y = True
        equal_aspect = False
        density_slice = density[:, y_idx, :] if density is not None else None
    elif plane_key == "yz":
        axes = (result.y, result.z)
        labels = ("Y (m)", "Depth (m)")
        invert_y = True
        equal_aspect = False
        density_slice = density[x_idx, :, :] if density is not None else None
    else:
        raise ValueError(f"Unsupported plane key: {plane_key}")
    bbox = _binary_slice_bbox(density_slice, axes[0], axes[1]) if density_slice is not None else None
    return axes, labels, invert_y, equal_aspect, bbox


def _save_prediction_2d_slices(
    result: InferenceResult,
    output_dir: Path,
    indices: tuple[int, int, int],
    prediction_norm,
    prediction_extend: str,
    boundary_volume: np.ndarray | None,
) -> list[Path]:
    paths: list[Path] = []
    files = {
        "xy": "predicted_gravity_field_slice_xy.png",
        "xz": "predicted_gravity_field_slice_xz.png",
        "yz": "predicted_gravity_field_slice_yz.png",
    }
    for plane_key, filename in files.items():
        axes, labels, invert_y, equal_aspect, _ = _plane_axes_and_bbox(result, None, plane_key, indices)
        boundary_polylines = _slice_boundary_polylines(boundary_volume, result, plane_key, indices)
        paths.append(
            _save_gui_2d_map(
                output_dir / "predicted_gravity_field_slices_2d" / filename,
                axes[0],
                axes[1],
                _plane_slice(result.prediction, plane_key, *indices),
                xlabel=labels[0],
                ylabel=labels[1],
                cmap=_display_cmap("jet"),
                norm=prediction_norm,
                extend=prediction_extend,
                field_name="prediction",
                invert_y=invert_y,
                equal_aspect=equal_aspect,
                boundary_polylines=boundary_polylines,
            )
        )
    return paths


def _save_body_response_2d_slices(
    result: InferenceResult,
    output_dir: Path,
    indices: tuple[int, int, int],
    response_norm,
    response_extend: str,
    boundary_volume: np.ndarray | None,
) -> list[Path]:
    paths: list[Path] = []
    for plane_key in ("xy", "xz", "yz"):
        axes, labels, invert_y, equal_aspect, _ = _plane_axes_and_bbox(result, None, plane_key, indices)
        boundary_polylines = _slice_boundary_polylines(boundary_volume, result, plane_key, indices)
        paths.append(
            _save_gui_2d_map(
                output_dir / "gradient_magnitude_slices_2d" / f"gradient_magnitude_slice_{plane_key}.png",
                axes[0],
                axes[1],
                _plane_slice(result.source_response, plane_key, *indices),
                xlabel=labels[0],
                ylabel=labels[1],
                cmap=_display_cmap("magma"),
                norm=response_norm,
                extend=response_extend,
                field_name="body_response",
                invert_y=invert_y,
                equal_aspect=equal_aspect,
                boundary_polylines=boundary_polylines,
            )
        )
    return paths


def _save_volume_3d_slice_set(
    result: InferenceResult,
    output_dir: Path,
    volume: np.ndarray,
    indices: tuple[int, int, int],
    *,
    field_name: str,
    norm,
    cmap,
    prefix: str,
    include_bboxes: bool,
    boundary_volume: np.ndarray | None,
) -> list[Path]:
    paths: list[Path] = []
    for plane_key in ("xy", "xz", "yz"):
        _ = include_bboxes
        boundary_polylines = _slice_boundary_polylines(boundary_volume, result, plane_key, indices)
        paths.append(
            _save_gui_3d_slice(
                output_dir / f"{prefix}_{plane_key}.png",
                volume,
                result,
                plane_key,
                indices,
                field_name=field_name,
                norm=norm,
                cmap=cmap,
                boundary_polylines=boundary_polylines,
            )
        )
    return paths


def _save_true_voxel_outputs(
    result: InferenceResult,
    output_dir: Path,
    indices: tuple[int, int, int],
) -> list[Path]:
    if result.true_density is None:
        return []

    paths: list[Path] = []
    density_norm, _ = _build_publication_norm(result.true_density, symmetric=True)
    density_cmap = _display_cmap("jet")
    test_root = output_dir / "synthetic_model"
    corresponding_root = test_root / "synthetic_model_slices"
    paths.append(
        _save_gui_3d_isosurface(
            test_root / "synthetic_model_3d_isosurface.png",
            result.true_density,
            result,
            field_name="density_model",
            norm=density_norm,
            cmap=density_cmap,
            signed=True,
            bboxes=None,
        )
    )

    x_idx, y_idx, z_idx = indices
    for plane_key in ("xy", "xz", "yz"):
        axes, labels, invert_y, equal_aspect, _ = _plane_axes_and_bbox(result, None, plane_key, indices)
        paths.append(
            _save_gui_2d_map(
                corresponding_root / "2d_slices" / f"synthetic_model_2d_slice_{plane_key}.png",
                axes[0],
                axes[1],
                _plane_slice(result.true_density, plane_key, *indices),
                xlabel=labels[0],
                ylabel=labels[1],
                cmap=density_cmap,
                norm=density_norm,
                extend="neither",
                field_name="density_model",
                invert_y=invert_y,
                equal_aspect=equal_aspect,
            )
        )
        paths.append(
            _save_gui_3d_slice(
                corresponding_root / "3d_slices" / f"synthetic_model_3d_slice_{plane_key}.png",
                result.true_density,
                result,
                plane_key,
                indices,
                field_name="density_model",
                norm=density_norm,
                cmap=density_cmap,
            )
        )

    position_info = {
        "volume_key": "density_model",
        "volume_label": "Test Voxel Model",
        "x_index": int(x_idx),
        "y_index": int(y_idx),
        "z_index": int(z_idx),
        "x_coord": float(result.x[x_idx]),
        "y_coord": float(result.y[y_idx]),
        "z_coord": float(result.z[z_idx]),
        "depth_coord": float(result.z[z_idx]),
        "threshold_ratio": 0.3,
        "show_negative": True,
    }
    position_path = corresponding_root / "position_info.json"
    position_path.parent.mkdir(parents=True, exist_ok=True)
    with open(position_path, "w", encoding="utf-8") as file_obj:
        json.dump(position_info, file_obj, ensure_ascii=False, indent=2)
    paths.append(position_path)
    return paths


def save_reference_case_outputs(
    result: InferenceResult,
    observed_gzz: np.ndarray,
    metadata: dict,
    output_dir: Path,
    *,
    synthetic: bool,
) -> list[Path]:
    """Save figures in the same folder layout used by the GUI reference examples."""
    paths: list[Path] = []
    indices = _slice_indices_from_metadata(result, metadata)
    prediction_norm, prediction_extend = _build_publication_norm(result.prediction, symmetric=True)
    response_norm, response_extend = _build_body_response_projection_norm(result.source_response)
    prediction_cmap = _display_cmap("jet")
    response_cmap = _display_cmap("magma")
    raw_source_response = result.source_response_raw if result.source_response_raw is not None else result.source_response

    paths.append(
        _save_gui_2d_map(
            output_dir / "observed_gravity_gradient" / "observed_gravity_gradient_gzz.png",
            result.x,
            result.y,
            observed_gzz,
            xlabel="X (m)",
            ylabel="Y (m)",
            cmap=_input_display_cmap("gzz"),
            field_name="gzz",
            equal_aspect=True,
        )
    )
    boundary_volume = result.true_density

    paths.extend(_save_prediction_2d_slices(result, output_dir, indices, prediction_norm, prediction_extend, boundary_volume))
    paths.append(
        _save_gui_3d_isosurface(
            output_dir / "predicted_3d_gravity_field" / "predicted_3d_gravity_field.png",
            result.prediction,
            result,
            field_name="prediction",
            norm=prediction_norm,
            cmap=prediction_cmap,
            signed=True,
            bboxes=None,
            boundary_volume=None,
        )
    )
    paths.extend(
        _save_volume_3d_slice_set(
            result,
            output_dir / "predicted_gravity_field_slices_3d",
            result.prediction,
            indices,
            field_name="prediction",
            norm=prediction_norm,
            cmap=prediction_cmap,
            prefix="predicted_gravity_field_3d_slice",
            include_bboxes=False,
            boundary_volume=boundary_volume,
        )
    )
    paths.extend(_save_body_response_2d_slices(result, output_dir, indices, response_norm, response_extend, boundary_volume))
    paths.append(
        _save_gui_raw_body_response_isosurface(
            output_dir / "gradient_magnitude_3d_field" / "gradient_magnitude_3d_field.png",
            raw_source_response,
            result,
            threshold_ratio=0.30,
        )
    )
    paths.extend(
        _save_volume_3d_slice_set(
            result,
            output_dir / "gradient_magnitude_slices_3d",
            result.source_response,
            indices,
            field_name="body_response",
            norm=response_norm,
            cmap=response_cmap,
            prefix="gradient_magnitude_3d_slice",
            include_bboxes=False,
            boundary_volume=boundary_volume,
        )
    )

    if synthetic:
        paths.extend(_save_true_voxel_outputs(result, output_dir, indices))
    return paths


def save_model_and_input_figure(result: InferenceResult, observed_gzz: np.ndarray, metadata: dict, output_dir: Path) -> Path:
    target = output_dir / metadata["fig_input"]
    fig = plt.figure(figsize=(11.0, 5.0), dpi=180)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    _plot_abs_isosurface(
        ax1,
        result.true_density,
        (result.x, result.y, result.z),
        "(a) Synthetic model",
        cmap_name="viridis",
    )
    mesh = _plot_surface(ax2, result.x, result.y, observed_gzz, "(b) Observed Gzz with 3% noise", cmap_name="bwr", symmetric=True)
    fig.colorbar(mesh, ax=ax2, fraction=0.046, pad=0.04, label="Gzz")
    fig.suptitle(metadata["label"], fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(target, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return target


def save_prediction_gradient_figure(result: InferenceResult, metadata: dict, output_dir: Path) -> Path:
    target = output_dir / metadata["fig_prediction"]
    fig = plt.figure(figsize=(11.0, 5.0), dpi=180)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    _plot_abs_isosurface(
        ax1,
        result.prediction,
        (result.x, result.y, result.z),
        "(a) Predicted 3D gravity field",
        cmap_name="viridis",
    )
    _plot_abs_isosurface(
        ax2,
        result.gradient_magnitude,
        (result.x, result.y, result.z),
        "(b) Gradient magnitude",
        cmap_name="magma",
    )
    fig.suptitle(metadata["label"], fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(target, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return target


def save_synthetic_slice_figure(result: InferenceResult, metadata: dict, output_dir: Path) -> Path:
    target = output_dir / metadata["fig_slices"]
    coords = (result.x, result.y, result.z)
    volumes = [
        ("Model", result.true_density, "viridis", False),
        ("Prediction", result.prediction, "bwr", True),
        ("Gradient", result.gradient_magnitude, "magma", False),
    ]
    planes = [("z", metadata["slices"]["z"]), ("y", metadata["slices"]["y"]), ("x", metadata["slices"]["x"])]
    fig = plt.figure(figsize=(18.0, 12.0), dpi=160)
    panel = 1
    for plane, value in planes:
        for name, volume, cmap_name, symmetric in volumes:
            ax = fig.add_subplot(3, 6, panel, projection="3d")
            _plot_3d_slice(
                ax,
                volume,
                coords,
                plane,
                value,
                f"{name} 3D {plane}={value:g} m",
                cmap_name=cmap_name,
                symmetric=symmetric,
            )
            panel += 1
        for name, volume, cmap_name, symmetric in volumes:
            ax = fig.add_subplot(3, 6, panel)
            mesh = _plot_2d_slice(
                ax,
                volume,
                coords,
                plane,
                value,
                f"{name} 2D {plane}={value:g} m",
                cmap_name=cmap_name,
                symmetric=symmetric,
            )
            fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
            panel += 1
    fig.suptitle(metadata["label"] + " - 3D and 2D slices", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(target, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return target


def save_real_input_figure(x: np.ndarray, y: np.ndarray, gzz: np.ndarray, output_dir: Path) -> Path:
    target = output_dir / REAL_FIGURES["fig_input"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=180)
    mesh = _plot_surface(ax, x, y, gzz, "Observed airborne vertical gravity-gradient at Vinton", cmap_name="bwr", symmetric=True)
    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04, label="Gzz")
    fig.tight_layout()
    fig.savefig(target, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return target


def save_real_slice_figure(result: InferenceResult, output_dir: Path) -> Path:
    target = output_dir / REAL_FIGURES["fig_slices"]
    coords = (result.x, result.y, result.z)
    planes = [("y", REAL_FIGURES["slices"]["y"]), ("x", REAL_FIGURES["slices"]["x"])]
    volumes = [
        ("Prediction", result.prediction, "bwr", True),
        ("Gradient", result.gradient_magnitude, "magma", False),
    ]
    fig = plt.figure(figsize=(15.0, 8.0), dpi=160)
    panel = 1
    for plane, value in planes:
        for name, volume, cmap_name, symmetric in volumes:
            ax = fig.add_subplot(2, 4, panel, projection="3d")
            _plot_3d_slice(
                ax,
                volume,
                coords,
                plane,
                value,
                f"{name} 3D {plane}={value:g} m",
                cmap_name=cmap_name,
                symmetric=symmetric,
            )
            panel += 1
        for name, volume, cmap_name, symmetric in volumes:
            ax = fig.add_subplot(2, 4, panel)
            mesh = _plot_2d_slice(
                ax,
                volume,
                coords,
                plane,
                value,
                f"{name} 2D {plane}={value:g} m",
                cmap_name=cmap_name,
                symmetric=symmetric,
            )
            fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
            panel += 1
    fig.suptitle("Field data imaging results - 3D and 2D slices", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(target, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return target


def process_synthetic_case(
    case_name: str,
    data_dir: Path,
    checkpoint_path: Path,
    model_capacity: str | None,
    input_mode: str | None,
    output_dir: Path,
    *,
    surface_calibration: bool = False,
) -> tuple[InferenceResult, list[Path]]:
    case_dir = _case_data_dir(data_dir, case_name)
    file_prefix = _case_data_prefix(case_name)
    metadata = SYNTHETIC_CASES[case_name]
    true_gravity_bundle = _load_npz_volume(case_dir / f"{file_prefix}_true_gravity_field.npz", "prediction")
    true_density_bundle = _load_npz_volume(case_dir / f"{file_prefix}_true_voxel_model.npz", "density_model")

    _run_gui_real_data_inference(
        output_dir,
        checkpoint_path=checkpoint_path,
        model_capacity=model_capacity,
        input_mode=input_mode,
        real_gz_path=case_dir / f"{file_prefix}_gz.txt",
        real_gzz_path=case_dir / f"{file_prefix}_gzz.txt",
        surface_calibration=surface_calibration,
    )
    result, observed_gzz = _load_gui_real_data_result(
        case_name=case_name,
        true_gravity=true_gravity_bundle.volume,
        true_density=true_density_bundle.volume,
        output_dir=output_dir,
    )
    paths = save_reference_case_outputs(result, observed_gzz, metadata, output_dir, synthetic=True)
    _cleanup_gui_backend_intermediates(output_dir)
    return result, paths


def process_real_case(
    data_dir: Path,
    checkpoint_path: Path,
    model_capacity: str | None,
    input_mode: str | None,
    output_dir: Path,
    *,
    surface_calibration: bool = False,
) -> tuple[InferenceResult, list[Path]]:
    real_gzz_path = _real_data_gzz_path(data_dir)
    _run_gui_real_data_inference(
        output_dir,
        checkpoint_path=checkpoint_path,
        model_capacity=model_capacity,
        input_mode=input_mode,
        real_gz_path=None,
        real_gzz_path=real_gzz_path,
        surface_calibration=surface_calibration,
    )
    result, observed_gzz = _load_gui_real_data_result(
        case_name="vinton_field_data",
        output_dir=output_dir,
    )
    paths = save_reference_case_outputs(result, observed_gzz, REAL_FIGURES, output_dir, synthetic=False)
    _cleanup_gui_backend_intermediates(output_dir)
    return result, paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate GUI-style manuscript figures from best_model.pth and packaged data.")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "best_model.pth")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "figures" / "manuscript_outputs")
    parser.add_argument("--model-capacity", type=str, default=None, help="Optional override for checkpoint model capacity.")
    parser.add_argument("--input-mode", type=str, default=None, help="Optional override for checkpoint input mode.")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "synthetic_prism_model",
            "synthetic_two_prisms_model",
            "synthetic_irregular_model",
            "vinton_field_data",
        ],
        choices=[
            "synthetic_prism_model",
            "synthetic_two_prisms_model",
            "synthetic_irregular_model",
            "vinton_field_data",
        ],
        help="Cases to process.",
    )
    parser.add_argument(
        "--surface-calibration",
        action="store_true",
        help="Enable optional post-inference surface calibration.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = _ensure_dir(args.output_dir)
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Missing data directory: {args.data_dir}")

    _load_project_modules()
    manifest = {
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "output_dir": str(output_dir),
        "surface_calibration": bool(args.surface_calibration),
        "generated": [],
    }

    for case_name in args.cases:
        print(f"\nProcessing {case_name}...")
        case_output_dir = _ensure_dir(_case_output_dir(output_dir, case_name))
        if case_name == "vinton_field_data":
            _, paths = process_real_case(
                args.data_dir,
                args.checkpoint,
                args.model_capacity,
                args.input_mode,
                case_output_dir,
                surface_calibration=args.surface_calibration,
            )
        else:
            _, paths = process_synthetic_case(
                case_name,
                args.data_dir,
                args.checkpoint,
                args.model_capacity,
                args.input_mode,
                case_output_dir,
                surface_calibration=args.surface_calibration,
            )
        for path in paths:
            print(f"  saved {path}")
            manifest["generated"].append(str(path))

    manifest_path = output_dir / "figure_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as file_obj:
        json.dump(manifest, file_obj, ensure_ascii=False, indent=2)
    print(f"\nFigure manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
