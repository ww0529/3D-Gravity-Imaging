"""
使用训练好的模型进行推断和可视化
包含解析信号振幅计算和8个子图展示
"""

import argparse
import os
import pickle
import warnings
import torch
import numpy as np
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from matplotlib import cm, colors
from matplotlib.ticker import FuncFormatter, MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_fill_holes, gaussian_filter, uniform_filter, label, sobel
from scipy.fft import fftn, ifftn, fftfreq
from skimage import measure

from data_generator import GravityDataGenerator, build_model_input, get_input_channel_names, get_target_scale
from network import LocalizationAwareHybridUNet


PAPER_DIVERGING_CMAP = 'bwr'
PAPER_CLIP_PERCENTILE = 99.5
PAPER_LINEAR_LOWER_PERCENTILE = 0.5
PAPER_SYMLOG_LINTHRESH_RATIO = 0.04
SHOW_BOX_ANNOTATIONS = False


def _resolve_plot_cmap(cmap_name: str):
    """Use the shared red-white-blue palette for all scalar field displays."""
    _ = cmap_name
    return PAPER_DIVERGING_CMAP


def _resolve_input_plot_cmap(channel_name: str):
    """Use jet only for gzz input panels while keeping the shared palette elsewhere."""
    return 'jet' if str(channel_name) == 'gzz' else _resolve_plot_cmap('jet')


def _build_publication_norm(values, symmetric=True):
    """Use a robust paper-style scale so weak anomalies remain visible without hiding sign."""
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return colors.Normalize(vmin=0.0, vmax=1.0), 'neither'

    if symmetric:
        abs_values = np.abs(finite_values)
        peak = max(float(np.max(abs_values)), 1e-12)
        robust_limit = float(np.percentile(abs_values, PAPER_CLIP_PERCENTILE))
        limit = max(robust_limit, peak * 0.15, 1e-12)
        positive_abs = abs_values[abs_values > 0]
        if positive_abs.size:
            linthresh = max(
                float(np.percentile(positive_abs, 25)),
                limit * PAPER_SYMLOG_LINTHRESH_RATIO,
                1e-12,
            )
            linthresh = min(linthresh, limit * 0.2)
        else:
            linthresh = max(limit * PAPER_SYMLOG_LINTHRESH_RATIO, 1e-12)

        norm = colors.SymLogNorm(
            linthresh=linthresh,
            linscale=1.0,
            vmin=-limit,
            vmax=limit,
            base=10,
        )
        return norm, ('both' if limit < peak * 0.999 else 'neither')

    data_min = float(np.min(finite_values))
    data_max = float(np.max(finite_values))
    vmin = float(np.percentile(finite_values, PAPER_LINEAR_LOWER_PERCENTILE))
    vmax = float(np.percentile(finite_values, PAPER_CLIP_PERCENTILE))
    if abs(vmax - vmin) < 1e-12:
        vmin, vmax = data_min, data_max
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1e-12

    clipped = vmin > data_min + 1e-12 or vmax < data_max - 1e-12
    return colors.Normalize(vmin=vmin, vmax=vmax), ('both' if clipped else 'neither')


def _build_main_visualization_norm(values):
    """Use a linear symmetric scale for the main 8-panel visualization only."""
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return colors.Normalize(vmin=0.0, vmax=1.0), 'neither'

    abs_values = np.abs(finite_values)
    peak = max(float(np.max(abs_values)), 1e-12)
    robust_limit = float(np.percentile(abs_values, PAPER_CLIP_PERCENTILE))
    limit = max(robust_limit, peak * 0.15, 1e-12)
    extend = 'both' if limit < peak * 0.999 else 'neither'
    return colors.Normalize(vmin=-limit, vmax=limit), extend


def _build_body_response_projection_norm(values):
    """Center anomaly-body projection colorbars at zero so the quiet background stays white."""
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        limit = 1.0
        return colors.Normalize(vmin=-limit, vmax=limit), 'neither', np.linspace(-limit, limit, 21)

    abs_values = np.abs(finite_values)
    peak = max(float(np.max(abs_values)), 1e-12)
    robust_limit = float(np.percentile(abs_values, PAPER_CLIP_PERCENTILE))
    limit = max(robust_limit, peak * 0.15, 1e-12)
    extend = 'both' if limit < peak * 0.999 else 'neither'
    return colors.Normalize(vmin=-limit, vmax=limit), extend, np.linspace(-limit, limit, 21)


def _build_body_extent_projection_xz(source_response, envelope_level=None):
    """Project the estimated body extent along y and fill interior holes for XZ display."""
    source_response = np.asarray(source_response, dtype=np.float32)
    if source_response.ndim != 3:
        raise ValueError("source_response must be a 3D volume")

    if envelope_level is not None and np.isfinite(float(envelope_level)):
        threshold = float(envelope_level)
    else:
        positive_values = source_response[source_response > 0.0]
        threshold = float(np.quantile(positive_values, 0.75)) if positive_values.size else 0.0

    support_volume = source_response >= threshold
    support_projection_xz = np.max(support_volume, axis=1)
    if np.any(support_projection_xz):
        filled_projection_xz = binary_fill_holes(support_projection_xz)
        extent_projection_xz = gaussian_filter(filled_projection_xz.astype(np.float32), sigma=0.9)
        peak = float(np.max(extent_projection_xz))
        if peak > 1e-6:
            extent_projection_xz = extent_projection_xz / peak
        extent_projection_xz = extent_projection_xz * filled_projection_xz.astype(np.float32)
        return extent_projection_xz.astype(np.float32)

    return np.max(source_response, axis=1).astype(np.float32)


def _format_signed_colorbar_tick(value, _position=None):
    """Format colorbar ticks with an ASCII minus sign so negative labels remain legible."""
    numeric = float(value)
    if not np.isfinite(numeric):
        return ''
    magnitude = abs(numeric)
    if magnitude < 1e-12:
        return '0'

    sign = '-' if numeric < 0 else ''
    exponent = int(np.floor(np.log10(magnitude)))
    if magnitude >= 1e3 or magnitude < 1e-2:
        coefficient = magnitude / (10 ** exponent)
        if np.isclose(coefficient, 1.0, rtol=1e-6, atol=1e-12):
            return f'{sign}1e{exponent}'
        return f'{sign}{coefficient:.1f}e{exponent}'
    if magnitude >= 100:
        return f'{sign}{magnitude:.0f}'
    if magnitude >= 10:
        return f'{sign}{magnitude:.1f}'
    if magnitude >= 0.1:
        return f'{sign}{magnitude:.2f}'.rstrip('0').rstrip('.')
    return f'{sign}{magnitude:.3f}'.rstrip('0').rstrip('.')


def _nice_linear_colorbar_ticks(vmin, vmax, nbins=8):
    """Build clipped, evenly spaced, human-friendly ticks for linear colorbars."""
    raw_min = float(vmin)
    raw_max = float(vmax)
    if not np.isfinite(raw_min) or not np.isfinite(raw_max):
        return np.array([0.0], dtype=np.float64)
    if abs(raw_max - raw_min) < 1e-12:
        return np.array([raw_min], dtype=np.float64)

    if raw_min > raw_max:
        raw_min, raw_max = raw_max, raw_min

    locator = MaxNLocator(nbins=nbins, steps=[1, 2, 2.5, 5, 10], min_n_ticks=5)
    ticks = np.asarray(locator.tick_values(raw_min, raw_max), dtype=np.float64)
    tolerance = max(abs(raw_max - raw_min), 1.0) * 1e-9
    ticks = ticks[(ticks >= raw_min - tolerance) & (ticks <= raw_max + tolerance)]
    if ticks.size == 0:
        ticks = np.array([raw_min, raw_max], dtype=np.float64)

    if raw_min < 0.0 < raw_max and not np.any(np.isclose(ticks, 0.0, atol=tolerance, rtol=0.0)):
        ticks = np.sort(np.concatenate([ticks, np.array([0.0], dtype=np.float64)]))

    ticks[np.isclose(ticks, 0.0, atol=max(abs(raw_max - raw_min), 1.0) * 1e-12, rtol=0.0)] = 0.0
    return ticks


def _apply_publication_colorbar_ticks(colorbar):
    """Apply the shared publication colorbar tick style."""
    colorbar.ax.tick_params(which='both', direction='in')
    formatter = FuncFormatter(_format_signed_colorbar_tick)
    norm = getattr(colorbar, 'norm', None)
    if isinstance(norm, colors.Normalize) and not isinstance(norm, colors.SymLogNorm):
        vmin = float(getattr(norm, 'vmin', np.nan))
        vmax = float(getattr(norm, 'vmax', np.nan))
        if np.isfinite(vmin) and np.isfinite(vmax):
            colorbar.set_ticks(_nice_linear_colorbar_ticks(vmin, vmax))
        colorbar.formatter = formatter
        colorbar.ax.yaxis.set_major_formatter(formatter)
        colorbar.update_ticks()
    elif isinstance(norm, colors.SymLogNorm):
        colorbar.formatter = formatter
        colorbar.ax.yaxis.set_major_formatter(formatter)
        colorbar.update_ticks()
    return colorbar


def _publication_colorbar(*args, **kwargs):
    """Create a colorbar and apply the shared publication tick styling."""
    return _apply_publication_colorbar_ticks(plt.colorbar(*args, **kwargs))


def _fixed_spatial_ticks(coords, tick_step=500.0):
    """Build stable spatial ticks at a fixed metric spacing."""
    values = np.asarray(coords, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.array([], dtype=np.float64)

    step = max(abs(float(tick_step)), 1e-12)
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if upper < lower:
        lower, upper = upper, lower

    start = np.ceil(lower / step) * step
    end = np.floor(upper / step) * step
    ticks = np.arange(start, end + step * 0.5, step, dtype=np.float64)
    tolerance = step * 1e-9
    ticks = ticks[(ticks >= lower - tolerance) & (ticks <= upper + tolerance)]
    if ticks.size == 0:
        if np.isclose(lower, upper):
            ticks = np.array([lower], dtype=np.float64)
        else:
            ticks = np.array([lower, upper], dtype=np.float64)

    ticks[np.isclose(ticks, 0.0, atol=step * 1e-12, rtol=0.0)] = 0.0
    return ticks


def _apply_plan_view_spatial_ticks(ax, x_coords, y_coords, tick_step=500.0):
    """Keep XY plan-view axes on the same fixed 500 m tick spacing."""
    x_values = np.asarray(x_coords, dtype=np.float64).reshape(-1)
    y_values = np.asarray(y_coords, dtype=np.float64).reshape(-1)
    finite_x = x_values[np.isfinite(x_values)]
    finite_y = y_values[np.isfinite(y_values)]
    if finite_x.size:
        ax.set_xlim(float(np.min(finite_x)), float(np.max(finite_x)))
        ax.set_xticks(_fixed_spatial_ticks(finite_x, tick_step=tick_step))
    if finite_y.size:
        ax.set_ylim(float(np.min(finite_y)), float(np.max(finite_y)))
        ax.set_yticks(_fixed_spatial_ticks(finite_y, tick_step=tick_step))


def _apply_section_horizontal_spatial_ticks(ax, coords, tick_step=500.0):
    """Keep section-view horizontal axes on the same fixed 500 m tick spacing."""
    values = np.asarray(coords, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size:
        ax.set_xlim(float(np.min(finite)), float(np.max(finite)))
        ax.set_xticks(_fixed_spatial_ticks(finite, tick_step=tick_step))


def _configure_plot_language():
    """Use Chinese labels when a CJK font is available, otherwise fall back to English."""
    cjk_font_candidates = [
        'Noto Sans CJK SC',
        'Noto Sans CJK JP',
        'Noto Sans CJK TC',
        'WenQuanYi Zen Hei',
        'SimHei',
        'Microsoft YaHei',
        'Source Han Sans SC',
        'AR PL UMing CN',
        'PingFang SC',
    ]

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in cjk_font_candidates:
        if font_name in available_fonts:
            plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return 'zh'

    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return 'en'


def _safe_torch_load(checkpoint_path, device):
    """Prefer safe checkpoint loading on newer PyTorch versions."""
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            return torch.load(checkpoint_path, map_location=device)
    except pickle.UnpicklingError:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            return torch.load(checkpoint_path, map_location=device, weights_only=False)


def _load_model_from_checkpoint(checkpoint_path, device, model_capacity=None, input_mode=None):
    """Load a trained model and return both the model and checkpoint metadata."""
    checkpoint = _safe_torch_load(checkpoint_path, device)
    checkpoint_capacity = checkpoint.get('model_capacity')
    checkpoint_input_mode = checkpoint.get('input_mode')
    state_dict = checkpoint.get('model_state_dict', {})

    if checkpoint_capacity is not None and model_capacity is not None and model_capacity != checkpoint_capacity:
        warnings.warn(
            f"Checkpoint capacity is '{checkpoint_capacity}', ignoring requested '{model_capacity}'."
        )
    if checkpoint_input_mode is not None and input_mode is not None and input_mode != checkpoint_input_mode:
        warnings.warn(
            f"Checkpoint input_mode is '{checkpoint_input_mode}', ignoring requested '{input_mode}'."
        )
    if checkpoint_input_mode is None and input_mode not in (None, 'gz_amp'):
        warnings.warn(
            "Legacy checkpoint lacks input_mode metadata; falling back to gz_amp compatibility."
        )

    resolved_capacity = checkpoint_capacity or model_capacity or 'small'
    resolved_input_mode = checkpoint_input_mode or 'gz_amp'
    input_channels = checkpoint.get('input_channels', len(get_input_channel_names(resolved_input_mode)))
    resolved_use_multimodal_stem = checkpoint.get(
        'use_multimodal_stem',
        any(key.startswith('input_adapter.') for key in state_dict)
    )
    model = LocalizationAwareHybridUNet(
        capacity=resolved_capacity,
        input_channels=input_channels,
        input_mode=resolved_input_mode,
        use_multimodal_stem=resolved_use_multimodal_stem
    ).to(device)
    try:
        incompatible = model.load_state_dict(state_dict, strict=False)
        allowed_missing_prefixes = (
            'decoder_3d.body_head.',
            'decoder_3d.center_head.',
            'decoder_3d.positive_body_head.',
            'decoder_3d.negative_body_head.',
            'decoder_3d.axis_head.',
        )
        disallowed_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        if disallowed_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Checkpoint architecture mismatch. The current model uses the newer depth-aware "
                "2D-to-3D lifting in SkipLift2Dto3D/LocalizationBridge, so older checkpoints need "
                "to be retrained with the updated network before inference."
            )
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint architecture mismatch. The current model uses the newer depth-aware "
            "2D-to-3D lifting in SkipLift2Dto3D/LocalizationBridge, so older checkpoints need "
            "to be retrained with the updated network before inference."
        ) from exc

    aux_head_prefixes = (
        'decoder_3d.body_head.',
        'decoder_3d.center_head.',
        'decoder_3d.positive_body_head.',
        'decoder_3d.negative_body_head.',
        'decoder_3d.axis_head.',
    )
    has_aux_heads = (
        not incompatible.missing_keys or
        any(key.startswith(aux_head_prefixes) for key in state_dict)
    )
    return (
        model,
        checkpoint,
        resolved_capacity,
        resolved_input_mode,
        resolved_use_multimodal_stem,
        has_aux_heads
    )


def _get_default_checkpoint_path(model_capacity, input_mode):
    candidates = [
        Path(f'./checkpoints_localization_{model_capacity}_{input_mode}_deep_balanced/best_model.pth'),
        Path(f'./checkpoints_localization_{model_capacity}_{input_mode}/best_model.pth'),
        Path(f'./checkpoints_localization_{model_capacity}/best_model.pth'),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def _select_aux_channel(input_mode):
    aux_channels = [name for name in get_input_channel_names(input_mode) if name != 'gz']
    return aux_channels[0] if aux_channels else 'gz'


def _select_primary_channel(input_mode):
    return get_input_channel_names(input_mode)[0]


def _get_aux_channel_labels(language, channel_name):
    labels = {
        'zh': {
            'amp': (r'解析信号振幅: $|A| = \sqrt{V_{zx}^2 + V_{zy}^2 + V_{zz}^2}$', '解析信号振幅等值线', '振幅 (mGal/m)'),
            'gzz': ('地表 Gzz 输入', '地表 Gzz 等值线', 'Gzz'),
            'gz': ('地表 Gz 输入', '地表 Gz 等值线', 'Gz'),
        },
        'en': {
            'amp': (r'Analytic Signal Amplitude: $|A| = \sqrt{V_{zx}^2 + V_{zy}^2 + V_{zz}^2}$', 'Analytic Signal Contours', 'Amplitude (mGal/m)'),
            'gzz': ('Surface Gzz Input', 'Surface Gzz Contours', 'Gzz'),
            'gz': ('Surface Gz Input', 'Surface Gz Contours', 'Gz'),
        },
    }
    return labels[language][channel_name]


def _resolve_input_normalization_mode(checkpoint):
    if 'input_normalization_mode' in checkpoint:
        return checkpoint['input_normalization_mode']
    if 'input_mode' in checkpoint:
        return 'gz_ref_aux_separate'
    return 'legacy_shared'


def _build_inference_input(surface_channels, input_mode, checkpoint, label_3d=None):
    normalization_mode = _resolve_input_normalization_mode(checkpoint)
    channel_names = get_input_channel_names(input_mode)

    if normalization_mode == 'legacy_shared':
        scale_candidates = [float(np.max(np.abs(surface_channels[channel_name]))) for channel_name in channel_names]
        if label_3d is not None:
            scale_candidates.append(float(np.max(np.abs(label_3d))))
        shared_scale = max(max(scale_candidates), 1e-12)
        input_array = np.stack(
            [(surface_channels[channel_name] / shared_scale).astype(np.float32) for channel_name in channel_names],
            axis=0
        )
        return input_array, shared_scale, normalization_mode

    target_scale = get_target_scale(surface_channels, label_3d)
    input_array, _, target_scale = build_model_input(
        surface_channels,
        input_mode=input_mode,
        target_scale=target_scale
    )
    return input_array, target_scale, normalization_mode


def _load_real_surface_grid(txt_path):
    """Load a regular x,y,value text grid from CSV-like text."""
    arr = np.loadtxt(txt_path, delimiter=',')
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


def _resample_grid_to_model_size(xs, ys, grid, target_size=64):
    """Linearly resample a regular grid to the model input size."""
    interpolator = RegularGridInterpolator((xs, ys), grid, method='linear', bounds_error=False, fill_value=None)
    x_new = np.linspace(xs.min(), xs.max(), target_size)
    y_new = np.linspace(ys.min(), ys.max(), target_size)
    x_mesh, y_mesh = np.meshgrid(x_new, y_new, indexing='ij')
    samples = np.stack([x_mesh.ravel(), y_mesh.ravel()], axis=1)
    grid_new = interpolator(samples).reshape(target_size, target_size)
    return x_new, y_new, grid_new


def _compute_surface_analytic_amplitude(field_2d, dx, dy):
    """Approximate the surface analytic signal amplitude from a 2D observed field."""
    nx, ny = field_2d.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
    radial_k = np.sqrt(kx_grid ** 2 + ky_grid ** 2)

    field_fft = np.fft.fft2(field_2d)
    dfdx = np.fft.ifft2(1j * kx_grid * field_fft).real
    dfdy = np.fft.ifft2(1j * ky_grid * field_fft).real
    dfdz = np.fft.ifft2(radial_k * field_fft).real

    return np.sqrt(dfdx ** 2 + dfdy ** 2 + dfdz ** 2)


def _compute_surface_vertical_gradient_from_volume(field_3d, z_coords):
    """Approximate surface Gzz from the top layers of a 3D gz volume."""
    volume = np.asarray(field_3d, dtype=np.float32)
    if volume.ndim != 3 or volume.shape[2] < 2:
        return np.zeros(volume.shape[:2], dtype=np.float32)

    z_coords = np.asarray(z_coords, dtype=np.float64).reshape(-1)
    dz = float(z_coords[1] - z_coords[0]) if z_coords.size > 1 else 1.0
    dz = max(abs(dz), 1e-12)

    surface = volume[:, :, 0]
    next_1 = volume[:, :, 1]
    if volume.shape[2] >= 3:
        next_2 = volume[:, :, 2]
        gradient = (-3.0 * surface + 4.0 * next_1 - next_2) / (2.0 * dz)
    else:
        gradient = (next_1 - surface) / dz
    return np.asarray(gradient, dtype=np.float32)


def _surface_fit_metrics(pred_surface, obs_surface):
    """Return lightweight fit diagnostics for optional post-inference correction."""
    pred = np.asarray(pred_surface, dtype=np.float64).reshape(-1)
    obs = np.asarray(obs_surface, dtype=np.float64).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(obs)
    if not np.any(mask):
        return {'corr': np.nan, 'rmse': np.nan, 'mae': np.nan}

    pred = pred[mask]
    obs = obs[mask]
    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    mae = float(np.mean(np.abs(pred - obs)))
    corr = float(np.corrcoef(pred, obs)[0, 1]) if pred.size > 1 else np.nan
    return {'corr': corr, 'rmse': rmse, 'mae': mae}


def _apply_post_inference_surface_calibration(
    field_3d,
    z_coords,
    observed_gz=None,
    observed_gzz=None,
    gz_weight=5.0,
    gzz_weight=5.0,
    prior_weights=(1.0, 1.0, 1.0),
):
    """
    Adjust only the top few depth layers after inference so the surface response
    better matches observed gz/gzz without retraining the network.
    """
    volume = np.asarray(field_3d, dtype=np.float32)
    corrected = volume.copy()

    if volume.ndim != 3 or volume.shape[2] < 2:
        return corrected, {
            'applied': False,
            'reason': 'insufficient_depth_layers',
        }

    z_coords = np.asarray(z_coords, dtype=np.float64).reshape(-1)
    dz = float(z_coords[1] - z_coords[0]) if z_coords.size > 1 else 1.0
    dz = max(abs(dz), 1e-12)

    n_layers = 3 if volume.shape[2] >= 3 else 2
    prior_weights = np.asarray(prior_weights, dtype=np.float64).reshape(-1)
    if prior_weights.size < n_layers:
        prior_weights = np.pad(prior_weights, (0, n_layers - prior_weights.size), mode='edge')
    prior_weights = np.maximum(prior_weights[:n_layers], 1e-8)

    if n_layers == 3:
        deriv = np.array([-3.0, 4.0, -1.0], dtype=np.float64) / (2.0 * dz)
    else:
        deriv = np.array([-1.0, 1.0], dtype=np.float64) / dz

    observed_gz_arr = None if observed_gz is None else np.asarray(observed_gz, dtype=np.float64)
    observed_gzz_arr = None if observed_gzz is None else np.asarray(observed_gzz, dtype=np.float64)
    effective_gz_weight = float(gz_weight) if observed_gz_arr is not None else 0.0
    effective_gzz_weight = float(gzz_weight) if observed_gzz_arr is not None else 0.0

    if effective_gz_weight <= 0.0 and effective_gzz_weight <= 0.0:
        return corrected, {
            'applied': False,
            'reason': 'no_surface_targets',
        }

    prior_diag = np.diag(prior_weights)
    surface_selector = np.zeros((n_layers,), dtype=np.float64)
    surface_selector[0] = 1.0
    system = prior_diag.copy()
    if effective_gz_weight > 0.0:
        system += effective_gz_weight * np.outer(surface_selector, surface_selector)
    if effective_gzz_weight > 0.0:
        system += effective_gzz_weight * np.outer(deriv, deriv)
    system_inv = np.linalg.inv(system)

    top_layers = volume[:, :, :n_layers].astype(np.float64)
    rhs = top_layers * prior_weights.reshape(1, 1, -1)
    if effective_gz_weight > 0.0:
        rhs += effective_gz_weight * observed_gz_arr[..., None] * surface_selector.reshape(1, 1, -1)
    if effective_gzz_weight > 0.0:
        rhs += effective_gzz_weight * observed_gzz_arr[..., None] * deriv.reshape(1, 1, -1)

    corrected[:, :, :n_layers] = np.matmul(rhs, system_inv.T).astype(np.float32)

    diagnostics = {
        'applied': True,
        'num_layers': int(n_layers),
        'gz_weight': float(effective_gz_weight),
        'gzz_weight': float(effective_gzz_weight),
        'prior_weights': prior_weights.astype(np.float64),
    }

    if observed_gz_arr is not None:
        diagnostics['surface_gz_before'] = _surface_fit_metrics(volume[:, :, 0], observed_gz_arr)
        diagnostics['surface_gz_after'] = _surface_fit_metrics(corrected[:, :, 0], observed_gz_arr)
    if observed_gzz_arr is not None:
        diagnostics['surface_gzz_before'] = _surface_fit_metrics(
            _compute_surface_vertical_gradient_from_volume(volume, z_coords),
            observed_gzz_arr,
        )
        diagnostics['surface_gzz_after'] = _surface_fit_metrics(
            _compute_surface_vertical_gradient_from_volume(corrected, z_coords),
            observed_gzz_arr,
        )

    return corrected, diagnostics


def _convert_real_channel_to_training_units(channel_name, grid):
    """Map common SI-unit real grids into the units used by the synthetic generator."""
    grid = np.asarray(grid, dtype=np.float32)
    max_abs = float(np.max(np.abs(grid))) if grid.size else 0.0
    applied_scale = 1.0
    assumed_unit = 'training_native'

    if channel_name == 'gzz':
        # Synthetic training data uses gzz magnitudes around 1e-3 to 1e-2,
        # which correspond to about mGal/m. Field data often arrives in SI (s^-2).
        if 0.0 < max_abs < 1e-5:
            applied_scale = 1e5
            assumed_unit = 'SI_to_mGal_per_m'
    elif channel_name == 'gz':
        # Synthetic training data uses gz in mGal-like magnitudes. Real data may
        # arrive in SI acceleration units (m/s^2), where 1 mGal = 1e-5 m/s^2.
        if 0.0 < max_abs < 1e-2:
            applied_scale = 1e5
            assumed_unit = 'SI_to_mGal'

    return (grid * applied_scale).astype(np.float32), {
        'channel': channel_name,
        'input_abs_max': max_abs,
        'applied_scale': applied_scale,
        'assumed_unit': assumed_unit,
    }


def _rotation_matrix_from_degrees(yaw_degrees=0.0, pitch_degrees=0.0, roll_degrees=0.0):
    """Build a 3D rotation matrix from yaw/pitch/roll angles in degrees."""
    yaw = np.deg2rad(float(yaw_degrees))
    pitch = np.deg2rad(float(pitch_degrees))
    roll = np.deg2rad(float(roll_degrees))

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    return rz @ ry @ rx


def _generate_center_box_case(
    generator,
    noise_level=0.0,
    add_regional=False,
    center=None,
    half_size_xyz=(450.0, 250.0, 180.0),
    density=900.0,
    rotation_degrees=0.0,
    return_density_model=False,
):
    """Build a deterministic centered box sample without modifying the training data generator."""
    if center is None:
        center = (
            float(generator.x[generator.nx // 2]),
            float(generator.y[generator.ny // 2]),
            float(generator.z[generator.nz // 2]),
        )

    rotation = _rotation_matrix_from_degrees(yaw_degrees=rotation_degrees)

    rho_model = np.zeros((generator.nx, generator.ny, generator.nz), dtype=np.float32)
    mask = generator._build_box_mask(center, half_size_xyz, rotation)
    rho_model[mask] = float(density)

    anomaly_info = [{
        'type': 'center_box',
        'center': tuple(float(v) for v in center),
        'size': float(max(half_size_xyz)),
        'size_xyz': tuple(float(v) for v in half_size_xyz),
        'density': float(density),
    }]
    return _simulate_density_model_case(
        generator,
        rho_model,
        anomaly_info,
        noise_level=noise_level,
        add_regional=add_regional,
        return_density_model=return_density_model,
        top_padding_cells=max(generator.nz // 2, 16),
    )


def _generate_deep_box_case(generator, noise_level=0.0, add_regional=False, return_density_model=False):
    """Generate a deterministic deeper compact body for depth-recovery testing."""
    center = (
        float(generator.x[generator.nx // 2]),
        float(generator.y[generator.ny // 2]),
        2200.0,
    )
    half_size_xyz = (450.0, 320.0, 280.0)
    density = 1250.0
    rotation = _rotation_matrix_from_degrees(yaw_degrees=18.0, pitch_degrees=-10.0, roll_degrees=6.0)

    rho_model = np.zeros((generator.nx, generator.ny, generator.nz), dtype=np.float32)
    mask = generator._build_box_mask(center, half_size_xyz, rotation)
    rho_model[mask] = density

    anomaly_info = [{
        'type': 'deep_box',
        'center': tuple(float(v) for v in center),
        'size': float(max(half_size_xyz)),
        'size_xyz': tuple(float(v) for v in half_size_xyz),
        'density': float(density),
        'bbox': _compute_mask_bounds(mask, generator.x, generator.y, generator.z),
    }]
    return _simulate_density_model_case(
        generator,
        rho_model,
        anomaly_info,
        noise_level=noise_level,
        add_regional=add_regional,
        return_density_model=return_density_model,
    )


def _simulate_density_model_case(
    generator,
    rho_model,
    anomaly_info,
    noise_level=0.0,
    add_regional=False,
    return_density_model=False,
    top_padding_cells=0,
):
    """Forward model a deterministic density volume using the same synthetic-physics pipeline."""
    rho_model = np.asarray(rho_model, dtype=np.float32)
    top_padding_cells = max(int(top_padding_cells), 0)

    kx = 2 * np.pi * fftfreq(generator.nx, generator.dx)
    ky = 2 * np.pi * fftfreq(generator.ny, generator.dy)
    z_slice = slice(0, generator.nz)

    if top_padding_cells > 0:
        padded_nz = generator.nz + top_padding_cells
        rho_model_for_fft = np.zeros((generator.nx, generator.ny, padded_nz), dtype=np.float32)
        # Add empty air cells above the physical surface before cropping back.
        rho_model_for_fft[:, :, top_padding_cells:] = rho_model
        z_slice = slice(top_padding_cells, top_padding_cells + generator.nz)

        kz = 2 * np.pi * fftfreq(padded_nz, generator.dz)
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        K2 = KX ** 2 + KY ** 2 + KZ ** 2
        K2[0, 0, 0] = 1e-10
        H = (1j * 4 * np.pi * generator.G * KZ) / K2
        H[0, 0, 0] = 0
    else:
        rho_model_for_fft = rho_model
        kz = 2 * np.pi * fftfreq(generator.nz, generator.dz)
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        H = generator.H

    F_rho = fftn(rho_model_for_fft)
    F_gz = H * F_rho
    gz_3d_anomaly_full = np.real(ifftn(F_gz)) * 1e5

    F_Vzx = 1j * KX * F_gz
    F_Vzy = 1j * KY * F_gz
    F_Vzz = 1j * KZ * F_gz

    Vzx_3d_full = np.real(ifftn(F_Vzx)) * 1e5
    Vzy_3d_full = np.real(ifftn(F_Vzy)) * 1e5
    Vzz_3d_full = np.real(ifftn(F_Vzz)) * 1e5
    analytic_amp_full = np.sqrt(Vzx_3d_full ** 2 + Vzy_3d_full ** 2 + Vzz_3d_full ** 2)

    gz_3d_anomaly = gz_3d_anomaly_full[:, :, z_slice]
    Vzz_3d = Vzz_3d_full[:, :, z_slice]
    analytic_amp = analytic_amp_full[:, :, z_slice]

    gz_3d_observed = gz_3d_anomaly.copy()
    if add_regional:
        gz_3d_observed = generator._add_regional_field(gz_3d_observed)

    gz_surface = generator._add_mixed_noise(gz_3d_observed[:, :, 0].copy(), noise_level)
    gzz_surface = generator._add_mixed_noise(Vzz_3d[:, :, 0].copy(), noise_level)
    amp_surface = generator._add_mixed_noise(analytic_amp[:, :, 0].copy(), noise_level * 0.7)

    surface_channels = {
        'gz': gz_surface.astype(np.float32),
        'gzz': gzz_surface.astype(np.float32),
        'amp': amp_surface.astype(np.float32),
    }
    if return_density_model:
        return surface_channels, gz_3d_anomaly.astype(np.float32), anomaly_info, rho_model.astype(np.float32)
    return surface_channels, gz_3d_anomaly.astype(np.float32), anomaly_info


def _generate_dual_boxes_case(generator, noise_level=0.0, add_regional=False, return_density_model=False):
    """Generate two axis-aligned boxes with opposite density contrasts at different depths."""
    cx = float(generator.x[generator.nx // 2])
    cy = float(generator.y[generator.ny // 2])

    box_specs = [
        {
            'type': 'positive_box',
            'center': (cx - 300.0, cy, 750.0),
            'size_xyz': (320.0, 220.0, 170.0),
            'density': 900.0,
        },
        {
            'type': 'negative_box',
            'center': (cx + 350.0, cy, 1450.0),
            'size_xyz': (380.0, 240.0, 230.0),
            'density': -850.0,
        },
    ]

    rho_model = np.zeros((generator.nx, generator.ny, generator.nz), dtype=np.float32)
    for info in box_specs:
        rotation = np.eye(3, dtype=np.float32)
        mask = generator._build_box_mask(info['center'], info['size_xyz'], rotation)
        rho_model[mask] = float(info['density'])
        info['size'] = float(max(info['size_xyz']))

    anomaly_info = [{
        'type': info['type'],
        'center': tuple(float(v) for v in info['center']),
        'size': float(info['size']),
        'size_xyz': tuple(float(v) for v in info['size_xyz']),
        'density': float(info['density']),
    } for info in box_specs]

    return _simulate_density_model_case(
        generator,
        rho_model,
        anomaly_info,
        noise_level=noise_level,
        add_regional=add_regional,
        return_density_model=return_density_model,
    )


def _generate_single_tilted_bar_case(generator, noise_level=0.0, add_regional=False, return_density_model=False):
    """Generate one long tilted cylindrical body to test whether XZ slices preserve oblique structure."""
    center = (
        float(generator.x[generator.nx // 2] - 80.0),
        float(generator.y[generator.ny // 2]),
        1400.0,
    )
    radius_xy = (90.0, 135.0)
    half_height = 820.0
    density = 1080.0
    rotation_degrees = (0.0, 34.0, 0.0)
    rotation = _rotation_matrix_from_degrees(*rotation_degrees)

    rho_model = np.zeros((generator.nx, generator.ny, generator.nz), dtype=np.float32)
    mask = generator._build_cylinder_mask(center, radius_xy, half_height, rotation)
    mask = gaussian_filter(mask.astype(np.float32), sigma=0.75) >= 0.42
    rho_model[mask] = float(density)

    occupied = np.argwhere(mask)
    x_center = float(np.mean(generator.x[occupied[:, 0]]))
    y_center = float(np.mean(generator.y[occupied[:, 1]]))
    z_center = float(np.mean(generator.z[occupied[:, 2]]))
    anomaly_info = [{
        'type': 'single_tilted_bar',
        'shape': 'tilted_bar',
        'center': (x_center, y_center, z_center),
        'size': float(max(radius_xy[0], radius_xy[1], half_height)),
        'size_xyz': (float(radius_xy[0]), float(radius_xy[1]), float(half_height)),
        'density': float(density),
        'bbox': _compute_mask_bounds(mask, generator.x, generator.y, generator.z),
        'rotation_degrees': tuple(float(v) for v in rotation_degrees),
    }]
    return _simulate_density_model_case(
        generator,
        rho_model,
        anomaly_info,
        noise_level=noise_level,
        add_regional=add_regional,
        return_density_model=return_density_model,
    )


def _generate_wide_dual_tilted_bars_case(generator, noise_level=0.0, add_regional=False, return_density_model=False):
    """Generate two opposite-sign tilted bars that stay in similar y but are farther separated in x."""
    cx = float(generator.x[generator.nx // 2])
    cy = float(generator.y[generator.ny // 2])
    branch_specs = [
        {
            'type': 'positive_bar',
            'shape': 'cylinder',
            'center': (cx - 620.0, cy, 1100.0),
            'radius_xy': (92.0, 138.0),
            'half_height': 780.0,
            'rotation_degrees': (0.0, 33.0, 0.0),
            'density': 1020.0,
        },
        {
            'type': 'negative_bar',
            'shape': 'cylinder',
            'center': (cx + 620.0, cy, 1750.0),
            'radius_xy': (96.0, 138.0),
            'half_height': 740.0,
            'rotation_degrees': (0.0, -29.0, 0.0),
            'density': -980.0,
        },
    ]

    rho_model = np.zeros((generator.nx, generator.ny, generator.nz), dtype=np.float32)
    anomaly_info = []

    for spec in branch_specs:
        rotation = _rotation_matrix_from_degrees(*spec['rotation_degrees'])
        if spec['shape'] == 'cylinder':
            component_mask = generator._build_cylinder_mask(spec['center'], spec['radius_xy'], spec['half_height'], rotation)
        else:
            raise ValueError(f"Unsupported component shape '{spec['shape']}'")

        component_mask = gaussian_filter(component_mask.astype(np.float32), sigma=0.75) >= 0.42
        rho_model[component_mask] += float(spec['density'])

        occupied = np.argwhere(component_mask)
        bbox = _compute_mask_bounds(component_mask, generator.x, generator.y, generator.z)
        x_center = float(np.mean(generator.x[occupied[:, 0]]))
        y_center = float(np.mean(generator.y[occupied[:, 1]]))
        z_center = float(np.mean(generator.z[occupied[:, 2]]))
        half_size_xyz = (
            float(spec['radius_xy'][0]),
            float(spec['radius_xy'][1]),
            float(spec['half_height']),
        )
        anomaly_info.append({
            'type': spec['type'],
            'shape': 'tilted_bar',
            'center': (x_center, y_center, z_center),
            'size': float(max(half_size_xyz)),
            'size_xyz': tuple(float(v) for v in half_size_xyz),
            'density': float(spec['density']),
            'bbox': bbox,
            'rotation_degrees': tuple(float(v) for v in spec['rotation_degrees']),
        })

    return _simulate_density_model_case(
        generator,
        rho_model,
        anomaly_info,
        noise_level=noise_level,
        add_regional=add_regional,
        return_density_model=return_density_model,
    )


def _generate_complex_geology_case(generator, noise_level=0.0, add_regional=False, return_density_model=False):
    """Generate two tilted elongated branches with opposite density contrast."""
    cx = float(generator.x[generator.nx // 2])
    cy = float(generator.y[generator.ny // 2])
    branch_specs = [
        {
            'type': 'positive_branch',
            'shape': 'cylinder',
            'center': (cx - 360.0, cy, 1200.0),
            'radius_xy': (85.0, 130.0),
            'half_height': 700.0,
            # Keep both bars near the same y slice, but separate them in x and use opposite
            # pitch so the XZ projection looks like two long oblique branches approaching/crossing.
            'rotation_degrees': (0.0, 34.0, 0.0),
            'density': 980.0,
        },
        {
            'type': 'negative_branch',
            'shape': 'cylinder',
            'center': (cx + 120.0, cy, 1450.0),
            'radius_xy': (85.0, 130.0),
            'half_height': 620.0,
            'rotation_degrees': (0.0, -28.0, 0.0),
            'density': -920.0,
        },
    ]

    rho_model = np.zeros((generator.nx, generator.ny, generator.nz), dtype=np.float32)
    anomaly_info = []

    for spec in branch_specs:
        rotation = _rotation_matrix_from_degrees(*spec['rotation_degrees'])
        if spec['shape'] == 'cylinder':
            component_mask = generator._build_cylinder_mask(spec['center'], spec['radius_xy'], spec['half_height'], rotation)
        else:
            raise ValueError(f"Unsupported component shape '{spec['shape']}'")

        # Slight smoothing keeps the synthetic geometry geological-looking without breaking
        # the long, bar-like character of the two branches.
        component_mask = gaussian_filter(component_mask.astype(np.float32), sigma=0.75) >= 0.42
        rho_model[component_mask] += float(spec['density'])

        occupied = np.argwhere(component_mask)
        bbox = _compute_mask_bounds(component_mask, generator.x, generator.y, generator.z)
        x_center = float(np.mean(generator.x[occupied[:, 0]]))
        y_center = float(np.mean(generator.y[occupied[:, 1]]))
        z_center = float(np.mean(generator.z[occupied[:, 2]]))
        half_size_xyz = (
            float(spec['radius_xy'][0]),
            float(spec['radius_xy'][1]),
            float(spec['half_height']),
        )
        anomaly_info.append({
            'type': spec['type'],
            'shape': 'tilted_branch',
            'center': (x_center, y_center, z_center),
            'size': float(max(half_size_xyz)),
            'size_xyz': tuple(float(v) for v in half_size_xyz),
            'density': float(spec['density']),
            'bbox': bbox,
            'rotation_degrees': tuple(float(v) for v in spec['rotation_degrees']),
        })

    return _simulate_density_model_case(
        generator,
        rho_model,
        anomaly_info,
        noise_level=noise_level,
        add_regional=add_regional,
        return_density_model=return_density_model,
    )


def _build_boundary_taper(shape, border=4):
    """Build a smooth cosine taper so edge artifacts do not dominate derivatives."""
    border = max(0, int(border))
    if border == 0:
        return np.ones(shape, dtype=np.float32)

    def _axis_window(length):
        if length <= 2:
            return np.ones(length, dtype=np.float32)

        current_border = min(border, max(length // 2 - 1, 1))
        if current_border <= 0:
            return np.ones(length, dtype=np.float32)

        window = np.ones(length, dtype=np.float32)
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, current_border, dtype=np.float32))
        window[:current_border] = ramp
        window[-current_border:] = ramp[::-1]
        return window

    wx = _axis_window(shape[0])
    wy = _axis_window(shape[1])
    wz = _axis_window(shape[2])
    return wx[:, None, None] * wy[None, :, None] * wz[None, None, :]


def _zero_boundary_shell(volume, shell_width=4, fill_value=0.0):
    """Zero out a thin outer shell so marching-cubes ignores box-edge artifacts."""
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


def _interior_volume_view(volume, edge_ignore=4):
    edge_ignore = max(0, int(edge_ignore))
    if edge_ignore == 0 or min(volume.shape) <= 2 * edge_ignore:
        return volume
    return volume[edge_ignore:-edge_ignore, edge_ignore:-edge_ignore, edge_ignore:-edge_ignore]


def _compute_volume_derivative_fields(field_3d, dx, dy, dz, edge_taper_width=4, smooth_sigma=0.8):
    """Compute derivative-based 3D attribute volumes with boundary suppression."""
    centered_field = field_3d - np.median(field_3d)
    boundary_taper = _build_boundary_taper(field_3d.shape, border=edge_taper_width)
    tapered_field = centered_field * boundary_taper

    dfdx, dfdy, dfdz = np.gradient(tapered_field, dx, dy, dz, edge_order=1)
    thdr = np.sqrt(dfdx ** 2 + dfdy ** 2)
    asa3d = np.sqrt(dfdx ** 2 + dfdy ** 2 + dfdz ** 2)

    if smooth_sigma and smooth_sigma > 0:
        thdr = gaussian_filter(thdr, sigma=smooth_sigma)
        asa3d = gaussian_filter(asa3d, sigma=smooth_sigma)

    dfdx = _zero_boundary_shell(dfdx, shell_width=edge_taper_width, fill_value=0.0)
    dfdy = _zero_boundary_shell(dfdy, shell_width=edge_taper_width, fill_value=0.0)
    dfdz = _zero_boundary_shell(dfdz, shell_width=edge_taper_width, fill_value=0.0)
    thdr = _zero_boundary_shell(thdr, shell_width=edge_taper_width, fill_value=0.0)
    asa3d = _zero_boundary_shell(asa3d, shell_width=edge_taper_width, fill_value=0.0)

    return {
        'dfdx': dfdx,
        'dfdy': dfdy,
        'dfdz': dfdz,
        'thdr': thdr,
        'asa3d': asa3d,
        'tapered_prediction': tapered_field,
        'edge_taper_width': edge_taper_width,
    }


def _build_interior_weight(shape, edge_ignore=4):
    """Return a binary interior mask used to downweight box-edge artifacts."""
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


def _build_interior_weight_2d(shape, edge_ignore=4):
    """2D counterpart used for robust surface-map priors."""
    edge_ignore = max(0, int(edge_ignore))
    weight = np.ones(shape, dtype=np.float32)
    if edge_ignore == 0 or min(shape) <= 2 * edge_ignore:
        return weight

    weight[:edge_ignore, :] = 0.0
    weight[-edge_ignore:, :] = 0.0
    weight[:, :edge_ignore] = 0.0
    weight[:, -edge_ignore:] = 0.0
    return weight


def _normalize_source_response(response, edge_ignore=4):
    """Robustly normalize a source-like response volume to [0, 1]."""
    response = np.abs(np.asarray(response, dtype=np.float32))
    interior_weight = _build_interior_weight(response.shape, edge_ignore=edge_ignore)
    interior_values = response[interior_weight > 0.5]
    if interior_values.size == 0:
        interior_values = response.reshape(-1)

    low_q = float(np.quantile(interior_values, 0.50))
    high_q = float(np.quantile(interior_values, 0.995))
    robust_scale = max(high_q - low_q, 1e-6)

    normalized = np.clip((response - low_q) / robust_scale, 0.0, 1.0)
    attenuated_weight = 0.25 + 0.75 * interior_weight
    return (normalized * attenuated_weight).astype(np.float32)


def _normalize_surface_response(response, edge_ignore=4):
    """Robustly normalize a 2D response map to [0, 1]."""
    response = np.abs(np.asarray(response, dtype=np.float32))
    interior_weight = _build_interior_weight_2d(response.shape, edge_ignore=edge_ignore)
    interior_values = response[interior_weight > 0.5]
    if interior_values.size == 0:
        interior_values = response.reshape(-1)

    low_q = float(np.quantile(interior_values, 0.50))
    high_q = float(np.quantile(interior_values, 0.995))
    robust_scale = max(high_q - low_q, 1e-6)

    normalized = np.clip((response - low_q) / robust_scale, 0.0, 1.0)
    attenuated_weight = 0.25 + 0.75 * interior_weight
    return (normalized * attenuated_weight).astype(np.float32)


def _compute_source_localization_response(field_3d, dx, dy, dz, edge_ignore=4, smooth_sigma=0.8):
    """Estimate anomaly-body support from a predicted 3D gravity field."""
    centered_field = np.asarray(field_3d, dtype=np.float32) - np.float32(np.median(field_3d))
    tapered_field = centered_field * _build_boundary_taper(centered_field.shape, border=edge_ignore)

    grad_x = sobel(tapered_field, axis=0, mode='nearest') / max(float(dx), 1e-6)
    grad_y = sobel(tapered_field, axis=1, mode='nearest') / max(float(dy), 1e-6)
    grad_z = sobel(tapered_field, axis=2, mode='nearest') / max(float(dz), 1e-6)
    gradient_response = np.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2)
    envelope_response = uniform_filter(np.abs(tapered_field), size=5, mode='nearest')

    if smooth_sigma and smooth_sigma > 0:
        gradient_response = gaussian_filter(gradient_response, sigma=smooth_sigma)
        envelope_response = gaussian_filter(envelope_response, sigma=smooth_sigma)

    gradient_response = _normalize_source_response(gradient_response, edge_ignore=edge_ignore)
    envelope_response = _normalize_source_response(envelope_response, edge_ignore=edge_ignore)
    source_response = 0.78 * gradient_response + 0.22 * envelope_response
    source_response = _normalize_source_response(source_response, edge_ignore=edge_ignore)

    return {
        'source_response': source_response,
        'gradient_response': gradient_response,
        'envelope_response': envelope_response,
        'tapered_prediction': tapered_field,
        'edge_ignore': max(0, int(edge_ignore)),
    }


def _compute_surface_focus_prior(surface_map, x_coords, y_coords, edge_ignore=4):
    """Estimate one or more observed anomaly footprints and build a multi-peak XY prior."""
    surface_map = np.asarray(surface_map, dtype=np.float32)
    detrended = surface_map - gaussian_filter(surface_map, sigma=4.2)
    anomaly_response = _normalize_surface_response(gaussian_filter(detrended, sigma=1.1), edge_ignore=edge_ignore)
    peak_index = np.unravel_index(np.argmax(anomaly_response), anomaly_response.shape)
    peak_x = float(x_coords[peak_index[0]])
    peak_y = float(y_coords[peak_index[1]])

    interior_values = anomaly_response[_build_interior_weight_2d(anomaly_response.shape, edge_ignore=edge_ignore) > 0.5]
    if interior_values.size == 0:
        interior_values = anomaly_response.reshape(-1)
    threshold = max(float(np.percentile(interior_values, 92.0)), 0.55 * float(anomaly_response.max()))
    support_mask = anomaly_response >= threshold
    support_components = _extract_connected_components(
        support_mask,
        axis_coords=(x_coords, y_coords),
        axis_names=('x', 'y'),
        value_volume=anomaly_response,
        min_size=max(6, int(edge_ignore) * 2),
        max_components=6,
    )

    x_mesh, y_mesh = np.meshgrid(x_coords, y_coords, indexing='ij')
    prior_fields: list[np.ndarray] = []
    prior_bboxes = [component['bbox'] for component in support_components]
    prior_centers = [component['center'] for component in support_components]

    if support_components:
        dominant_center = tuple(float(v) for v in support_components[0]['center'])
        dominant_bbox = support_components[0]['bbox']
        for component in support_components:
            bbox = component['bbox']
            center_x, center_y = component['center']
            sigma_x = max(0.8 * max(bbox['x_max'] - bbox['x_min'], 1.0), 650.0)
            sigma_y = max(0.8 * max(bbox['y_max'] - bbox['y_min'], 1.0), 650.0)
            prior_fields.append(
                np.exp(
                    -0.5 * (((x_mesh - center_x) / sigma_x) ** 2 + ((y_mesh - center_y) / sigma_y) ** 2)
                ).astype(np.float32)
            )
    else:
        dominant_center = (peak_x, peak_y)
        dominant_bbox = None
        sigma_x = max(650.0, float(np.abs(x_coords[1] - x_coords[0]) * 6)) if len(x_coords) > 1 else 650.0
        sigma_y = max(650.0, float(np.abs(y_coords[1] - y_coords[0]) * 6)) if len(y_coords) > 1 else 650.0
        prior_fields.append(
            np.exp(
                -0.5 * (((x_mesh - peak_x) / sigma_x) ** 2 + ((y_mesh - peak_y) / sigma_y) ** 2)
            ).astype(np.float32)
        )
        prior_centers = [dominant_center]

    gaussian_prior = np.max(np.stack(prior_fields, axis=0), axis=0).astype(np.float32)
    prior_weight = (0.22 + 0.78 * gaussian_prior).astype(np.float32)

    return {
        'response': anomaly_response,
        'center': dominant_center,
        'bbox': dominant_bbox,
        'centers': np.asarray(prior_centers, dtype=np.float64),
        'bboxes': prior_bboxes,
        'weight': prior_weight,
        'threshold': threshold,
    }


def _assess_aux_response_confidence(volume, edge_ignore=4, min_peak=0.08, min_dynamic_range=0.03):
    """Reject nearly-flat auxiliary heads so weak noise is not normalized into a fake target."""
    volume = np.asarray(volume, dtype=np.float32)
    interior_values = _interior_volume_view(volume, edge_ignore=edge_ignore).reshape(-1)
    interior_values = interior_values[np.isfinite(interior_values)]
    if interior_values.size == 0:
        return False, {'peak': 0.0, 'median': 0.0, 'dynamic_range': 0.0}

    peak = float(np.max(interior_values))
    median = float(np.median(interior_values))
    p995 = float(np.quantile(interior_values, 0.995))
    dynamic_range = p995 - median
    is_confident = peak >= min_peak and dynamic_range >= min_dynamic_range
    return is_confident, {
        'peak': peak,
        'median': median,
        'dynamic_range': dynamic_range,
    }


def _compute_weighted_volume_center(volume, x_coords, y_coords, z_coords):
    """Compute a continuous center of mass from a non-negative response volume."""
    weights = np.clip(np.asarray(volume, dtype=np.float64), 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        return None

    x_profile = weights.sum(axis=(1, 2))
    y_profile = weights.sum(axis=(0, 2))
    z_profile = weights.sum(axis=(0, 1))
    return (
        float(np.dot(x_profile, x_coords) / total),
        float(np.dot(y_profile, y_coords) / total),
        float(np.dot(z_profile, z_coords) / total),
    )


def _build_trusted_body_response(source_response, surface_prior_weight, edge_ignore=4, min_depth_idx=5):
    """Suppress edge and ultra-shallow artifacts before center/bbox extraction."""
    trusted = np.asarray(source_response, dtype=np.float32) * np.asarray(surface_prior_weight, dtype=np.float32)[..., None]
    trusted = _zero_boundary_shell(trusted, shell_width=edge_ignore, fill_value=0.0)
    min_depth_idx = max(0, int(min_depth_idx))
    if min_depth_idx > 0 and min_depth_idx < trusted.shape[2]:
        trusted[:, :, :min_depth_idx] = 0.0
    return _normalize_source_response(trusted, edge_ignore=edge_ignore)


def _combine_body_support_response(source_response, pred_body_mask=None, pred_center_heatmap=None, edge_ignore=4):
    """Fuse field-derived response with directly predicted body support when available."""
    combined = _normalize_source_response(source_response, edge_ignore=edge_ignore)
    diagnostics = {
        'body_mask_used': False,
        'center_heatmap_used': False,
        'body_mask_stats': None,
        'center_heatmap_stats': None,
    }

    if pred_body_mask is not None:
        body_ok, body_stats = _assess_aux_response_confidence(
            pred_body_mask,
            edge_ignore=edge_ignore,
            min_peak=0.08,
            min_dynamic_range=0.025
        )
        diagnostics['body_mask_stats'] = body_stats
        if body_ok:
            body_mask = _normalize_source_response(pred_body_mask, edge_ignore=edge_ignore)
            combined = 0.58 * body_mask + 0.42 * combined
            diagnostics['body_mask_used'] = True

    if pred_center_heatmap is not None:
        center_ok, center_stats = _assess_aux_response_confidence(
            pred_center_heatmap,
            edge_ignore=edge_ignore,
            min_peak=0.12,
            min_dynamic_range=0.04
        )
        diagnostics['center_heatmap_stats'] = center_stats
        if center_ok:
            center_heatmap = _normalize_source_response(pred_center_heatmap, edge_ignore=edge_ignore)
            combined = 0.90 * combined + 0.10 * center_heatmap
            diagnostics['center_heatmap_used'] = True

    return _normalize_source_response(combined, edge_ignore=edge_ignore), diagnostics


def _resolve_positive_level(volume, percentile=97.0, edge_ignore=4, min_factor=0.25, max_factor=0.98):
    """Pick a positive threshold suitable for source-response contours/isosurfaces."""
    positive_values = _interior_volume_view(volume, edge_ignore=edge_ignore)
    positive_values = positive_values[np.isfinite(positive_values)]
    positive_values = positive_values[positive_values > 0]
    if positive_values.size == 0:
        return None

    vmax = float(positive_values.max())
    if vmax <= 0:
        return None

    level = float(np.percentile(positive_values, percentile))
    return min(max(level, min_factor * vmax), max_factor * vmax)


def _largest_component_mask(mask):
    """Keep the dominant connected response region and discard isolated speckles."""
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return mask

    labeled, num_components = label(mask)
    if num_components <= 1:
        return mask

    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0
    dominant_label = int(np.argmax(component_sizes))
    return labeled == dominant_label


def _extract_connected_components(mask, axis_coords, axis_names, value_volume=None, min_size=1, max_components=None):
    """Return connected components sorted by integrated response instead of keeping only the largest."""
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return []

    labeled, num_components = label(mask)
    if num_components <= 0:
        return []

    component_sizes = np.bincount(labeled.ravel())
    candidate_labels = [label_id for label_id in range(1, num_components + 1) if component_sizes[label_id] >= max(1, int(min_size))]
    if not candidate_labels:
        candidate_labels = [int(np.argmax(component_sizes[1:]) + 1)]

    coords_tuple = tuple(np.asarray(coords, dtype=np.float64).reshape(-1) for coords in axis_coords)
    value_array = None
    if value_volume is not None:
        candidate_values = np.asarray(value_volume, dtype=np.float32)
        if candidate_values.shape == mask.shape:
            value_array = candidate_values

    components = []
    for label_id in candidate_labels:
        component_mask = labeled == label_id
        occupied = np.argwhere(component_mask)
        if occupied.size == 0:
            continue

        bbox = {}
        for axis_index, (axis_name, coords) in enumerate(zip(axis_names, coords_tuple)):
            bbox[f'{axis_name}_min'] = float(coords[occupied[:, axis_index].min()])
            bbox[f'{axis_name}_max'] = float(coords[occupied[:, axis_index].max()])

        if value_array is not None:
            weights = np.clip(value_array, 0.0, None) * component_mask.astype(np.float32)
            total = float(np.sum(weights))
        else:
            weights = None
            total = 0.0

        if total > 0.0:
            center = []
            for axis_index, coords in enumerate(coords_tuple):
                reduce_axes = tuple(idx for idx in range(component_mask.ndim) if idx != axis_index)
                profile = np.sum(weights, axis=reduce_axes)
                center.append(float(np.dot(profile, coords) / total))
            peak = float(np.max(value_array[component_mask]))
            score = total
        else:
            center = [
                float(np.mean(coords[occupied[:, axis_index]]))
                for axis_index, coords in enumerate(coords_tuple)
            ]
            peak = float(component_sizes[label_id])
            score = float(component_sizes[label_id])

        components.append({
            'label': int(label_id),
            'mask': component_mask,
            'size': int(component_sizes[label_id]),
            'bbox': bbox,
            'center': tuple(center),
            'peak': float(peak),
            'score': float(score),
        })

    components.sort(key=lambda item: (item['score'], item['peak'], item['size']), reverse=True)
    if max_components is not None and max_components > 0:
        return components[:int(max_components)]
    return components


def _compute_mask_bounds(mask, x_coords, y_coords, z_coords):
    """Compute an axis-aligned physical bounding box from a voxel mask."""
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return None

    occupied = np.argwhere(mask)
    return {
        'x_min': float(x_coords[occupied[:, 0].min()]),
        'x_max': float(x_coords[occupied[:, 0].max()]),
        'y_min': float(y_coords[occupied[:, 1].min()]),
        'y_max': float(y_coords[occupied[:, 1].max()]),
        'z_min': float(z_coords[occupied[:, 2].min()]),
        'z_max': float(z_coords[occupied[:, 2].max()]),
    }


def _bbox_to_array(bbox):
    """Serialize an optional bounding box into a compact NumPy array."""
    if bbox is None:
        return np.array([], dtype=np.float64)
    return np.array(
        [bbox['x_min'], bbox['x_max'], bbox['y_min'], bbox['y_max'], bbox['z_min'], bbox['z_max']],
        dtype=np.float64
    )


def _bbox_list_to_array(bboxes, axis_names=('x', 'y', 'z')):
    """Serialize multiple bounding boxes into an (N, 2*D) array."""
    rows = []
    for bbox in bboxes or []:
        if bbox is None:
            continue
        row = []
        valid = True
        for axis_name in axis_names:
            min_key = f'{axis_name}_min'
            max_key = f'{axis_name}_max'
            if min_key not in bbox or max_key not in bbox:
                valid = False
                break
            row.extend([float(bbox[min_key]), float(bbox[max_key])])
        if valid and np.all(np.isfinite(row)):
            rows.append(row)
    if not rows:
        return np.array([], dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _format_bbox(bbox):
    """Format a bounding box for concise logging."""
    if bbox is None:
        return "unresolved"
    return (
        f"x=[{bbox['x_min']:.1f}, {bbox['x_max']:.1f}], "
        f"y=[{bbox['y_min']:.1f}, {bbox['y_max']:.1f}], "
        f"z=[{bbox['z_min']:.1f}, {bbox['z_max']:.1f}]"
    )


def _format_bbox_collection(bboxes):
    """Format several bounding boxes for concise logging."""
    if not bboxes:
        return "unresolved"
    return '; '.join(_format_bbox(bbox) for bbox in bboxes)


def _estimate_body_components(source_response, pred_body_mask, aux_diagnostics, x_coords, y_coords, z_coords, edge_ignore=4, min_depth_idx=0):
    """Estimate one or more body extents, preferring the dedicated body-mask head when it is trustworthy."""
    component_source = 'source_response'
    component_volume = np.asarray(source_response, dtype=np.float32)
    core_level = _resolve_positive_level(component_volume, percentile=97.0, edge_ignore=edge_ignore, min_factor=0.45)
    envelope_level = _resolve_positive_level(component_volume, percentile=90.0, edge_ignore=edge_ignore, min_factor=0.25)

    if pred_body_mask is not None and aux_diagnostics.get('body_mask_used', False):
        body_volume = np.asarray(pred_body_mask, dtype=np.float32)
        body_volume = _zero_boundary_shell(body_volume, shell_width=edge_ignore, fill_value=0.0)
        if 0 < min_depth_idx < body_volume.shape[2]:
            body_volume[:, :, :min_depth_idx] = 0.0
        interior_values = _interior_volume_view(body_volume, edge_ignore=edge_ignore)
        interior_values = interior_values[np.isfinite(interior_values)]
        interior_values = interior_values[interior_values > 0.05]
        if interior_values.size > 0:
            component_source = 'body_mask'
            component_volume = body_volume
            envelope_level = float(np.clip(np.percentile(interior_values, 60.0), 0.22, 0.58))
            core_level = float(np.clip(np.percentile(interior_values, 82.0), 0.40, 0.80))

    component_axis = (x_coords, y_coords, z_coords)
    core_components = _extract_connected_components(
        component_volume >= core_level,
        axis_coords=component_axis,
        axis_names=('x', 'y', 'z'),
        value_volume=component_volume,
        min_size=max(18, int(edge_ignore) * 4),
        max_components=8,
    ) if core_level is not None else []
    envelope_components = _extract_connected_components(
        component_volume >= envelope_level,
        axis_coords=component_axis,
        axis_names=('x', 'y', 'z'),
        value_volume=component_volume,
        min_size=max(24, int(edge_ignore) * 6),
        max_components=8,
    ) if envelope_level is not None else []

    if not envelope_components and component_source == 'body_mask':
        component_source = 'source_response'
        component_volume = np.asarray(source_response, dtype=np.float32)
        core_level = _resolve_positive_level(component_volume, percentile=97.0, edge_ignore=edge_ignore, min_factor=0.45)
        envelope_level = _resolve_positive_level(component_volume, percentile=90.0, edge_ignore=edge_ignore, min_factor=0.25)
        core_components = _extract_connected_components(
            component_volume >= core_level,
            axis_coords=component_axis,
            axis_names=('x', 'y', 'z'),
            value_volume=component_volume,
            min_size=max(18, int(edge_ignore) * 4),
            max_components=8,
        ) if core_level is not None else []
        envelope_components = _extract_connected_components(
            component_volume >= envelope_level,
            axis_coords=component_axis,
            axis_names=('x', 'y', 'z'),
            value_volume=component_volume,
            min_size=max(24, int(edge_ignore) * 6),
            max_components=8,
        ) if envelope_level is not None else []

    dominant_component = envelope_components[0] if envelope_components else (core_components[0] if core_components else None)
    dominant_mask = dominant_component['mask'] if dominant_component is not None else None
    dominant_center = dominant_component['center'] if dominant_component is not None else None

    return {
        'component_source': component_source,
        'component_volume': component_volume,
        'core_level': core_level,
        'envelope_level': envelope_level,
        'core_components': core_components,
        'envelope_components': envelope_components,
        'dominant_mask': dominant_mask,
        'dominant_center': dominant_center,
        'dominant_core_bbox': core_components[0]['bbox'] if core_components else None,
        'dominant_envelope_bbox': envelope_components[0]['bbox'] if envelope_components else None,
        'centers': np.asarray([component['center'] for component in envelope_components], dtype=np.float64) if envelope_components else np.array([], dtype=np.float64),
    }


def _plot_bbox_collection(ax, bboxes, plane, edgecolor, linestyle='-', linewidth=2.0):
    """Draw every bounding box in a collection on a chosen 2D projection."""
    if not SHOW_BOX_ANNOTATIONS:
        return
    axis_pairs = {
        'xy': ('x', 'y'),
        'xz': ('x', 'z'),
        'yz': ('y', 'z'),
    }
    axis_a, axis_b = axis_pairs[plane]
    for bbox in bboxes or []:
        if bbox is None:
            continue
        ax.add_patch(
            Rectangle(
                (bbox[f'{axis_a}_min'], bbox[f'{axis_b}_min']),
                bbox[f'{axis_a}_max'] - bbox[f'{axis_a}_min'],
                bbox[f'{axis_b}_max'] - bbox[f'{axis_b}_min'],
                linewidth=linewidth,
                edgecolor=edgecolor,
                facecolor='none',
                linestyle=linestyle,
            )
        )


def _plot_center_collection(ax, centers, axes, primary_color='white', secondary_color='white', primary_marker='x', secondary_marker='+', primary_size=80, secondary_size=55):
    """Draw the dominant center first and keep additional centers visible but lighter."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.size == 0:
        return
    if centers.ndim == 1:
        centers = centers.reshape(1, -1)
    coords = centers[:, axes]
    if coords.shape[0] > 1:
        ax.scatter(
            coords[1:, 0],
            coords[1:, 1],
            c=secondary_color,
            marker=secondary_marker,
            s=secondary_size,
            linewidths=1.4,
        )
    ax.scatter(
        coords[0, 0],
        coords[0, 1],
        c=primary_color,
        marker=primary_marker,
        s=primary_size,
        linewidths=1.8,
    )


def _plot_depth_spans(ax, bboxes, color, alpha, label=None):
    """Draw depth ranges for multiple bodies without duplicating legend labels."""
    if not SHOW_BOX_ANNOTATIONS:
        return
    first = True
    for bbox in bboxes or []:
        if bbox is None:
            continue
        ax.axvspan(
            bbox['z_min'],
            bbox['z_max'],
            color=color,
            alpha=alpha,
            label=label if first else None,
        )
        first = False


def _resolve_isosurface_level(volume, percentile=97.0, edge_ignore=4):
    """Pick a robust positive level for marching-cubes visualization."""
    positive_values = _interior_volume_view(volume, edge_ignore=edge_ignore)
    positive_values = positive_values[np.isfinite(positive_values)]
    positive_values = positive_values[positive_values > 0]
    if positive_values.size == 0:
        return None

    vmax = float(positive_values.max())
    if vmax <= 0:
        return None

    level = float(np.percentile(positive_values, percentile))
    return min(max(level, 0.35 * vmax), 0.98 * vmax)


def _plot_isosurface(ax, volume, spacing, title, level=None, cmap_name='viridis', edge_ignore=4):
    """Render a scalar-volume isosurface using marching cubes."""
    plot_volume = _zero_boundary_shell(volume, shell_width=edge_ignore, fill_value=0.0)

    if level is None:
        level = _resolve_isosurface_level(plot_volume, edge_ignore=edge_ignore)

    if level is None or not np.isfinite(level):
        ax.text2D(0.5, 0.5, "No positive isosurface", transform=ax.transAxes, ha='center', va='center')
        ax.set_title(title, fontsize=11, fontweight='bold')
        return None

    volume_min = float(np.min(plot_volume))
    volume_max = float(np.max(plot_volume))
    if not (volume_min < level < volume_max):
        ax.text2D(0.5, 0.5, "Invalid isosurface level", transform=ax.transAxes, ha='center', va='center')
        ax.set_title(title, fontsize=11, fontweight='bold')
        return level

    verts, faces, _, _ = measure.marching_cubes(plot_volume.astype(np.float32), level=level, spacing=spacing)
    face_vertices = verts[faces]
    face_depth = face_vertices[:, :, 2].mean(axis=1)
    norm = colors.Normalize(vmin=float(face_depth.min()), vmax=float(face_depth.max()) + 1e-12)
    face_colors = cm.get_cmap(_resolve_plot_cmap(cmap_name))(norm(face_depth))

    mesh = Poly3DCollection(face_vertices, facecolors=face_colors, edgecolor='none', alpha=0.68)
    ax.add_collection3d(mesh)

    x_extent = volume.shape[0] * spacing[0]
    y_extent = volume.shape[1] * spacing[1]
    z_extent = volume.shape[2] * spacing[2]
    ax.set_xlim(0.0, x_extent)
    ax.set_ylim(0.0, y_extent)
    ax.set_zlim(z_extent, 0.0)
    ax.set_box_aspect((x_extent, y_extent, z_extent))
    ax.view_init(elev=24, azim=-56)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{title}\nIso={level:.3e}", fontsize=11, fontweight='bold')

    return level


def _save_derivative_visualization(
    field_3d,
    output_png_path,
    output_npz_path,
    language='zh',
    dx=50.0,
    dy=50.0,
    dz=50.0,
    x_coords=None,
    y_coords=None,
    z_coords=None,
):
    """Save ASA/THDR projections and 3D isosurfaces for a predicted volume."""
    edge_ignore = 4
    derivative_fields = _compute_volume_derivative_fields(
        field_3d,
        dx=dx,
        dy=dy,
        dz=dz,
        edge_taper_width=edge_ignore
    )
    asa3d = derivative_fields['asa3d']
    thdr = derivative_fields['thdr']

    if x_coords is None:
        x_coords = np.arange(field_3d.shape[0]) * dx
    if y_coords is None:
        y_coords = np.arange(field_3d.shape[1]) * dy
    if z_coords is None:
        z_coords = np.arange(field_3d.shape[2]) * dz

    labels = {
        'zh': {
            'asa_xy': '3D 解析信号振幅投影 (XY)',
            'thdr_xy': '总水平导数投影 (XY)',
            'asa_iso': '3D 解析信号振幅等值面',
            'thdr_iso': 'THDR 等值面',
            'value': '响应强度',
            'saved': f'导数场可视化已保存为 {output_png_path}',
            'saved_npz': f'导数场体数据已保存为 {output_npz_path}',
        },
        'en': {
            'asa_xy': '3D Analytic Signal Projection (XY)',
            'thdr_xy': 'THDR Projection (XY)',
            'asa_iso': '3D Analytic Signal Isosurface',
            'thdr_iso': 'THDR Isosurface',
            'value': 'Response',
            'saved': f'Saved derivative-field visualization to {output_png_path}',
            'saved_npz': f'Saved derivative-field volumes to {output_npz_path}',
        },
    }[language]

    asa_xy = np.max(asa3d, axis=2)
    thdr_xy = np.max(thdr, axis=2)

    fig = plt.figure(figsize=(18, 12))

    ax1 = fig.add_subplot(2, 2, 1)
    pc1 = ax1.pcolormesh(x_coords, y_coords, asa_xy.T, cmap=_resolve_plot_cmap('magma'), shading='auto')
    ax1.set_title(labels['asa_xy'], fontsize=12, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal')
    _publication_colorbar(pc1, ax=ax1, label=labels['value'])

    ax2 = fig.add_subplot(2, 2, 2)
    pc2 = ax2.pcolormesh(x_coords, y_coords, thdr_xy.T, cmap=_resolve_plot_cmap('cividis'), shading='auto')
    ax2.set_title(labels['thdr_xy'], fontsize=12, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal')
    _publication_colorbar(pc2, ax=ax2, label=labels['value'])

    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    asa_level = _plot_isosurface(
        ax3,
        asa3d,
        spacing=(dx, dy, dz),
        title=labels['asa_iso'],
        cmap_name='magma',
        edge_ignore=edge_ignore
    )

    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    thdr_level = _plot_isosurface(
        ax4,
        thdr,
        spacing=(dx, dy, dz),
        title=labels['thdr_iso'],
        cmap_name='cividis',
        edge_ignore=edge_ignore
    )

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=160, bbox_inches='tight')
    np.savez(
        output_npz_path,
        prediction=field_3d,
        prediction_tapered=derivative_fields['tapered_prediction'],
        asa3d=asa3d,
        thdr=thdr,
        x_coords=x_coords,
        y_coords=y_coords,
        z_coords=z_coords,
        derivative_edge_ignore=edge_ignore,
        asa_isosurface_level=asa_level if asa_level is not None else np.nan,
        thdr_isosurface_level=thdr_level if thdr_level is not None else np.nan,
    )
    print(labels['saved'])
    print(labels['saved_npz'])
    plt.show()

    return derivative_fields


def test_trained_model(
    checkpoint_path=None,
    noise_level=0.03,
    add_regional=False,
    model_capacity=None,
    input_mode=None,
    synthetic_case='random'
):
    """使用训练好的模型进行推断"""
    language = _configure_plot_language()
    labels = {
        'zh': {
            'title': "使用训练好的模型进行推断与可视化",
            'input_title': '输入: 地表含噪重力观测',
            'true_title': '真实标签 (XZ切面)',
            'pred_title': '预测结果 (XZ切面)',
            'error_title': '误差分布 (XZ切面)',
            'true_boundary_title': '真实阈值边界 (XZ切面)',
            'pred_boundary_title': '预测阈值边界 (XZ切面)',
            'x_label': 'X 网格索引',
            'y_label': 'Y 网格索引',
            'z_label': 'Z 网格索引 (深度)',
            'saved': '可视化已保存为 test_inference_visualization.png',
        },
        'en': {
            'title': "Inference And Visualization",
            'input_title': 'Input: Surface Gravity With Noise',
            'true_title': 'Ground Truth (XZ Slice)',
            'pred_title': 'Prediction (XZ Slice)',
            'error_title': 'Error Map (XZ Slice)',
            'true_boundary_title': 'Ground Truth Threshold Boundary',
            'pred_boundary_title': 'Prediction Threshold Boundary',
            'x_label': 'X Grid Index',
            'y_label': 'Y Grid Index',
            'z_label': 'Z Grid Index (Depth)',
            'saved': 'Saved visualization to test_inference_visualization.png',
        },
    }[language]

    print("\n" + "="*60)
    print(labels['title'])
    print("="*60)

    if checkpoint_path is None:
        default_capacity = model_capacity or 'medium'
        default_input_mode = input_mode or 'gz_gzz'
        checkpoint_path = _get_default_checkpoint_path(default_capacity, default_input_mode)

    nx, ny, nz = 64, 64, 64
    dx, dy, dz = 50, 50, 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    if Path(checkpoint_path).exists():
        print(f"加载模型权重: {checkpoint_path}")
        model, checkpoint, resolved_capacity, resolved_input_mode, resolved_use_multimodal_stem, has_aux_heads = _load_model_from_checkpoint(
            checkpoint_path,
            device,
            model_capacity=model_capacity,
            input_mode=input_mode
        )
        print(f"模型容量配置: {resolved_capacity}")
        print(f"输入模式: {resolved_input_mode}")
        print(f"多模态输入干路: {'开启' if resolved_use_multimodal_stem else '关闭'}")
        print(f"异常体辅助头: {'可用' if has_aux_heads else '检查点不包含'}")
        print(f"已加载 epoch {checkpoint['epoch']} 的模型")
    else:
        print(f"错误: 未找到检查点 {checkpoint_path}")
        return

    primary_channel_name = _select_primary_channel(resolved_input_mode)
    aux_channel_name = _select_aux_channel(resolved_input_mode)
    aux_title, aux_contour_title, aux_colorbar = _get_aux_channel_labels(language, aux_channel_name)

    generator = GravityDataGenerator()
    if synthetic_case == 'center_box':
        surface_channels, Y_test, anomaly_info = _generate_center_box_case(
            generator,
            noise_level=noise_level,
            add_regional=add_regional
        )
        output_png_path = 'test_center_box_inference_visualization.png'
        derivative_png_path = 'test_center_box_derivative_isosurfaces.png'
        derivative_npz_path = 'test_center_box_derivative_fields.npz'
        print("合成测试体: 中心长方体")
        if anomaly_info:
            info = anomaly_info[0]
            print(
                f"中心长方体参数: center=({info['center'][0]:.1f}, {info['center'][1]:.1f}, {info['center'][2]:.1f}), "
                f"half_size=({info['size_xyz'][0]:.1f}, {info['size_xyz'][1]:.1f}, {info['size_xyz'][2]:.1f}), "
                f"density={info['density']:.1f}"
            )
    elif synthetic_case == 'deep_box':
        surface_channels, Y_test, anomaly_info = _generate_deep_box_case(
            generator,
            noise_level=noise_level,
            add_regional=add_regional
        )
        output_png_path = 'test_deep_box_inference_visualization.png'
        derivative_png_path = 'test_deep_box_derivative_isosurfaces.png'
        derivative_npz_path = 'test_deep_box_derivative_fields.npz'
        print("Synthetic test case: deep_box")
        if anomaly_info:
            info = anomaly_info[0]
            print(
                f"Deep box parameters: center=({info['center'][0]:.1f}, {info['center'][1]:.1f}, {info['center'][2]:.1f}), "
                f"half_size=({info['size_xyz'][0]:.1f}, {info['size_xyz'][1]:.1f}, {info['size_xyz'][2]:.1f}), "
                f"density={info['density']:.1f}"
            )
    elif synthetic_case == 'complex_geology':
        surface_channels, Y_test, anomaly_info = _generate_complex_geology_case(
            generator,
            noise_level=noise_level,
            add_regional=add_regional
        )
        output_png_path = 'test_complex_geology_inference_visualization.png'
        derivative_png_path = 'test_complex_geology_derivative_isosurfaces.png'
        derivative_npz_path = 'test_complex_geology_derivative_fields.npz'
        print("Synthetic test case: complex_geology")
        for info in anomaly_info:
            print(
                f"  - {info['type']}: center=({info['center'][0]:.1f}, {info['center'][1]:.1f}, {info['center'][2]:.1f}), "
                f"size=({info['size_xyz'][0]:.1f}, {info['size_xyz'][1]:.1f}, {info['size_xyz'][2]:.1f}), "
                f"density={info['density']:.1f}"
            )
    elif synthetic_case == 'single_tilted_bar':
        surface_channels, Y_test, anomaly_info = _generate_single_tilted_bar_case(
            generator,
            noise_level=noise_level,
            add_regional=add_regional
        )
        output_png_path = 'test_single_tilted_bar_inference_visualization.png'
        derivative_png_path = 'test_single_tilted_bar_derivative_isosurfaces.png'
        derivative_npz_path = 'test_single_tilted_bar_derivative_fields.npz'
        print("Synthetic test case: single_tilted_bar")
        if anomaly_info:
            info = anomaly_info[0]
            print(
                f"Tilted bar parameters: center=({info['center'][0]:.1f}, {info['center'][1]:.1f}, {info['center'][2]:.1f}), "
                f"size=({info['size_xyz'][0]:.1f}, {info['size_xyz'][1]:.1f}, {info['size_xyz'][2]:.1f}), "
                f"density={info['density']:.1f}, rotation={info.get('rotation_degrees', ())}"
            )
    elif synthetic_case == 'wide_dual_tilted_bars':
        surface_channels, Y_test, anomaly_info = _generate_wide_dual_tilted_bars_case(
            generator,
            noise_level=noise_level,
            add_regional=add_regional
        )
        output_png_path = 'test_wide_dual_tilted_bars_inference_visualization.png'
        derivative_png_path = 'test_wide_dual_tilted_bars_derivative_isosurfaces.png'
        derivative_npz_path = 'test_wide_dual_tilted_bars_derivative_fields.npz'
        print("Synthetic test case: wide_dual_tilted_bars")
        for info in anomaly_info:
            print(
                f"  - {info['type']}: center=({info['center'][0]:.1f}, {info['center'][1]:.1f}, {info['center'][2]:.1f}), "
                f"size=({info['size_xyz'][0]:.1f}, {info['size_xyz'][1]:.1f}, {info['size_xyz'][2]:.1f}), "
                f"density={info['density']:.1f}, rotation={info.get('rotation_degrees', ())}"
            )
    else:
        surface_channels, Y_test, anomaly_info = generator.generate_sample(
            noise_level=noise_level,
            add_regional=add_regional,
            return_surface_channels=True
        )
        output_png_path = 'test_inference_visualization.png'
        derivative_png_path = 'test_derivative_isosurfaces.png'
        derivative_npz_path = 'test_derivative_fields.npz'
    input_array, target_scale, normalization_mode = _build_inference_input(
        surface_channels,
        input_mode=resolved_input_mode,
        checkpoint=checkpoint,
        label_3d=Y_test
    )
    X_test = surface_channels[primary_channel_name]
    aux_surface = surface_channels[aux_channel_name]
    print(f"输入归一化: {normalization_mode}")

    model.eval()
    with torch.no_grad():
        input_tensor = torch.from_numpy(input_array[np.newaxis, :, :, :]).float().to(device)
        output_dict = model(input_tensor)
        Y_pred_norm = output_dict['output'].squeeze().cpu().numpy()

    Y_pred = Y_pred_norm * target_scale

    mae = np.mean(np.abs(Y_test - Y_pred))
    rmse = np.sqrt(np.mean((Y_test - Y_pred)**2))

    print(f"MAE: {mae:.6f} mGal")
    print(f"RMSE: {rmse:.6f} mGal")

    # 获取异常体信息用于绘制边界
    cx_idx = cy_idx = cz_idx = 32
    hx_idx = hy_idx = hz_idx = 10

    if anomaly_info:
        info = anomaly_info[0]
        cx, cy, cz = info['center']
        cx_idx, cy_idx, cz_idx = cx / dx, cy / dy, cz / dz
        if isinstance(info['size'], tuple):
            hx_idx, hy_idx, hz_idx = info['size'][0] / dx, info['size'][1] / dy, info['size'][2] / dz
        else:
            hx_idx = hy_idx = hz_idx = info['size'] / dx

    # 可视化 - 2行4列
    fig = plt.figure(figsize=(20, 10))

    # 1. 输入 (2D 地表观测)
    ax1 = fig.add_subplot(2, 4, 1)
    pc1 = ax1.pcolormesh(X_test.T, cmap=_resolve_plot_cmap('jet'), shading='auto')
    ax1.set_title(labels['input_title'], fontsize=12, fontweight='bold')
    ax1.set_xlabel(labels['x_label'])
    ax1.set_ylabel(labels['y_label'])
    ax1.set_aspect('equal')
    _publication_colorbar(pc1, ax=ax1, label='mGal')

    # 2. 真实标签 (XZ 切面)
    ax2 = fig.add_subplot(2, 4, 2)
    y_idx = int(cy_idx)
    slice_true_xz = Y_test[:, y_idx, :]
    slice_pred_xz = Y_pred[:, y_idx, :]
    slice_norm, slice_extend = _build_publication_norm(
        np.concatenate((slice_true_xz.reshape(-1), slice_pred_xz.reshape(-1))),
        symmetric=True,
    )
    x_coords = np.arange(slice_true_xz.shape[0])
    z_coords = np.arange(slice_true_xz.shape[1])
    boundary_threshold_ratio = 0.30
    pc2 = ax2.pcolormesh(
        slice_true_xz.T,
        cmap=_resolve_plot_cmap('jet'),
        shading='auto',
        norm=slice_norm,
    )
    ax2.invert_yaxis()
    ax2.set_title(labels['true_title'], fontsize=12, fontweight='bold')
    ax2.set_xlabel(labels['x_label'])
    ax2.set_ylabel(labels['z_label'])
    ax2.set_aspect('equal')
    _publication_colorbar(pc2, ax=ax2, label='mGal', extend=slice_extend, extendrect=True)
    if SHOW_BOX_ANNOTATIONS:
        rect = Rectangle((cx_idx - hx_idx, cz_idx - hz_idx), 2*hx_idx, 2*hz_idx,
                        linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
        ax2.add_patch(rect)

    # 3. 预测结果 (XZ 切面)
    ax3 = fig.add_subplot(2, 4, 3)
    pc3 = ax3.pcolormesh(
        slice_pred_xz.T,
        cmap=_resolve_plot_cmap('jet'),
        shading='auto',
        norm=slice_norm,
    )
    ax3.invert_yaxis()
    ax3.set_title(labels['pred_title'], fontsize=12, fontweight='bold')
    ax3.set_xlabel(labels['x_label'])
    ax3.set_ylabel(labels['z_label'])
    ax3.set_aspect('equal')
    _publication_colorbar(pc3, ax=ax3, label='mGal', extend=slice_extend, extendrect=True)
    if SHOW_BOX_ANNOTATIONS:
        rect = Rectangle((cx_idx - hx_idx, cz_idx - hz_idx), 2*hx_idx, 2*hz_idx,
                        linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
        ax3.add_patch(rect)
    pred_weights = np.abs(slice_pred_xz)
    if pred_weights.sum() > 0:
        pred_x_center = float((pred_weights.sum(axis=1) * np.arange(pred_weights.shape[0])).sum() / pred_weights.sum())
        pred_z_center = float((pred_weights.sum(axis=0) * np.arange(pred_weights.shape[1])).sum() / pred_weights.sum())
        ax3.scatter(pred_x_center, pred_z_center, c='white', marker='x', s=70)

    abs_true_xz = np.abs(slice_true_xz)
    abs_pred_xz = np.abs(slice_pred_xz)
    boundary_threshold = boundary_threshold_ratio * max(abs_true_xz.max(), abs_pred_xz.max(), 1e-8)

    # 4. 辅助输入通道 (2D)
    ax4 = fig.add_subplot(2, 4, 4)
    pc4 = ax4.pcolormesh(aux_surface.T, cmap=_resolve_plot_cmap('jet'), shading='auto')
    ax4.set_title(aux_title, fontsize=12, fontweight='bold')
    ax4.set_xlabel(labels['x_label'])
    ax4.set_ylabel(labels['y_label'])
    ax4.set_aspect('equal')
    _publication_colorbar(pc4, ax=ax4, label=aux_colorbar)

    # 5. 误差分布 (XZ 切面)
    ax5 = fig.add_subplot(2, 4, 5)
    error_xz = slice_true_xz - slice_pred_xz
    error_norm, error_extend = _build_publication_norm(error_xz, symmetric=True)
    pc5 = ax5.pcolormesh(
        error_xz.T,
        cmap=_resolve_plot_cmap('jet'),
        shading='auto',
        norm=error_norm,
    )
    ax5.invert_yaxis()
    ax5.set_title(labels['error_title'], fontsize=12, fontweight='bold')
    ax5.set_xlabel(labels['x_label'])
    ax5.set_ylabel(labels['z_label'])
    ax5.set_aspect('equal')
    _publication_colorbar(pc5, ax=ax5, label='mGal', extend=error_extend, extendrect=True)

    # 6. 真实阈值边界 (XZ 切面)
    ax6 = fig.add_subplot(2, 4, 6)
    ax6.contourf(x_coords, z_coords, abs_true_xz.T, levels=20, cmap=_resolve_plot_cmap('jet'))
    if boundary_threshold <= abs_true_xz.max():
        ax6.contour(
            x_coords,
            z_coords,
            abs_true_xz.T,
            levels=[boundary_threshold],
            colors='white',
            linewidths=2
        )
    ax6.invert_yaxis()
    ax6.set_title(
        f"{labels['true_boundary_title']} ({int(boundary_threshold_ratio * 100)}%)",
        fontsize=12,
        fontweight='bold'
    )
    ax6.set_xlabel(labels['x_label'])
    ax6.set_ylabel(labels['z_label'])
    ax6.set_aspect('equal')
    if SHOW_BOX_ANNOTATIONS:
        rect = Rectangle((cx_idx - hx_idx, cz_idx - hz_idx), 2*hx_idx, 2*hz_idx,
                        linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
        ax6.add_patch(rect)

    # 7. 预测阈值边界 (XZ 切面)
    ax7 = fig.add_subplot(2, 4, 7)
    ax7.contourf(x_coords, z_coords, abs_pred_xz.T, levels=20, cmap=_resolve_plot_cmap('jet'))
    if boundary_threshold <= abs_pred_xz.max():
        ax7.contour(
            x_coords,
            z_coords,
            abs_pred_xz.T,
            levels=[boundary_threshold],
            colors='white',
            linewidths=2
        )
    ax7.invert_yaxis()
    ax7.set_title(
        f"{labels['pred_boundary_title']} ({int(boundary_threshold_ratio * 100)}%)",
        fontsize=12,
        fontweight='bold'
    )
    ax7.set_xlabel(labels['x_label'])
    ax7.set_ylabel(labels['z_label'])
    ax7.set_aspect('equal')
    if SHOW_BOX_ANNOTATIONS:
        rect = Rectangle((cx_idx - hx_idx, cz_idx - hz_idx), 2*hx_idx, 2*hz_idx,
                        linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
        ax7.add_patch(rect)
    if pred_weights.sum() > 0:
        ax7.scatter(pred_x_center, pred_z_center, c='white', marker='x', s=70)

    # 8. 辅助输入通道等值线
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.contourf(np.arange(64), np.arange(64), aux_surface.T, levels=20, cmap=_resolve_plot_cmap('jet'))
    ax8.set_title(aux_contour_title, fontsize=12, fontweight='bold')
    ax8.set_xlabel(labels['x_label'])
    ax8.set_ylabel(labels['y_label'])
    ax8.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=150, bbox_inches='tight')
    print(f"可视化已保存为 {output_png_path}" if language == 'zh' else f"Saved visualization to {output_png_path}")
    plt.show()

    _save_derivative_visualization(
        Y_pred,
        output_png_path=derivative_png_path,
        output_npz_path=derivative_npz_path,
        language=language,
        dx=dx,
        dy=dy,
        dz=dz,
        x_coords=np.arange(nx) * dx,
        y_coords=np.arange(ny) * dy,
        z_coords=np.arange(nz) * dz,
    )


def test_real_data_model(
    real_data_path=None,
    real_gz_path=None,
    real_gzz_path=None,
    checkpoint_path=None,
    model_capacity=None,
    input_mode=None,
    surface_calibration=False,
    surface_calibration_gz_weight=5.0,
    surface_calibration_gzz_weight=5.0,
):
    """Run inference on real gz/gzz surface grids."""
    language = _configure_plot_language()
    labels = {
        'zh': {
            'title': "真实数据推断与可视化",
            'input_title': '真实数据输入 (重采样后)',
            'xy_title': '预测 XY 切面',
            'xz_title': '预测 XZ 切面',
            'yz_title': '预测 YZ 切面',
            'source_xy_title': '异常体响应投影 (XY)',
            'source_xz_title': '异常体范围估计 (XZ)',
            'depth_profile_title': '异常体响应深度曲线',
            'x_label': 'X 坐标',
            'y_label': 'Y 坐标',
            'z_label': '深度',
            'value_label': '场值',
            'energy_label': '能量',
            'response_label': '异常体响应',
            'core_label': '核心范围',
            'envelope_label': '外围范围',
            'saved': '真实数据推断图已保存为 real_data_inference_visualization.png',
        },
        'en': {
            'title': "Real Data Inference And Visualization",
            'input_title': 'Real Input (Resampled)',
            'xy_title': 'Predicted XY Slice',
            'xz_title': 'Predicted XZ Slice',
            'yz_title': 'Predicted YZ Slice',
            'source_xy_title': 'Anomaly-Body Response Projection (XY)',
            'source_xz_title': 'Estimated Body Extent (XZ)',
            'depth_profile_title': 'Body-Response Depth Profile',
            'x_label': 'X Coordinate',
            'y_label': 'Y Coordinate',
            'z_label': 'Depth',
            'value_label': 'Field Value',
            'energy_label': 'Energy',
            'response_label': 'Body Response',
            'core_label': 'Core Extent',
            'envelope_label': 'Envelope Extent',
            'saved': 'Saved real-data visualization to real_data_inference_visualization.png',
        },
    }[language]

    if checkpoint_path is None:
        default_capacity = model_capacity or 'medium'
        default_input_mode = input_mode or 'gz_gzz'
        checkpoint_path = _get_default_checkpoint_path(default_capacity, default_input_mode)

    print("\n" + "=" * 60)
    print(labels['title'])
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"加载模型权重: {checkpoint_path}")

    model, checkpoint, resolved_capacity, resolved_input_mode, resolved_use_multimodal_stem, has_aux_heads = _load_model_from_checkpoint(
        checkpoint_path,
        device,
        model_capacity=model_capacity,
        input_mode=input_mode
    )
    print(f"模型容量配置: {resolved_capacity}")
    print(f"输入模式: {resolved_input_mode}")
    print(f"多模态输入干路: {'开启' if resolved_use_multimodal_stem else '关闭'}")
    print(f"异常体辅助头: {'可用' if has_aux_heads else '检查点不包含，回退到场响应后处理'}")
    print(f"已加载 epoch {checkpoint['epoch']} 的模型")

    required_channels = get_input_channel_names(resolved_input_mode)

    if real_data_path is not None:
        if 'gz' in required_channels and real_gz_path is None:
            real_gz_path = real_data_path
        elif 'gzz' in required_channels and real_gzz_path is None:
            real_gzz_path = real_data_path

    primary_grid_path = real_gz_path if 'gz' in required_channels else real_gzz_path
    if primary_grid_path is None:
        raise ValueError(
            f"Real-data inference for input_mode='{resolved_input_mode}' requires at least one of: "
            f"--real-gz (for gz-based modes) or --real-gzz (for gzz-based modes)."
        )

    primary_xs, primary_ys, primary_grid_raw = _load_real_surface_grid(primary_grid_path)
    x_model, y_model, primary_grid_model = _resample_grid_to_model_size(primary_xs, primary_ys, primary_grid_raw, target_size=64)
    primary_grid_model = primary_grid_model - np.mean(primary_grid_model)

    dx = float(x_model[1] - x_model[0]) if x_model.size > 1 else 1.0
    dy = float(y_model[1] - y_model[0]) if y_model.size > 1 else 1.0

    surface_channels = {}
    unit_conversions = []
    gz_raw = None
    gz_model = None
    gz_model_centered = None
    if real_gz_path is not None:
        gz_xs, gz_ys, gz_raw = _load_real_surface_grid(real_gz_path)
        _, _, gz_model = _resample_grid_to_model_size(gz_xs, gz_ys, gz_raw, target_size=64)
        gz_model, gz_unit_info = _convert_real_channel_to_training_units('gz', gz_model)
        unit_conversions.append(gz_unit_info)
        gz_model_centered = gz_model - np.mean(gz_model)

    if 'gz' in required_channels:
        if gz_model_centered is None:
            raise ValueError("This checkpoint expects gz input. Please provide --real-gz.")
        surface_channels['gz'] = gz_model_centered

    if 'gzz' in required_channels:
        if real_gzz_path is None:
            raise ValueError("This checkpoint expects gzz input. Please provide --real-gzz.")
        gzz_xs, gzz_ys, gzz_raw = _load_real_surface_grid(real_gzz_path)
        _, _, gzz_model = _resample_grid_to_model_size(gzz_xs, gzz_ys, gzz_raw, target_size=64)
        gzz_model, gzz_unit_info = _convert_real_channel_to_training_units('gzz', gzz_model)
        unit_conversions.append(gzz_unit_info)
        surface_channels['gzz'] = gzz_model - np.mean(gzz_model)
    else:
        gzz_raw = None
        gzz_model = None

    if 'amp' in required_channels:
        if 'gz' in surface_channels:
            amp_source = surface_channels['gz']
        else:
            raise ValueError("Analytic-signal input mode requires gz input to derive amp.")
        surface_channels['amp'] = _compute_surface_analytic_amplitude(amp_source, dx=dx, dy=dy)

    aux_channel_name = _select_aux_channel(resolved_input_mode)
    primary_channel_name = _select_primary_channel(resolved_input_mode)
    aux_surface = surface_channels[aux_channel_name]
    aux_title, _, aux_colorbar = _get_aux_channel_labels(language, aux_channel_name)

    input_array, target_scale, normalization_mode = _build_inference_input(
        surface_channels,
        input_mode=resolved_input_mode,
        checkpoint=checkpoint
    )
    print(f"输入归一化: {normalization_mode}")
    for unit_info in unit_conversions:
        print(
            f"真实{unit_info['channel']}单位处理: scale={unit_info['applied_scale']:.1e}, "
            f"mode={unit_info['assumed_unit']}, abs_max={unit_info['input_abs_max']:.3e}"
        )

    print(
        f"输入网格: 原始 {primary_grid_raw.shape[0]}x{primary_grid_raw.shape[1]} -> "
        f"模型 {primary_grid_model.shape[0]}x{primary_grid_model.shape[1]}"
    )

    model.eval()
    with torch.no_grad():
        input_tensor = torch.from_numpy(input_array[np.newaxis, :, :, :]).float().to(device)
        output_dict = model(input_tensor)
        y_pred_norm = output_dict['output'].squeeze().cpu().numpy()
        pred_body_mask = output_dict.get('body_mask')
        pred_center_heatmap = output_dict.get('center_heatmap')

    pred_body_mask_np = None
    pred_center_heatmap_np = None
    if has_aux_heads and pred_body_mask is not None:
        pred_body_mask_np = pred_body_mask.squeeze().cpu().numpy().astype(np.float32)
    if has_aux_heads and pred_center_heatmap is not None:
        pred_center_heatmap_np = pred_center_heatmap.squeeze().cpu().numpy().astype(np.float32)

    y_pred = y_pred_norm * target_scale
    raw_y_pred = y_pred.copy()
    surface_calibration_diagnostics = {'applied': False, 'reason': 'disabled'}
    if surface_calibration:
        observed_surface_gz = gz_model_centered
        observed_surface_gzz = surface_channels.get('gzz')
        y_pred, surface_calibration_diagnostics = _apply_post_inference_surface_calibration(
            y_pred,
            z_coords=np.arange(y_pred.shape[2], dtype=np.float64) * 50.0,
            observed_gz=observed_surface_gz,
            observed_gzz=observed_surface_gzz,
            gz_weight=surface_calibration_gz_weight,
            gzz_weight=surface_calibration_gzz_weight,
        )
        print(
            f"Surface calibration enabled: gz_weight={surface_calibration_gz_weight:.2f}, "
            f"gzz_weight={surface_calibration_gzz_weight:.2f}, applied={surface_calibration_diagnostics['applied']}"
        )
        if 'surface_gz_before' in surface_calibration_diagnostics and 'surface_gz_after' in surface_calibration_diagnostics:
            before = surface_calibration_diagnostics['surface_gz_before']
            after = surface_calibration_diagnostics['surface_gz_after']
            print(
                f"Surface gz fit: corr {before['corr']:.3f} -> {after['corr']:.3f}, "
                f"rmse {before['rmse']:.3e} -> {after['rmse']:.3e}"
            )
        if 'surface_gzz_before' in surface_calibration_diagnostics and 'surface_gzz_after' in surface_calibration_diagnostics:
            before = surface_calibration_diagnostics['surface_gzz_before']
            after = surface_calibration_diagnostics['surface_gzz_after']
            print(
                f"Surface gzz fit: corr {before['corr']:.3f} -> {after['corr']:.3f}, "
                f"rmse {before['rmse']:.3e} -> {after['rmse']:.3e}"
            )

    pred_mass = np.abs(y_pred) + 1e-12
    z_coords = np.arange(y_pred.shape[2]) * 50.0
    edge_ignore = 4
    min_depth_idx = int(np.searchsorted(z_coords, 250.0, side='left'))

    focus_surface_name = 'gzz' if 'gzz' in surface_channels else primary_channel_name
    surface_focus = _compute_surface_focus_prior(
        surface_channels[focus_surface_name],
        x_model,
        y_model,
        edge_ignore=edge_ignore
    )
    surface_prior_weight = surface_focus['weight']
    surface_prior_bbox = surface_focus['bbox']
    surface_prior_center = surface_focus['center']
    surface_prior_bboxes = list(surface_focus.get('bboxes', []))
    surface_prior_centers = np.asarray(
        surface_focus.get('centers', np.array([surface_prior_center], dtype=np.float64)),
        dtype=np.float64
    )
    if surface_prior_centers.size == 0:
        surface_prior_centers = np.asarray([surface_prior_center], dtype=np.float64)
    surface_prior_bboxes = list(surface_focus.get('bboxes', []))
    surface_prior_centers = np.asarray(
        surface_focus.get('centers', np.array([surface_prior_center], dtype=np.float64)),
        dtype=np.float64
    )
    if surface_prior_centers.size == 0:
        surface_prior_centers = np.asarray([surface_prior_center], dtype=np.float64)

    field_energy_xy = np.sqrt(np.mean(pred_mass ** 2, axis=2))
    field_depth_energy = pred_mass.sum(axis=(0, 1))
    field_x_energy = pred_mass.sum(axis=(1, 2))
    field_y_energy = pred_mass.sum(axis=(0, 2))

    field_x_idx = int(np.argmax(field_x_energy))
    field_y_idx = int(np.argmax(field_y_energy))
    field_z_idx = int(np.argmax(field_depth_energy))
    raw_field_center = np.array(
        [float(x_model[field_x_idx]), float(y_model[field_y_idx]), float(z_coords[field_z_idx])],
        dtype=np.float64
    )

    trusted_field_mass = _zero_boundary_shell(
        pred_mass * surface_prior_weight[..., None],
        shell_width=edge_ignore,
        fill_value=0.0
    )
    if 0 < min_depth_idx < trusted_field_mass.shape[2]:
        trusted_field_mass[:, :, :min_depth_idx] = 0.0
    field_center = _compute_weighted_volume_center(trusted_field_mass, x_model, y_model, z_coords)
    if field_center is None:
        field_center = tuple(raw_field_center.tolist())
    field_x_center, field_y_center, field_z_center = field_center

    source_fields = _compute_source_localization_response(y_pred, dx=dx, dy=dy, dz=50.0, edge_ignore=4)
    source_response_raw, aux_diagnostics = _combine_body_support_response(
        source_fields['source_response'],
        pred_body_mask=pred_body_mask_np,
        pred_center_heatmap=pred_center_heatmap_np,
        edge_ignore=edge_ignore
    )
    source_response = _build_trusted_body_response(
        source_response_raw,
        surface_prior_weight,
        edge_ignore=edge_ignore,
        min_depth_idx=min_depth_idx
    )
    source_projection_xy = np.max(source_response, axis=2)
    source_depth_profile = source_response.sum(axis=(0, 1))
    source_x_energy = source_response.sum(axis=(1, 2))
    source_y_energy = source_response.sum(axis=(0, 2))

    body_estimate = _estimate_body_components(
        source_response,
        pred_body_mask_np,
        aux_diagnostics,
        x_model,
        y_model,
        z_coords,
        edge_ignore=edge_ignore,
        min_depth_idx=min_depth_idx,
    )
    component_source = body_estimate['component_source']
    core_level = body_estimate['core_level']
    envelope_level = body_estimate['envelope_level']
    core_components = body_estimate['core_components']
    envelope_components = body_estimate['envelope_components']
    core_bboxes = [component['bbox'] for component in core_components]
    envelope_bboxes = [component['bbox'] for component in envelope_components]
    core_bbox = body_estimate['dominant_core_bbox']
    envelope_bbox = body_estimate['dominant_envelope_bbox']
    estimated_centers = np.asarray(body_estimate['centers'], dtype=np.float64)

    estimated_center = body_estimate['dominant_center']
    if estimated_center is None:
        x_idx = int(np.argmax(source_x_energy))
        y_idx = int(np.argmax(source_y_energy))
        z_idx = int(np.argmax(source_depth_profile))
        x_center = float(x_model[x_idx])
        y_center = float(y_model[y_idx])
        z_center = float(z_coords[z_idx])
        estimated_centers = np.asarray([[x_center, y_center, z_center]], dtype=np.float64)
    else:
        x_center, y_center, z_center = estimated_center
        x_idx = int(np.argmin(np.abs(x_model - x_center)))
        y_idx = int(np.argmin(np.abs(y_model - y_center)))
        z_idx = int(np.argmin(np.abs(z_coords - z_center)))
        if estimated_centers.size == 0:
            estimated_centers = np.asarray([[x_center, y_center, z_center]], dtype=np.float64)

    print(
        f"地表异常先验中心: x={surface_prior_center[0]:.1f}, y={surface_prior_center[1]:.1f} "
        f"(source={focus_surface_name})"
    )
    print(
        f"辅助体响应融合: body_mask={'启用' if aux_diagnostics['body_mask_used'] else '禁用'}, "
        f"center_heatmap={'启用' if aux_diagnostics['center_heatmap_used'] else '禁用'}"
    )
    body_peak = aux_diagnostics['body_mask_stats']['peak'] if aux_diagnostics['body_mask_stats'] is not None else 0.0
    center_peak = aux_diagnostics['center_heatmap_stats']['peak'] if aux_diagnostics['center_heatmap_stats'] is not None else 0.0
    print(f"body_mask峰值={body_peak:.3e}, center峰值={center_peak:.3e}")
    print(f"多异常体估计源: {component_source}, envelope_count={len(envelope_components)}, core_count={len(core_components)}")
    print(f"3D场可信中心: x={field_x_center:.1f}, y={field_y_center:.1f}, z={field_z_center:.1f}")
    print(f"异常体估计中心: x={x_center:.1f}, y={y_center:.1f}, z={z_center:.1f}")
    print(f"{labels['core_label']}: {_format_bbox_collection(core_bboxes)}")
    print(f"{labels['envelope_label']}: {_format_bbox_collection(envelope_bboxes)}")
    print(f"预测场范围: [{y_pred.min():.3e}, {y_pred.max():.3e}]")

    slice_xy = y_pred[:, :, z_idx]
    slice_xz = y_pred[:, y_idx, :]
    slice_yz = y_pred[x_idx, :, :]
    source_projection_xz = np.max(source_response, axis=1)
    source_extent_xz_display = _build_body_extent_projection_xz(source_response, envelope_level)
    source_slice_yz = source_response[x_idx, :, :]
    prediction_norm, prediction_extend = _build_main_visualization_norm(y_pred)
    source_response_norm, source_response_extend, source_response_levels = _build_body_response_projection_norm(source_response)

    fig = plt.figure(figsize=(20, 10))

    ax1 = fig.add_subplot(2, 4, 1)
    pc1 = ax1.pcolormesh(
        x_model,
        y_model,
        surface_channels[primary_channel_name].T,
        cmap=_resolve_input_plot_cmap(primary_channel_name),
        shading='auto',
    )
    ax1.set_title(labels['input_title'], fontsize=12, fontweight='bold')
    ax1.set_xlabel(labels['x_label'])
    ax1.set_ylabel(labels['y_label'])
    ax1.set_aspect('equal')
    _apply_plan_view_spatial_ticks(ax1, x_model, y_model)
    _plot_bbox_collection(ax1, surface_prior_bboxes, plane='xy', edgecolor='lime', linestyle=':', linewidth=2.0)
    _publication_colorbar(pc1, ax=ax1, label=labels['value_label'])

    ax2 = fig.add_subplot(2, 4, 2)
    pc2 = ax2.pcolormesh(
        x_model,
        y_model,
        aux_surface.T,
        cmap=_resolve_input_plot_cmap(aux_channel_name),
        shading='auto',
    )
    ax2.set_title(aux_title, fontsize=12, fontweight='bold')
    ax2.set_xlabel(labels['x_label'])
    ax2.set_ylabel(labels['y_label'])
    ax2.set_aspect('equal')
    _apply_plan_view_spatial_ticks(ax2, x_model, y_model)
    _plot_bbox_collection(ax2, surface_prior_bboxes, plane='xy', edgecolor='lime', linestyle=':', linewidth=2.0)
    _publication_colorbar(pc2, ax=ax2, label=aux_colorbar)

    ax3 = fig.add_subplot(2, 4, 3)
    pc3 = ax3.pcolormesh(
        x_model,
        y_model,
        slice_xy.T,
        cmap=_resolve_plot_cmap('jet'),
        shading='auto',
        norm=prediction_norm,
    )
    ax3.set_title(f"{labels['xy_title']} (z={z_center:.0f})", fontsize=12, fontweight='bold')
    ax3.set_xlabel(labels['x_label'])
    ax3.set_ylabel(labels['y_label'])
    ax3.set_aspect('equal')
    _apply_plan_view_spatial_ticks(ax3, x_model, y_model)
    _publication_colorbar(pc3, ax=ax3, label=labels['value_label'], extend=prediction_extend, extendrect=True)

    ax4 = fig.add_subplot(2, 4, 4)
    pc4 = ax4.pcolormesh(
        x_model,
        z_coords,
        slice_xz.T,
        cmap=_resolve_plot_cmap('jet'),
        shading='auto',
        norm=prediction_norm,
    )
    ax4.invert_yaxis()
    ax4.set_title(f"{labels['xz_title']} (y={y_center:.0f})", fontsize=12, fontweight='bold')
    ax4.set_xlabel(labels['x_label'])
    ax4.set_ylabel(labels['z_label'])
    _apply_section_horizontal_spatial_ticks(ax4, x_model)
    _publication_colorbar(pc4, ax=ax4, label=labels['value_label'], extend=prediction_extend, extendrect=True)

    ax5 = fig.add_subplot(2, 4, 5)
    pc5 = ax5.pcolormesh(
        y_model,
        z_coords,
        slice_yz.T,
        cmap=_resolve_plot_cmap('jet'),
        shading='auto',
        norm=prediction_norm,
    )
    ax5.invert_yaxis()
    ax5.set_title(f"{labels['yz_title']} (x={x_center:.0f})", fontsize=12, fontweight='bold')
    ax5.set_xlabel(labels['y_label'])
    ax5.set_ylabel(labels['z_label'])
    _apply_section_horizontal_spatial_ticks(ax5, y_model)
    _publication_colorbar(pc5, ax=ax5, label=labels['value_label'], extend=prediction_extend, extendrect=True)

    ax6 = fig.add_subplot(2, 4, 6)
    pc6 = ax6.pcolormesh(
        x_model,
        y_model,
        source_projection_xy.T,
        cmap=_resolve_plot_cmap('magma'),
        shading='auto',
        norm=source_response_norm,
    )
    _plot_bbox_collection(ax6, surface_prior_bboxes, plane='xy', edgecolor='lime', linestyle=':', linewidth=2.0)
    _plot_bbox_collection(ax6, envelope_bboxes, plane='xy', edgecolor='cyan', linestyle='--', linewidth=2.0)
    _plot_bbox_collection(ax6, core_bboxes, plane='xy', edgecolor='white', linestyle='-', linewidth=2.0)
    ax6.set_title(labels['source_xy_title'], fontsize=12, fontweight='bold')
    ax6.set_xlabel(labels['x_label'])
    ax6.set_ylabel(labels['y_label'])
    ax6.set_aspect('equal')
    _apply_plan_view_spatial_ticks(ax6, x_model, y_model)
    _publication_colorbar(pc6, ax=ax6, label=labels['response_label'], extend=source_response_extend, extendrect=True)

    ax7 = fig.add_subplot(2, 4, 7)
    pc7 = ax7.pcolormesh(
        x_model,
        z_coords,
        source_extent_xz_display.T,
        cmap=_resolve_plot_cmap('magma'),
        shading='auto',
        norm=source_response_norm,
    )
    _plot_bbox_collection(ax7, envelope_bboxes, plane='xz', edgecolor='cyan', linestyle='--', linewidth=2.0)
    _plot_bbox_collection(ax7, core_bboxes, plane='xz', edgecolor='white', linestyle='-', linewidth=2.0)
    ax7.invert_yaxis()
    ax7.set_title(labels['source_xz_title'], fontsize=12, fontweight='bold')
    ax7.set_xlabel(labels['x_label'])
    ax7.set_ylabel(labels['z_label'])

    ax8 = fig.add_subplot(2, 4, 8)
    ax8.plot(z_coords, source_depth_profile, color='tab:red', linewidth=2, label=labels['response_label'])
    for center_idx, center in enumerate(estimated_centers):
        ax8.axvline(
            center[2],
            color='black' if center_idx == 0 else 'gray',
            linestyle='--' if center_idx == 0 else ':',
            linewidth=1.5 if center_idx == 0 else 1.0
        )
    _plot_depth_spans(ax8, envelope_bboxes, color='cyan', alpha=0.12, label=labels['envelope_label'])
    _plot_depth_spans(ax8, core_bboxes, color='white', alpha=0.12, label=labels['core_label'])
    ax8.set_title(labels['depth_profile_title'], fontsize=12, fontweight='bold')
    ax8.set_xlabel(labels['z_label'])
    ax8.set_ylabel(labels['response_label'])
    ax8.grid(True, alpha=0.3)
    ax8.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig('real_data_inference_visualization.png', dpi=150, bbox_inches='tight')
    derivative_fields = _compute_volume_derivative_fields(y_pred, dx=dx, dy=dy, dz=50.0)
    np.savez(
        'real_data_inference_outputs.npz',
        primary_input=surface_channels[primary_channel_name],
        primary_input_name=primary_channel_name,
        gz_input_raw=(
            gz_model.astype(np.float32)
            if gz_model is not None else np.array([], dtype=np.float32)
        ),
        gz_input=(
            surface_channels.get('gz', gz_model_centered)
            if gz_model_centered is not None else np.array([], dtype=np.float32)
        ),
        aux_input=aux_surface,
        input_channel_names=np.array(required_channels, dtype=object),
        aux_input_name=aux_channel_name,
        prediction=y_pred,
        prediction_raw=raw_y_pred,
        prediction_asa3d=derivative_fields['asa3d'],
        prediction_thdr=derivative_fields['thdr'],
        prediction_source_response=source_response,
        prediction_source_response_raw=source_response_raw,
        prediction_source_gradient=source_fields['gradient_response'],
        prediction_source_envelope=source_fields['envelope_response'],
        prediction_body_mask=pred_body_mask_np if pred_body_mask_np is not None else np.array([], dtype=np.float32),
        prediction_center_heatmap=pred_center_heatmap_np if pred_center_heatmap_np is not None else np.array([], dtype=np.float32),
        real_input_unit_scales=np.array([info['applied_scale'] for info in unit_conversions], dtype=np.float64),
        real_input_unit_channels=np.array([info['channel'] for info in unit_conversions], dtype=object),
        surface_calibration_applied=np.array(int(surface_calibration_diagnostics.get('applied', False)), dtype=np.int8),
        surface_calibration_gz_weight=np.array(surface_calibration_gz_weight, dtype=np.float64),
        surface_calibration_gzz_weight=np.array(surface_calibration_gzz_weight, dtype=np.float64),
        surface_calibration_surface_gz_corr_before=np.array(surface_calibration_diagnostics.get('surface_gz_before', {}).get('corr', np.nan), dtype=np.float64),
        surface_calibration_surface_gz_corr_after=np.array(surface_calibration_diagnostics.get('surface_gz_after', {}).get('corr', np.nan), dtype=np.float64),
        surface_calibration_surface_gzz_corr_before=np.array(surface_calibration_diagnostics.get('surface_gzz_before', {}).get('corr', np.nan), dtype=np.float64),
        surface_calibration_surface_gzz_corr_after=np.array(surface_calibration_diagnostics.get('surface_gzz_after', {}).get('corr', np.nan), dtype=np.float64),
        prediction_surface_prior_response=surface_focus['response'],
        prediction_surface_prior_weight=surface_prior_weight,
        surface_prior_center=np.array(surface_prior_center, dtype=np.float64),
        surface_prior_centers=surface_prior_centers,
        surface_prior_bbox=(
            np.array(
                [
                    surface_prior_bbox['x_min'],
                    surface_prior_bbox['x_max'],
                    surface_prior_bbox['y_min'],
                    surface_prior_bbox['y_max'],
                ],
                dtype=np.float64
            )
            if surface_prior_bbox is not None else np.array([], dtype=np.float64)
        ),
        surface_prior_bboxes=_bbox_list_to_array(surface_prior_bboxes, axis_names=('x', 'y')),
        body_mask_used=np.array(int(aux_diagnostics['body_mask_used']), dtype=np.int8),
        center_heatmap_used=np.array(int(aux_diagnostics['center_heatmap_used']), dtype=np.int8),
        field_center=np.array([field_x_center, field_y_center, field_z_center], dtype=np.float64),
        field_center_raw=raw_field_center,
        estimated_body_center=np.array([x_center, y_center, z_center], dtype=np.float64),
        estimated_body_centers=estimated_centers,
        estimated_body_core_bbox=_bbox_to_array(core_bbox),
        estimated_body_core_bboxes=_bbox_list_to_array(core_bboxes),
        estimated_body_envelope_bbox=_bbox_to_array(envelope_bbox),
        estimated_body_envelope_bboxes=_bbox_list_to_array(envelope_bboxes),
        estimated_body_component_source=np.array(component_source),
        source_core_level=np.array(core_level if core_level is not None else np.nan, dtype=np.float64),
        source_envelope_level=np.array(envelope_level if envelope_level is not None else np.nan, dtype=np.float64),
        x_coords=x_model,
        y_coords=y_model,
        z_coords=z_coords,
    )
    print(labels['saved'])
    print("已保存推断体数据到 real_data_inference_outputs.npz")
    plt.show()

    _save_derivative_visualization(
        y_pred,
        output_png_path='real_data_derivative_isosurfaces.png',
        output_npz_path='real_data_derivative_fields.npz',
        language=language,
        dx=dx,
        dy=dy,
        dz=50.0,
        x_coords=x_model,
        y_coords=y_model,
        z_coords=z_coords,
    )


def test_synthetic_dual_boxes_realstyle(
    checkpoint_path=None,
    model_capacity=None,
    input_mode=None,
    noise_level=0.0,
    add_regional=False
):
    """Test a deterministic dual-box synthetic case using the same visualization style as real data."""
    language = _configure_plot_language()
    labels = {
        'zh': {
            'title': "双长方体合成测试与真实数据风格可视化",
            'input_title': '合成输入 (主通道)',
            'xy_title': '预测 XY 切面',
            'xz_title': '预测 XZ 切面',
            'yz_title': '预测 YZ 切面',
            'source_xy_title': '异常体响应投影 (XY)',
            'source_xz_title': '异常体范围估计 (XZ)',
            'depth_profile_title': '异常体响应深度曲线',
            'x_label': 'X 坐标',
            'y_label': 'Y 坐标',
            'z_label': '深度',
            'value_label': '场值',
            'response_label': '异常体响应',
            'core_label': '核心范围',
            'envelope_label': '外围范围',
        },
        'en': {
            'title': "Synthetic Dual-Box Test With Real-Style Visualization",
            'input_title': 'Synthetic Input (Primary Channel)',
            'xy_title': 'Predicted XY Slice',
            'xz_title': 'Predicted XZ Slice',
            'yz_title': 'Predicted YZ Slice',
            'source_xy_title': 'Anomaly-Body Response Projection (XY)',
            'source_xz_title': 'Estimated Body Extent (XZ)',
            'depth_profile_title': 'Body-Response Depth Profile',
            'x_label': 'X Coordinate',
            'y_label': 'Y Coordinate',
            'z_label': 'Depth',
            'value_label': 'Field Value',
            'response_label': 'Body Response',
            'core_label': 'Core Extent',
            'envelope_label': 'Envelope Extent',
        },
    }[language]

    if checkpoint_path is None:
        default_capacity = model_capacity or 'medium'
        default_input_mode = input_mode or 'gzz'
        checkpoint_path = _get_default_checkpoint_path(default_capacity, default_input_mode)

    print("\n" + "=" * 60)
    print(labels['title'])
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"加载模型权重: {checkpoint_path}")

    model, checkpoint, resolved_capacity, resolved_input_mode, resolved_use_multimodal_stem, has_aux_heads = _load_model_from_checkpoint(
        checkpoint_path,
        device,
        model_capacity=model_capacity,
        input_mode=input_mode
    )
    print(f"模型容量配置: {resolved_capacity}")
    print(f"输入模式: {resolved_input_mode}")
    print(f"多模态输入干路: {'开启' if resolved_use_multimodal_stem else '关闭'}")
    print(f"异常体辅助头: {'可用' if has_aux_heads else '检查点不包含，回退到场响应后处理'}")
    print(f"已加载 epoch {checkpoint['epoch']} 的模型")

    generator = GravityDataGenerator()
    surface_channels_full, Y_true, anomaly_info = _generate_dual_boxes_case(
        generator,
        noise_level=noise_level,
        add_regional=add_regional
    )
    required_channels = get_input_channel_names(resolved_input_mode)
    surface_channels = {name: surface_channels_full[name] for name in required_channels}

    print("合成测试体: 正负密度双长方体")
    for idx, info in enumerate(anomaly_info, start=1):
        print(
            f"  Box {idx}: type={info['type']}, density={info['density']:.1f}, "
            f"center=({info['center'][0]:.1f}, {info['center'][1]:.1f}, {info['center'][2]:.1f}), "
            f"half_size=({info['size_xyz'][0]:.1f}, {info['size_xyz'][1]:.1f}, {info['size_xyz'][2]:.1f})"
        )

    primary_channel_name = _select_primary_channel(resolved_input_mode)
    aux_channel_name = _select_aux_channel(resolved_input_mode)
    aux_surface = surface_channels[aux_channel_name]
    aux_title, _, aux_colorbar = _get_aux_channel_labels(language, aux_channel_name)

    x_model = generator.x.astype(np.float64)
    y_model = generator.y.astype(np.float64)
    z_coords = generator.z.astype(np.float64)
    dx = float(generator.dx)
    dy = float(generator.dy)
    dz = float(generator.dz)

    input_array, target_scale, normalization_mode = _build_inference_input(
        surface_channels,
        input_mode=resolved_input_mode,
        checkpoint=checkpoint,
        label_3d=Y_true
    )
    print(f"输入归一化: {normalization_mode}")
    print(f"输入网格: 原始 {surface_channels[primary_channel_name].shape[0]}x{surface_channels[primary_channel_name].shape[1]} -> 模型 64x64")

    model.eval()
    with torch.no_grad():
        input_tensor = torch.from_numpy(input_array[np.newaxis, :, :, :]).float().to(device)
        output_dict = model(input_tensor)
        y_pred_norm = output_dict['output'].squeeze().cpu().numpy()
        pred_body_mask = output_dict.get('body_mask')
        pred_center_heatmap = output_dict.get('center_heatmap')

    pred_body_mask_np = None
    pred_center_heatmap_np = None
    if has_aux_heads and pred_body_mask is not None:
        pred_body_mask_np = pred_body_mask.squeeze().cpu().numpy().astype(np.float32)
    if has_aux_heads and pred_center_heatmap is not None:
        pred_center_heatmap_np = pred_center_heatmap.squeeze().cpu().numpy().astype(np.float32)

    y_pred = y_pred_norm * target_scale
    pred_mass = np.abs(y_pred) + 1e-12
    edge_ignore = 4
    min_depth_idx = int(np.searchsorted(z_coords, 250.0, side='left'))

    focus_surface_name = 'gzz' if 'gzz' in surface_channels else primary_channel_name
    surface_focus = _compute_surface_focus_prior(
        surface_channels[focus_surface_name],
        x_model,
        y_model,
        edge_ignore=edge_ignore
    )
    surface_prior_weight = surface_focus['weight']
    surface_prior_bbox = surface_focus['bbox']
    surface_prior_center = surface_focus['center']
    surface_prior_bboxes = list(surface_focus.get('bboxes', []))
    surface_prior_centers = np.asarray(
        surface_focus.get('centers', np.array([surface_prior_center], dtype=np.float64)),
        dtype=np.float64
    )
    if surface_prior_centers.size == 0:
        surface_prior_centers = np.asarray([surface_prior_center], dtype=np.float64)

    field_depth_energy = pred_mass.sum(axis=(0, 1))
    field_x_energy = pred_mass.sum(axis=(1, 2))
    field_y_energy = pred_mass.sum(axis=(0, 2))
    field_x_idx = int(np.argmax(field_x_energy))
    field_y_idx = int(np.argmax(field_y_energy))
    field_z_idx = int(np.argmax(field_depth_energy))
    raw_field_center = np.array(
        [float(x_model[field_x_idx]), float(y_model[field_y_idx]), float(z_coords[field_z_idx])],
        dtype=np.float64
    )

    trusted_field_mass = _zero_boundary_shell(
        pred_mass * surface_prior_weight[..., None],
        shell_width=edge_ignore,
        fill_value=0.0
    )
    if 0 < min_depth_idx < trusted_field_mass.shape[2]:
        trusted_field_mass[:, :, :min_depth_idx] = 0.0
    field_center = _compute_weighted_volume_center(trusted_field_mass, x_model, y_model, z_coords)
    if field_center is None:
        field_center = tuple(raw_field_center.tolist())
    field_x_center, field_y_center, field_z_center = field_center

    source_fields = _compute_source_localization_response(y_pred, dx=dx, dy=dy, dz=dz, edge_ignore=edge_ignore)
    source_response_raw, aux_diagnostics = _combine_body_support_response(
        source_fields['source_response'],
        pred_body_mask=pred_body_mask_np,
        pred_center_heatmap=pred_center_heatmap_np,
        edge_ignore=edge_ignore
    )
    source_response = _build_trusted_body_response(
        source_response_raw,
        surface_prior_weight,
        edge_ignore=edge_ignore,
        min_depth_idx=min_depth_idx
    )
    source_projection_xy = np.max(source_response, axis=2)
    source_depth_profile = source_response.sum(axis=(0, 1))
    source_x_energy = source_response.sum(axis=(1, 2))
    source_y_energy = source_response.sum(axis=(0, 2))

    body_estimate = _estimate_body_components(
        source_response,
        pred_body_mask_np,
        aux_diagnostics,
        x_model,
        y_model,
        z_coords,
        edge_ignore=edge_ignore,
        min_depth_idx=min_depth_idx,
    )
    component_source = body_estimate['component_source']
    core_level = body_estimate['core_level']
    envelope_level = body_estimate['envelope_level']
    core_components = body_estimate['core_components']
    envelope_components = body_estimate['envelope_components']
    core_bboxes = [component['bbox'] for component in core_components]
    envelope_bboxes = [component['bbox'] for component in envelope_components]
    core_bbox = body_estimate['dominant_core_bbox']
    envelope_bbox = body_estimate['dominant_envelope_bbox']
    estimated_centers = np.asarray(body_estimate['centers'], dtype=np.float64)

    estimated_center = body_estimate['dominant_center']
    if estimated_center is None:
        x_idx = int(np.argmax(source_x_energy))
        y_idx = int(np.argmax(source_y_energy))
        z_idx = int(np.argmax(source_depth_profile))
        x_center = float(x_model[x_idx])
        y_center = float(y_model[y_idx])
        z_center = float(z_coords[z_idx])
        estimated_centers = np.asarray([[x_center, y_center, z_center]], dtype=np.float64)
    else:
        x_center, y_center, z_center = estimated_center
        x_idx = int(np.argmin(np.abs(x_model - x_center)))
        y_idx = int(np.argmin(np.abs(y_model - y_center)))
        z_idx = int(np.argmin(np.abs(z_coords - z_center)))
        if estimated_centers.size == 0:
            estimated_centers = np.asarray([[x_center, y_center, z_center]], dtype=np.float64)

    print(
        f"地表异常先验中心: x={surface_prior_center[0]:.1f}, y={surface_prior_center[1]:.1f} "
        f"(source={focus_surface_name})"
    )
    print(
        f"辅助体响应融合: body_mask={'启用' if aux_diagnostics['body_mask_used'] else '禁用'}, "
        f"center_heatmap={'启用' if aux_diagnostics['center_heatmap_used'] else '禁用'}"
    )
    body_peak = aux_diagnostics['body_mask_stats']['peak'] if aux_diagnostics['body_mask_stats'] is not None else 0.0
    center_peak = aux_diagnostics['center_heatmap_stats']['peak'] if aux_diagnostics['center_heatmap_stats'] is not None else 0.0
    print(f"body_mask峰值={body_peak:.3e}, center峰值={center_peak:.3e}")
    print(f"多异常体估计源: {component_source}, envelope_count={len(envelope_components)}, core_count={len(core_components)}")
    print(f"3D场可信中心: x={field_x_center:.1f}, y={field_y_center:.1f}, z={field_z_center:.1f}")
    print(f"异常体估计中心: x={x_center:.1f}, y={y_center:.1f}, z={z_center:.1f}")
    print(f"{labels['core_label']}: {_format_bbox_collection(core_bboxes)}")
    print(f"{labels['envelope_label']}: {_format_bbox_collection(envelope_bboxes)}")
    print(f"预测场范围: [{y_pred.min():.3e}, {y_pred.max():.3e}]")

    slice_xy = y_pred[:, :, z_idx]
    slice_xz = y_pred[:, y_idx, :]
    slice_yz = y_pred[x_idx, :, :]
    source_projection_xz = np.max(source_response, axis=1)
    source_extent_xz_display = _build_body_extent_projection_xz(source_response, envelope_level)
    prediction_norm, prediction_extend = _build_main_visualization_norm(y_pred)
    source_response_norm, source_response_extend, source_response_levels = _build_body_response_projection_norm(source_response)

    fig = plt.figure(figsize=(20, 10))

    ax1 = fig.add_subplot(2, 4, 1)
    pc1 = ax1.pcolormesh(
        x_model,
        y_model,
        surface_channels[primary_channel_name].T,
        cmap=_resolve_input_plot_cmap(primary_channel_name),
        shading='auto',
    )
    ax1.set_title(labels['input_title'], fontsize=12, fontweight='bold')
    ax1.set_xlabel(labels['x_label'])
    ax1.set_ylabel(labels['y_label'])
    ax1.set_aspect('equal')
    _plot_bbox_collection(ax1, surface_prior_bboxes, plane='xy', edgecolor='lime', linestyle=':', linewidth=2.0)
    _publication_colorbar(pc1, ax=ax1, label=labels['value_label'])

    ax2 = fig.add_subplot(2, 4, 2)
    pc2 = ax2.pcolormesh(
        x_model,
        y_model,
        aux_surface.T,
        cmap=_resolve_input_plot_cmap(aux_channel_name),
        shading='auto',
    )
    ax2.set_title(aux_title, fontsize=12, fontweight='bold')
    ax2.set_xlabel(labels['x_label'])
    ax2.set_ylabel(labels['y_label'])
    ax2.set_aspect('equal')
    _plot_bbox_collection(ax2, surface_prior_bboxes, plane='xy', edgecolor='lime', linestyle=':', linewidth=2.0)
    _publication_colorbar(pc2, ax=ax2, label=aux_colorbar)

    ax3 = fig.add_subplot(2, 4, 3)
    pc3 = ax3.pcolormesh(
        x_model,
        y_model,
        slice_xy.T,
        cmap=_resolve_plot_cmap('jet'),
        shading='auto',
        norm=prediction_norm,
    )
    ax3.set_title(f"{labels['xy_title']} (z={z_center:.0f})", fontsize=12, fontweight='bold')
    ax3.set_xlabel(labels['x_label'])
    ax3.set_ylabel(labels['y_label'])
    ax3.set_aspect('equal')
    _publication_colorbar(pc3, ax=ax3, label=labels['value_label'], extend=prediction_extend, extendrect=True)

    ax4 = fig.add_subplot(2, 4, 4)
    pc4 = ax4.pcolormesh(
        x_model,
        z_coords,
        slice_xz.T,
        cmap=_resolve_plot_cmap('jet'),
        shading='auto',
        norm=prediction_norm,
    )
    ax4.invert_yaxis()
    ax4.set_title(f"{labels['xz_title']} (y={y_center:.0f})", fontsize=12, fontweight='bold')
    ax4.set_xlabel(labels['x_label'])
    ax4.set_ylabel(labels['z_label'])
    _publication_colorbar(pc4, ax=ax4, label=labels['value_label'], extend=prediction_extend, extendrect=True)

    ax5 = fig.add_subplot(2, 4, 5)
    pc5 = ax5.pcolormesh(
        y_model,
        z_coords,
        slice_yz.T,
        cmap=_resolve_plot_cmap('jet'),
        shading='auto',
        norm=prediction_norm,
    )
    ax5.invert_yaxis()
    ax5.set_title(f"{labels['yz_title']} (x={x_center:.0f})", fontsize=12, fontweight='bold')
    ax5.set_xlabel(labels['y_label'])
    ax5.set_ylabel(labels['z_label'])
    _publication_colorbar(pc5, ax=ax5, label=labels['value_label'], extend=prediction_extend, extendrect=True)

    ax6 = fig.add_subplot(2, 4, 6)
    pc6 = ax6.pcolormesh(
        x_model,
        y_model,
        source_projection_xy.T,
        cmap=_resolve_plot_cmap('magma'),
        shading='auto',
        norm=source_response_norm,
    )
    _plot_bbox_collection(ax6, surface_prior_bboxes, plane='xy', edgecolor='lime', linestyle=':', linewidth=2.0)
    _plot_bbox_collection(ax6, envelope_bboxes, plane='xy', edgecolor='cyan', linestyle='--', linewidth=2.0)
    _plot_bbox_collection(ax6, core_bboxes, plane='xy', edgecolor='white', linestyle='-', linewidth=2.0)
    ax6.set_title(labels['source_xy_title'], fontsize=12, fontweight='bold')
    ax6.set_xlabel(labels['x_label'])
    ax6.set_ylabel(labels['y_label'])
    ax6.set_aspect('equal')
    _publication_colorbar(pc6, ax=ax6, label=labels['response_label'], extend=source_response_extend, extendrect=True)

    ax7 = fig.add_subplot(2, 4, 7)
    pc7 = ax7.pcolormesh(
        x_model,
        z_coords,
        source_extent_xz_display.T,
        cmap=_resolve_plot_cmap('magma'),
        shading='auto',
        norm=source_response_norm,
    )
    _plot_bbox_collection(ax7, envelope_bboxes, plane='xz', edgecolor='cyan', linestyle='--', linewidth=2.0)
    _plot_bbox_collection(ax7, core_bboxes, plane='xz', edgecolor='white', linestyle='-', linewidth=2.0)
    ax7.invert_yaxis()
    ax7.set_title(labels['source_xz_title'], fontsize=12, fontweight='bold')
    ax7.set_xlabel(labels['x_label'])
    ax7.set_ylabel(labels['z_label'])

    ax8 = fig.add_subplot(2, 4, 8)
    ax8.plot(z_coords, source_depth_profile, color='tab:red', linewidth=2, label=labels['response_label'])
    for center_idx, center in enumerate(estimated_centers):
        ax8.axvline(
            center[2],
            color='black' if center_idx == 0 else 'gray',
            linestyle='--' if center_idx == 0 else ':',
            linewidth=1.5 if center_idx == 0 else 1.0
        )
    _plot_depth_spans(ax8, envelope_bboxes, color='cyan', alpha=0.12, label=labels['envelope_label'])
    _plot_depth_spans(ax8, core_bboxes, color='white', alpha=0.12, label=labels['core_label'])
    ax8.set_title(labels['depth_profile_title'], fontsize=12, fontweight='bold')
    ax8.set_xlabel(labels['z_label'])
    ax8.set_ylabel(labels['response_label'])
    ax8.grid(True, alpha=0.3)
    ax8.legend(loc='upper right', fontsize=8)

    output_prefix = 'test_dual_boxes_realstyle'
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_inference_visualization.png', dpi=150, bbox_inches='tight')
    derivative_fields = _compute_volume_derivative_fields(y_pred, dx=dx, dy=dy, dz=dz)
    np.savez(
        f'{output_prefix}_inference_outputs.npz',
        primary_input=surface_channels[primary_channel_name],
        primary_input_name=primary_channel_name,
        gz_input=surface_channels.get('gz', np.array([], dtype=np.float32)),
        aux_input=aux_surface,
        aux_input_name=aux_channel_name,
        input_channel_names=np.array(required_channels, dtype=object),
        target=Y_true,
        prediction=y_pred,
        prediction_asa3d=derivative_fields['asa3d'],
        prediction_thdr=derivative_fields['thdr'],
        prediction_source_response=source_response,
        prediction_source_response_raw=source_response_raw,
        prediction_source_gradient=source_fields['gradient_response'],
        prediction_source_envelope=source_fields['envelope_response'],
        prediction_body_mask=pred_body_mask_np if pred_body_mask_np is not None else np.array([], dtype=np.float32),
        prediction_center_heatmap=pred_center_heatmap_np if pred_center_heatmap_np is not None else np.array([], dtype=np.float32),
        surface_prior_center=np.array(surface_prior_center, dtype=np.float64),
        surface_prior_centers=surface_prior_centers,
        surface_prior_bboxes=_bbox_list_to_array(surface_prior_bboxes, axis_names=('x', 'y')),
        field_center=np.array([field_x_center, field_y_center, field_z_center], dtype=np.float64),
        estimated_body_center=np.array([x_center, y_center, z_center], dtype=np.float64),
        estimated_body_centers=estimated_centers,
        estimated_body_core_bbox=_bbox_to_array(core_bbox),
        estimated_body_core_bboxes=_bbox_list_to_array(core_bboxes),
        estimated_body_envelope_bbox=_bbox_to_array(envelope_bbox),
        estimated_body_envelope_bboxes=_bbox_list_to_array(envelope_bboxes),
        estimated_body_component_source=np.array(component_source),
        true_body_centers=np.array([info['center'] for info in anomaly_info], dtype=np.float64),
        true_body_half_sizes=np.array([info['size_xyz'] for info in anomaly_info], dtype=np.float64),
        true_body_densities=np.array([info['density'] for info in anomaly_info], dtype=np.float64),
        x_coords=x_model,
        y_coords=y_model,
        z_coords=z_coords,
    )
    print(f"可视化已保存为 {output_prefix}_inference_visualization.png" if language == 'zh' else f"Saved visualization to {output_prefix}_inference_visualization.png")
    print(f"已保存推断体数据到 {output_prefix}_inference_outputs.npz" if language == 'zh' else f"Saved inferred volumes to {output_prefix}_inference_outputs.npz")
    plt.show()

    _save_derivative_visualization(
        y_pred,
        output_png_path=f'{output_prefix}_derivative_isosurfaces.png',
        output_npz_path=f'{output_prefix}_derivative_fields.npz',
        language=language,
        dx=dx,
        dy=dy,
        dz=dz,
        x_coords=x_model,
        y_coords=y_model,
        z_coords=z_coords,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run synthetic or real-data inference with the trained model.")
    parser.add_argument('--real-data', type=str, default=None, help='Path to a real x,y,value text grid.')
    parser.add_argument('--real-gz', type=str, default=None, help='Path to the real gz surface grid.')
    parser.add_argument('--real-gzz', type=str, default=None, help='Path to the real gzz surface grid.')
    parser.add_argument('--checkpoint-path', type=str, default=None, help='Checkpoint path to load.')
    parser.add_argument('--model-capacity', type=str, default=None, help='Model capacity preset, e.g. small or medium.')
    parser.add_argument('--input-mode', type=str, default=None, help='Surface input mode, e.g. gz_gzz or gz_amp.')
    parser.add_argument('--surface-calibration', action='store_true', help='Apply optional post-inference calibration on the top surface layers.')
    parser.add_argument('--surface-calibration-gz-weight', type=float, default=5.0, help='Observed gz weight used by post-inference surface calibration.')
    parser.add_argument('--surface-calibration-gzz-weight', type=float, default=5.0, help='Observed gzz weight used by post-inference surface calibration.')
    parser.add_argument('--noise-level', type=float, default=0.03, help='Synthetic test noise level.')
    parser.add_argument('--add-regional', action='store_true', help='Add regional field for synthetic testing.')
    parser.add_argument(
        '--synthetic-case',
        type=str,
        default='random',
        choices=['random', 'center_box', 'deep_box', 'dual_boxes', 'complex_geology', 'single_tilted_bar', 'wide_dual_tilted_bars'],
        help='Synthetic test case.',
    )
    args = parser.parse_args()

    if args.real_data or args.real_gz or args.real_gzz:
        test_real_data_model(
            real_data_path=args.real_data,
            real_gz_path=args.real_gz,
            real_gzz_path=args.real_gzz,
            checkpoint_path=args.checkpoint_path,
            model_capacity=args.model_capacity,
            input_mode=args.input_mode,
            surface_calibration=args.surface_calibration,
            surface_calibration_gz_weight=args.surface_calibration_gz_weight,
            surface_calibration_gzz_weight=args.surface_calibration_gzz_weight,
        )
    elif args.synthetic_case == 'dual_boxes':
        test_synthetic_dual_boxes_realstyle(
            checkpoint_path=args.checkpoint_path,
            model_capacity=args.model_capacity,
            input_mode=args.input_mode,
            noise_level=args.noise_level,
            add_regional=args.add_regional,
        )
    else:
        test_trained_model(
            checkpoint_path=args.checkpoint_path,
            noise_level=args.noise_level,
            add_regional=args.add_regional,
            model_capacity=args.model_capacity,
            input_mode=args.input_mode,
            synthetic_case=args.synthetic_case
        )
