"""
On-the-fly synthetic-data generator for PyTorch DataLoader.

The generator uses 3D FFT gravity forward modeling so training samples can be
created during training without storing a large synthetic data set on disk.
"""

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from scipy.fft import fftn, ifftn, fftfreq
from scipy.ndimage import gaussian_filter


INPUT_CHANNEL_MODES = {
    'gzz': ('gzz',),
    'gz_amp': ('gz', 'amp'),
    'gz_gzz': ('gz', 'gzz'),
    'gz_gzz_amp': ('gz', 'gzz', 'amp'),
}


# Fixed physical normalization scales keep absolute amplitude information.
# The values are chosen to cover the current synthetic generator's typical range
# without adapting per sample, so amplitude still carries depth/mass cues.
PHYSICAL_NORMALIZATION_SCALES = {
    'gz': 2000.0,
    'gzz': 0.03,
    'amp': 0.03,
    'target': 15.0,
}


def get_input_channel_names(input_mode='gz_amp'):
    """Return the ordered channel names used for a given surface-input mode."""
    if input_mode not in INPUT_CHANNEL_MODES:
        raise ValueError(
            f"Unsupported input_mode '{input_mode}'. Available options: {sorted(INPUT_CHANNEL_MODES)}"
        )
    return INPUT_CHANNEL_MODES[input_mode]


def get_target_scale(surface_channels, label_3d=None):
    """Return the fixed output normalization scale used by the model."""
    return PHYSICAL_NORMALIZATION_SCALES['target']


def build_model_input(surface_channels, input_mode='gz_amp', target_scale=None):
    """Compose inputs using fixed physical scales instead of per-sample max scaling."""
    channel_names = get_input_channel_names(input_mode)
    if target_scale is None:
        target_scale = get_target_scale(surface_channels)

    normalized_channels = []
    channel_scales = {}

    for channel_name in channel_names:
        channel_data = surface_channels[channel_name]
        scale = PHYSICAL_NORMALIZATION_SCALES[channel_name]
        channel_scales[channel_name] = scale
        normalized_channels.append((channel_data / scale).astype(np.float32))

    return np.stack(normalized_channels, axis=0), channel_scales, target_scale


class GravityDataGenerator:
    """Synthetic gravity-anomaly generation engine."""

    def __init__(
        self,
        nx=64,
        ny=64,
        nz=64,
        dx=50,
        dy=50,
        dz=50,
        depth_sampling_mode='shallow_biased',
        structured_case_probability=0.0,
        tilted_pair_probability=0.7,
        wide_pair_probability=0.75,
    ):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.depth_sampling_mode = depth_sampling_mode
        self.structured_case_probability = float(np.clip(structured_case_probability, 0.0, 1.0))
        self.tilted_pair_probability = float(np.clip(tilted_pair_probability, 0.0, 1.0))
        self.wide_pair_probability = float(np.clip(wide_pair_probability, 0.0, 1.0))
        self.G = 6.67430e-11  # Gravitational constant.
        self.x = np.arange(self.nx, dtype=np.float32) * self.dx
        self.y = np.arange(self.ny, dtype=np.float32) * self.dy
        self.z = np.arange(self.nz, dtype=np.float32) * self.dz
        self.X_3d, self.Y_3d, self.Z_3d = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        self.X_2d, self.Y_2d = np.meshgrid(self.x, self.y, indexing='ij')

        # Precompute the 3D wavenumber grid and transfer function once.
        self._precompute_transfer_function()

    def _precompute_transfer_function(self):
        """Precompute the 3D FFT transfer function."""
        kx = 2 * np.pi * fftfreq(self.nx, self.dx)
        ky = 2 * np.pi * fftfreq(self.ny, self.dy)
        kz = 2 * np.pi * fftfreq(self.nz, self.dz)
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        K2 = KX**2 + KY**2 + KZ**2
        K2[0, 0, 0] = 1e-10  # Avoid division by zero at the DC component.

        # Frequency-domain transfer function for the vertical gravity anomaly.
        self.H = (1j * 4 * np.pi * self.G * KZ) / K2
        self.H[0, 0, 0] = 0

    def _sample_body_count(self):
        return int(np.random.choice([1, 2, 3], p=[0.58, 0.29, 0.13]))

    def _sample_center(self):
        """Sample centers with mild shallow-depth bias for better real-data transfer."""
        if np.random.rand() < 0.25:
            x_idx = np.random.randint(6, self.nx - 6)
            y_idx = np.random.randint(6, self.ny - 6)
        else:
            x_idx = np.random.randint(12, self.nx - 12)
            y_idx = np.random.randint(12, self.ny - 12)

        z_idx = self._sample_depth_index()
        return x_idx * self.dx, y_idx * self.dy, z_idx * self.dz

    def _sample_depth_index(self):
        """Bias part of the synthetic set toward shallower targets without removing deeper cases."""
        min_idx = 5
        max_idx = self.nz - 8
        span = max(max_idx - min_idx, 1)

        if self.depth_sampling_mode == 'balanced':
            depth_fraction = np.random.rand()
        elif self.depth_sampling_mode == 'focused_shallow':
            if np.random.rand() < 0.85:
                depth_fraction = np.random.beta(1.2, 5.2)
            else:
                depth_fraction = np.random.beta(2.2, 2.4)
        elif self.depth_sampling_mode == 'shallow_biased':
            if np.random.rand() < 0.72:
                depth_fraction = np.random.beta(1.5, 4.2)
            else:
                depth_fraction = np.random.beta(2.2, 2.2)
        else:
            raise ValueError(
                f"Unsupported depth_sampling_mode '{self.depth_sampling_mode}'. "
                "Available options: ['balanced', 'shallow_biased', 'focused_shallow']"
            )

        z_idx = min_idx + int(round(depth_fraction * span))
        return int(np.clip(z_idx, min_idx, max_idx))

    def _sample_density(self):
        sign = -1.0 if np.random.rand() < 0.35 else 1.0
        magnitude = np.random.uniform(250.0, 1600.0)
        return sign * magnitude

    def _random_rotation_matrix(self):
        yaw = np.random.uniform(0.0, np.pi)
        pitch = np.random.uniform(-np.pi / 8, np.pi / 8)
        roll = np.random.uniform(-np.pi / 10, np.pi / 10)

        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
        return rz @ ry @ rx

    def _rotated_coordinates(self, center, rotation):
        coords = np.stack(
            [
                self.X_3d - center[0],
                self.Y_3d - center[1],
                self.Z_3d - center[2],
            ],
            axis=-1
        )
        return coords @ rotation

    def _build_ellipsoid_mask(self, center, size_xyz, rotation):
        rotated = self._rotated_coordinates(center, rotation)
        rx, ry, rz = [max(float(value), 1.0) for value in size_xyz]
        normalized = (
            (rotated[..., 0] / rx) ** 2 +
            (rotated[..., 1] / ry) ** 2 +
            (rotated[..., 2] / rz) ** 2
        )
        return normalized <= 1.0

    def _build_box_mask(self, center, size_xyz, rotation):
        rotated = self._rotated_coordinates(center, rotation)
        sx, sy, sz = [max(float(value), 1.0) for value in size_xyz]
        return (
            (np.abs(rotated[..., 0]) <= sx) &
            (np.abs(rotated[..., 1]) <= sy) &
            (np.abs(rotated[..., 2]) <= sz)
        )

    def _build_cylinder_mask(self, center, radius_xy, half_height, rotation):
        rotated = self._rotated_coordinates(center, rotation)
        rx, ry = radius_xy
        radial = (rotated[..., 0] / max(rx, 1.0)) ** 2 + (rotated[..., 1] / max(ry, 1.0)) ** 2
        return (radial <= 1.0) & (np.abs(rotated[..., 2]) <= max(float(half_height), 1.0))

    def _build_blob_cluster_mask(self, center):
        composite = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
        num_blobs = np.random.randint(2, 5)

        for _ in range(num_blobs):
            offset = np.array([
                np.random.uniform(-180.0, 180.0),
                np.random.uniform(-180.0, 180.0),
                np.random.uniform(-140.0, 140.0),
            ], dtype=np.float32)
            blob_center = tuple(np.array(center, dtype=np.float32) + offset)
            size_xyz = (
                np.random.uniform(120.0, 260.0),
                np.random.uniform(120.0, 260.0),
                np.random.uniform(100.0, 240.0),
            )
            rotation = self._random_rotation_matrix()
            rotated = self._rotated_coordinates(blob_center, rotation)
            normalized = (
                (rotated[..., 0] / size_xyz[0]) ** 2 +
                (rotated[..., 1] / size_xyz[1]) ** 2 +
                (rotated[..., 2] / size_xyz[2]) ** 2
            )
            composite += np.exp(-1.4 * normalized).astype(np.float32)

        composite = gaussian_filter(composite, sigma=np.random.uniform(0.6, 1.4))
        threshold = np.random.uniform(0.48, 0.72)
        mask = composite >= threshold

        occupied = np.argwhere(mask)
        if occupied.size == 0:
            size_xyz = (180.0, 180.0, 160.0)
        else:
            mins = occupied.min(axis=0)
            maxs = occupied.max(axis=0)
            size_xyz = (
                max((maxs[0] - mins[0] + 1) * self.dx / 2.0, self.dx),
                max((maxs[1] - mins[1] + 1) * self.dy / 2.0, self.dy),
                max((maxs[2] - mins[2] + 1) * self.dz / 2.0, self.dz),
            )

        return mask, size_xyz

    def _summarize_mask_component(self, mask, body_type, density, size_xyz, rotation, center_hint):
        occupied = np.argwhere(mask)
        if occupied.size == 0:
            center = tuple(float(v) for v in center_hint)
            bbox = {
                'x_min': float(center_hint[0]),
                'x_max': float(center_hint[0]),
                'y_min': float(center_hint[1]),
                'y_max': float(center_hint[1]),
                'z_min': float(center_hint[2]),
                'z_max': float(center_hint[2]),
            }
        else:
            center = (
                float(np.mean(self.x[occupied[:, 0]])),
                float(np.mean(self.y[occupied[:, 1]])),
                float(np.mean(self.z[occupied[:, 2]])),
            )
            bbox = {
                'x_min': float(self.x[occupied[:, 0].min()]),
                'x_max': float(self.x[occupied[:, 0].max()]),
                'y_min': float(self.y[occupied[:, 1].min()]),
                'y_max': float(self.y[occupied[:, 1].max()]),
                'z_min': float(self.z[occupied[:, 2].min()]),
                'z_max': float(self.z[occupied[:, 2].max()]),
            }

        return {
            'type': body_type,
            'center': center,
            'size': float(max(size_xyz)),
            'size_xyz': tuple(float(value) for value in size_xyz),
            'density': float(density),
            'bbox': bbox,
            'rotation_matrix': np.asarray(rotation, dtype=np.float32).tolist(),
        }

    def _build_tilted_bar_component(self, center, radius_xy, half_height, rotation, density, body_type):
        raw_mask = self._build_cylinder_mask(center, radius_xy, half_height, rotation)
        smooth_sigma = np.random.uniform(0.55, 0.95)
        threshold = np.random.uniform(0.40, 0.48)
        mask = gaussian_filter(raw_mask.astype(np.float32), sigma=smooth_sigma) >= threshold
        size_xyz = (float(radius_xy[0]), float(radius_xy[1]), float(half_height))
        info = self._summarize_mask_component(mask, body_type, density, size_xyz, rotation, center)
        info['shape'] = 'tilted_bar'
        return mask, info

    def _generate_single_tilted_bar_body(self):
        cx = float(self.x[self.nx // 2] + np.random.uniform(-220.0, 220.0))
        cy = float(self.y[self.ny // 2] + np.random.uniform(-80.0, 80.0))
        cz = float(np.random.uniform(950.0, 1900.0))
        radius_xy = (
            np.random.uniform(70.0, 115.0),
            np.random.uniform(110.0, 160.0),
        )
        half_height = np.random.uniform(620.0, 920.0)
        pitch = np.random.uniform(22.0, 42.0) * np.random.choice([-1.0, 1.0])
        density = self._sample_density()
        if abs(density) < 500.0:
            density = np.sign(density) * np.random.uniform(650.0, 1350.0)
        if abs(density) < 1e-6:
            density = np.random.uniform(650.0, 1350.0)

        pitch_rad = np.deg2rad(pitch)
        rotation = np.array(
            [
                [np.cos(pitch_rad), 0.0, np.sin(pitch_rad)],
                [0.0, 1.0, 0.0],
                [-np.sin(pitch_rad), 0.0, np.cos(pitch_rad)],
            ],
            dtype=np.float32,
        )
        return self._build_tilted_bar_component(
            center=(cx, cy, cz),
            radius_xy=radius_xy,
            half_height=half_height,
            rotation=rotation,
            density=density,
            body_type='single_tilted_bar',
        )

    def _generate_tilted_pair_body(self, wide_separation=True):
        cx = float(self.x[self.nx // 2])
        cy = float(self.y[self.ny // 2] + np.random.uniform(-100.0, 100.0))
        center_separation = np.random.uniform(900.0, 1300.0) if wide_separation else np.random.uniform(520.0, 840.0)
        x_shift = center_separation / 2.0

        branch_specs = [
            {
                'type': 'positive_tilted_bar',
                'center': (cx - x_shift, cy + np.random.uniform(-60.0, 60.0), np.random.uniform(900.0, 1450.0)),
                'radius_xy': (np.random.uniform(72.0, 110.0), np.random.uniform(110.0, 160.0)),
                'half_height': np.random.uniform(640.0, 960.0),
                'pitch': np.random.uniform(24.0, 42.0),
                'density': np.random.uniform(780.0, 1380.0),
            },
            {
                'type': 'negative_tilted_bar',
                'center': (cx + x_shift, cy + np.random.uniform(-60.0, 60.0), np.random.uniform(1400.0, 2250.0)),
                'radius_xy': (np.random.uniform(72.0, 118.0), np.random.uniform(110.0, 160.0)),
                'half_height': np.random.uniform(620.0, 920.0),
                'pitch': -np.random.uniform(22.0, 40.0),
                'density': -np.random.uniform(760.0, 1320.0),
            },
        ]

        rho_model = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
        anomaly_info = []
        for spec in branch_specs:
            pitch_rad = np.deg2rad(spec['pitch'])
            rotation = np.array(
                [
                    [np.cos(pitch_rad), 0.0, np.sin(pitch_rad)],
                    [0.0, 1.0, 0.0],
                    [-np.sin(pitch_rad), 0.0, np.cos(pitch_rad)],
                ],
                dtype=np.float32,
            )
            mask, info = self._build_tilted_bar_component(
                center=spec['center'],
                radius_xy=spec['radius_xy'],
                half_height=spec['half_height'],
                rotation=rotation,
                density=spec['density'],
                body_type=spec['type'],
            )
            rho_model[mask] += float(spec['density'])
            anomaly_info.append(info)
        return rho_model, anomaly_info

    def _generate_structured_bar_case(self):
        if np.random.rand() < self.tilted_pair_probability:
            return self._generate_tilted_pair_body(wide_separation=np.random.rand() < self.wide_pair_probability)

        mask, info = self._generate_single_tilted_bar_body()
        rho_model = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
        rho_model[mask] += float(info['density'])
        return rho_model, [info]

    def _build_single_body(self):
        body_type = np.random.choice(
            ['sphere', 'ellipsoid', 'box', 'cylinder', 'blob_cluster'],
            p=[0.16, 0.32, 0.18, 0.16, 0.18]
        )
        center = self._sample_center()
        density = self._sample_density()
        rotation = self._random_rotation_matrix()
        component_rotation = rotation

        if body_type == 'sphere':
            radius = np.random.uniform(120.0, 360.0)
            size_xyz = (radius, radius, radius)
            component_rotation = np.eye(3, dtype=np.float32)
            mask = self._build_ellipsoid_mask(center, size_xyz, component_rotation)
        elif body_type == 'ellipsoid':
            size_xyz = (
                np.random.uniform(120.0, 360.0),
                np.random.uniform(120.0, 360.0),
                np.random.uniform(100.0, 320.0),
            )
            mask = self._build_ellipsoid_mask(center, size_xyz, component_rotation)
        elif body_type == 'box':
            size_xyz = (
                np.random.uniform(100.0, 300.0),
                np.random.uniform(100.0, 300.0),
                np.random.uniform(80.0, 260.0),
            )
            mask = self._build_box_mask(center, size_xyz, component_rotation)
        elif body_type == 'cylinder':
            radius_xy = (
                np.random.uniform(90.0, 220.0),
                np.random.uniform(90.0, 220.0),
            )
            half_height = np.random.uniform(140.0, 520.0)
            size_xyz = (radius_xy[0], radius_xy[1], half_height)
            mask = self._build_cylinder_mask(center, radius_xy, half_height, component_rotation)
        else:
            mask, size_xyz = self._build_blob_cluster_mask(center)
            component_rotation = np.eye(3, dtype=np.float32)

        return mask, {
            'type': body_type,
            'center': center,
            'size': float(max(size_xyz)),
            'size_xyz': tuple(float(value) for value in size_xyz),
            'density': float(density),
            'rotation_matrix': np.asarray(component_rotation, dtype=np.float32).tolist(),
        }

    def _build_location_targets(self, rho_model, anomaly_info):
        """Build localization-friendly supervision volumes from the density model."""
        positive_mask = (rho_model > 1e-6).astype(np.float32)
        negative_mask = (rho_model < -1e-6).astype(np.float32)
        location_mask = np.clip(positive_mask + negative_mask, 0.0, 1.0).astype(np.float32)
        soft_mask = np.clip(location_mask + 0.35 * gaussian_filter(location_mask, sigma=1.0), 0.0, 1.0)
        positive_soft_mask = np.clip(positive_mask + 0.30 * gaussian_filter(positive_mask, sigma=0.9), 0.0, 1.0)
        negative_soft_mask = np.clip(negative_mask + 0.30 * gaussian_filter(negative_mask, sigma=0.9), 0.0, 1.0)
        center_heatmap = np.zeros_like(location_mask, dtype=np.float32)
        axis_heatmap = np.zeros_like(location_mask, dtype=np.float32)

        for info in anomaly_info:
            cx, cy, cz = info['center']
            sx, sy, sz = info['size_xyz']
            sigma_x = max(0.45 * sx, 1.5 * self.dx)
            sigma_y = max(0.45 * sy, 1.5 * self.dy)
            sigma_z = max(0.45 * sz, 1.5 * self.dz)
            gaussian_blob = np.exp(
                -0.5 * (
                    ((self.X_3d - cx) / sigma_x) ** 2 +
                    ((self.Y_3d - cy) / sigma_y) ** 2 +
                    ((self.Z_3d - cz) / sigma_z) ** 2
                )
            ).astype(np.float32)
            center_heatmap = np.maximum(center_heatmap, gaussian_blob)

            rotation = np.asarray(info.get('rotation_matrix', np.eye(3, dtype=np.float32)), dtype=np.float32)
            if rotation.shape != (3, 3):
                rotation = np.eye(3, dtype=np.float32)
            rotated = self._rotated_coordinates((cx, cy, cz), rotation)
            local_sigmas = np.asarray([
                max(0.20 * sx, 1.2 * self.dx),
                max(0.20 * sy, 1.2 * self.dy),
                max(0.20 * sz, 1.2 * self.dz),
            ], dtype=np.float32)
            principal_axis = int(np.argmax(np.asarray([sx, sy, sz], dtype=np.float32)))
            local_sigmas[principal_axis] = max(0.70 * float(max(sx, sy, sz)), 2.0 * self.dz)
            axis_response = np.exp(
                -0.5 * (
                    (rotated[..., 0] / local_sigmas[0]) ** 2 +
                    (rotated[..., 1] / local_sigmas[1]) ** 2 +
                    (rotated[..., 2] / local_sigmas[2]) ** 2
                )
            ).astype(np.float32)
            axis_heatmap = np.maximum(axis_heatmap, axis_response)

        return {
            'location_mask': soft_mask.astype(np.float32),
            'positive_body_mask': positive_soft_mask.astype(np.float32),
            'negative_body_mask': negative_soft_mask.astype(np.float32),
            'center_heatmap': center_heatmap.astype(np.float32),
            'axis_heatmap': axis_heatmap.astype(np.float32),
        }

    def _generate_anomaly_body(self):
        """Generate random anomaly bodies for the localization task."""
        if self.structured_case_probability > 0.0 and np.random.rand() < self.structured_case_probability:
            return self._generate_structured_bar_case()

        rho_model = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
        anomaly_info = []

        num_bodies = self._sample_body_count()

        for _ in range(num_bodies):
            mask, info = self._build_single_body()
            rho_model[mask] += info['density']
            anomaly_info.append(info)

        return rho_model, anomaly_info

    def _add_regional_field(self, gz_3d):
        """Add a low-frequency regional background field."""
        # Random second-order polynomial background field.
        a0 = np.random.uniform(-5, 5)
        a1 = np.random.uniform(-0.01, 0.01)
        a2 = np.random.uniform(-0.01, 0.01)
        a3 = np.random.uniform(-0.0001, 0.0001)
        a4 = np.random.uniform(-0.0001, 0.0001)
        a5 = np.random.uniform(-0.0001, 0.0001)

        regional = (
            a0 +
            a1 * self.X_2d +
            a2 * self.Y_2d +
            a3 * self.X_2d**2 +
            a4 * self.Y_2d**2 +
            a5 * self.X_2d * self.Y_2d
        )
        if np.random.rand() < 0.6:
            regional += np.random.uniform(0.5, 3.0) * np.sin(
                2 * np.pi * self.X_2d / np.random.uniform(1800.0, 4200.0) +
                np.random.uniform(0.0, 2 * np.pi)
            )
        if np.random.rand() < 0.6:
            regional += np.random.uniform(0.5, 3.0) * np.cos(
                2 * np.pi * self.Y_2d / np.random.uniform(1800.0, 4200.0) +
                np.random.uniform(0.0, 2 * np.pi)
            )

        # Extend the surface background through depth.
        regional_3d = np.tile(regional[:, :, np.newaxis], (1, 1, self.nz))

        return gz_3d + regional_3d

    def _add_mixed_noise(self, field, noise_level=0.03):
        """Add mixed noise to approximate field observations."""
        field_scale = max(float(np.max(np.abs(field))), 1e-12)

        # Gaussian white noise.
        white_noise = np.random.normal(0, noise_level * field_scale, field.shape)

        # Spatially correlated noise.
        correlated_noise = np.random.normal(0, noise_level * 0.5 * field_scale, field.shape)
        correlated_noise = gaussian_filter(correlated_noise, sigma=2)

        stripe_noise = np.zeros_like(field, dtype=np.float32)
        if field.ndim == 2 and np.random.rand() < 0.6:
            stripe_axis = np.random.choice([0, 1])
            stripe_scale = np.random.uniform(0.15, 0.45) * noise_level * field_scale
            if stripe_axis == 0:
                stripe_profile = gaussian_filter(
                    np.random.normal(0, stripe_scale, size=(field.shape[0],)),
                    sigma=np.random.uniform(1.0, 2.5)
                )
                stripe_noise = stripe_profile[:, np.newaxis]
            else:
                stripe_profile = gaussian_filter(
                    np.random.normal(0, stripe_scale, size=(field.shape[1],)),
                    sigma=np.random.uniform(1.0, 2.5)
                )
                stripe_noise = stripe_profile[np.newaxis, :]

        spike_noise = np.zeros_like(field, dtype=np.float32)
        if field.ndim == 2 and np.random.rand() < 0.35:
            spike_mask = np.random.rand(*field.shape) < np.random.uniform(0.001, 0.004)
            spike_noise = spike_mask * np.random.normal(0, 2.0 * noise_level * field_scale, field.shape)

        return field + white_noise + correlated_noise + stripe_noise + spike_noise

    def generate_sample(
        self,
        noise_level=0.03,
        add_regional=True,
        return_surface_channels=False,
        return_location_targets=False,
        return_density_model=False,
    ):
        """Generate one training sample with optional surface channels and labels."""
        # Generate the density-contrast anomaly body.
        rho_model, anomaly_info = self._generate_anomaly_body()
        location_targets = self._build_location_targets(rho_model, anomaly_info)

        # 3D FFT forward modeling.
        F_rho = fftn(rho_model)
        F_gz = self.H * F_rho
        gz_3d_anomaly = np.real(ifftn(F_gz)) * 1e5  # Convert to mGal.

        # Compute derivative components for the analytic-signal amplitude.
        from scipy.fft import fftfreq
        kx = 2 * np.pi * fftfreq(self.nx, self.dx)
        ky = 2 * np.pi * fftfreq(self.ny, self.dy)
        kz = 2 * np.pi * fftfreq(self.nz, self.dz)
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

        F_Vzx = 1j * KX * F_gz
        F_Vzy = 1j * KY * F_gz
        F_Vzz = 1j * KZ * F_gz

        Vzx_3d = np.real(ifftn(F_Vzx)) * 1e5
        Vzy_3d = np.real(ifftn(F_Vzy)) * 1e5
        Vzz_3d = np.real(ifftn(F_Vzz)) * 1e5

        # Analytic-signal amplitude.
        analytic_amp = np.sqrt(Vzx_3d**2 + Vzy_3d**2 + Vzz_3d**2)

        gz_3d_observed = gz_3d_anomaly.copy()
        if add_regional:
            gz_3d_observed = self._add_regional_field(gz_3d_observed)

        # Extract surface gravity as one 2D input channel.
        X_input = gz_3d_observed[:, :, 0].copy()

        # Add mixed observation noise.
        X_input = self._add_mixed_noise(X_input, noise_level)

        # Extract the surface vertical gradient and analytic amplitude.
        Gzz_input = Vzz_3d[:, :, 0].copy()
        A_input = analytic_amp[:, :, 0].copy()

        # Add independent noise to each modality.
        Gzz_input = self._add_mixed_noise(Gzz_input, noise_level)
        A_input = self._add_mixed_noise(A_input, noise_level * 0.7)

        # The target is the anomaly-only 3D gravity field.
        Y_label = gz_3d_anomaly.astype(np.float32)

        surface_channels = {
            'gz': X_input.astype(np.float32),
            'gzz': Gzz_input.astype(np.float32),
            'amp': A_input.astype(np.float32),
        }

        if return_surface_channels and return_location_targets and return_density_model:
            return surface_channels, Y_label, anomaly_info, location_targets, rho_model.astype(np.float32)
        if return_surface_channels and return_location_targets:
            return surface_channels, Y_label, anomaly_info, location_targets
        if return_surface_channels and return_density_model:
            return surface_channels, Y_label, anomaly_info, rho_model.astype(np.float32)
        if return_surface_channels:
            return surface_channels, Y_label, anomaly_info

        return X_input, Y_label, A_input, anomaly_info


class GravityDataset(IterableDataset):
    """PyTorch IterableDataset for streaming synthetic samples."""

    def __init__(self, num_samples=1000, nx=64, ny=64, nz=64,
                 noise_level=0.03, add_regional=True, input_mode='gz_amp',
                 use_location_targets=False, depth_sampling_mode='shallow_biased',
                 structured_case_probability=0.0, tilted_pair_probability=0.7, wide_pair_probability=0.75):
        self.num_samples = num_samples
        self.generator = GravityDataGenerator(
            nx,
            ny,
            nz,
            depth_sampling_mode=depth_sampling_mode,
            structured_case_probability=structured_case_probability,
            tilted_pair_probability=tilted_pair_probability,
            wide_pair_probability=wide_pair_probability,
        )
        self.noise_level = noise_level
        self.add_regional = add_regional
        self.input_mode = input_mode
        self.use_location_targets = use_location_targets

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            sample_count = self.num_samples
        else:
            base = self.num_samples // worker_info.num_workers
            remainder = self.num_samples % worker_info.num_workers
            sample_count = base + (1 if worker_info.id < remainder else 0)

        for _ in range(sample_count):
            sample = self.generator.generate_sample(
                noise_level=self.noise_level,
                add_regional=self.add_regional,
                return_surface_channels=True,
                return_location_targets=self.use_location_targets
            )

            if self.use_location_targets:
                surface_channels, Y, anomaly_info, location_targets = sample
            else:
                surface_channels, Y, anomaly_info = sample
                location_targets = None

            target_scale = get_target_scale(surface_channels, Y)
            input_tensor_np, channel_scales, target_scale = build_model_input(
                surface_channels,
                input_mode=self.input_mode,
                target_scale=target_scale
            )
            Y = (Y / target_scale).astype(np.float32)

            # Convert NumPy arrays to PyTorch tensors.
            input_tensor = torch.from_numpy(input_tensor_np).float()
            Y_tensor = torch.from_numpy(Y[np.newaxis, :, :, :]).float()  # (1, 64, 64, 64)
            meta = {
                'target_scale': torch.tensor(target_scale, dtype=torch.float32),
            }

            if location_targets is not None:
                for key, value in location_targets.items():
                    meta[key] = torch.from_numpy(value[np.newaxis, :, :, :]).float()

            # Return multichannel inputs, 3D targets, and metadata tensors.
            yield input_tensor, Y_tensor, meta


def create_dataloader(batch_size=4, num_samples=1000, num_workers=2,
                      noise_level=0.03, add_regional=True, input_mode='gz_amp',
                      use_location_targets=False, depth_sampling_mode='shallow_biased',
                      structured_case_probability=0.0, tilted_pair_probability=0.7, wide_pair_probability=0.75):
    """Create a DataLoader for streaming synthetic gravity samples."""
    dataset = GravityDataset(
        num_samples=num_samples,
        noise_level=noise_level,
        add_regional=add_regional,
        input_mode=input_mode,
        use_location_targets=use_location_targets,
        depth_sampling_mode=depth_sampling_mode,
        structured_case_probability=structured_case_probability,
        tilted_pair_probability=tilted_pair_probability,
        wide_pair_probability=wide_pair_probability,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


# Convenience helper for generating one sample without manually creating a generator.
_default_generator = None

def generate_random_sample(noise_level=0.03, add_regional=False):
    """
    Generate one random synthetic sample.

    Args:
        noise_level: Relative noise level.
        add_regional: Whether to add a regional background field.

    Returns:
        X: Noisy surface gravity anomaly, shape (64, 64).
        Y: Interior 3D gravity field, shape (64, 64, 64).
        anomaly_info: Metadata describing the synthetic anomaly bodies.
    """
    global _default_generator
    if _default_generator is None:
        _default_generator = GravityDataGenerator()

    return _default_generator.generate_sample(noise_level=noise_level, add_regional=add_regional)


if __name__ == "__main__":
    print("Testing the on-the-fly data generator...")
    generator = GravityDataGenerator()

    X, Y, A, anomaly_info = generator.generate_sample(noise_level=0.05)
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"X range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"Y range: [{Y.min():.4f}, {Y.max():.4f}]")
    print(f"A shape: {A.shape}")
    print(f"Anomaly count: {len(anomaly_info)}")

    print("\nTesting the DataLoader...")
    dataloader = create_dataloader(batch_size=2, num_samples=4, num_workers=0, use_location_targets=True)
    for batch_idx, (X_batch, Y_batch, meta) in enumerate(dataloader):
        print(f"Batch {batch_idx}: X shape {X_batch.shape}, Y shape {Y_batch.shape}")
        print(f"Meta keys: {list(meta.keys())}")
        if batch_idx == 0:
            break
