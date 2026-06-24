"""
Hybrid physics-guided loss functions for 3D gravity imaging.

The loss combines data fidelity with surface consistency, localization,
depth-profile, artifact-suppression, and auxiliary supervision terms. These
components are documented in English so reviewers can trace each term used by
the training script.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data_generator import get_input_channel_names


class Sobel3D(nn.Module):
    """3D Sobel gradient operator."""

    def __init__(self):
        super().__init__()

        sobel_x = torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ], dtype=torch.float32)

        sobel_y = torch.tensor([
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        ], dtype=torch.float32)

        sobel_z = torch.tensor([
            [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        ], dtype=torch.float32)

        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3, 3))
        self.register_buffer('sobel_z', sobel_z.view(1, 1, 3, 3, 3))

    def forward(self, x):
        """Compute the 3D gradient magnitude."""
        padded = F.pad(x, (1, 1, 1, 1, 1, 1), mode='replicate')
        grad_x = F.conv3d(padded, self.sobel_x)
        grad_y = F.conv3d(padded, self.sobel_y)
        grad_z = F.conv3d(padded, self.sobel_z)

        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-8)

        return grad_magnitude


class CorrectUpwardContinuation(nn.Module):
    """
    Upward-continuation operator that integrates all depth levels.

    In the horizontal wavenumber domain, each depth slice contributes as
    g(kx, ky) = sum_z F(kx, ky, z) * exp(-|k| z).
    """

    def __init__(self, nx=64, ny=64, nz=64, dx=50, dy=50, dz=50):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz

        # Precompute the horizontal wavenumber grid.
        kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
        ky = 2 * np.pi * np.fft.fftfreq(ny, dy)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        self.K = np.sqrt(KX**2 + KY**2)

    def forward(self, field_3d):
        """
        Continue a 3D field upward to the observation surface.

        field_3d: (B, 1, 64, 64, 64)
        Returns: (B, 1, 64, 64) surface observations.
        """
        B = field_3d.shape[0]
        device = field_3d.device

        # Initialize the complex-valued surface spectrum.
        field_surface = torch.zeros(B, self.nx, self.ny, dtype=torch.complex64, device=device)

        # Sum the upward-continued contribution from each depth level.
        for z_idx in range(self.nz):
            # Extract one 2D depth slice.
            field_z = field_3d[:, 0, :, :, z_idx]  # (B, 64, 64)

            # 2D FFT
            field_fft = torch.fft.fft2(field_z)  # (B, 64, 64)

            # Convert the index to physical depth.
            z_physical = z_idx * self.dz

            # Upward-continuation filter: exp(-|k| z).
            K_tensor = torch.from_numpy(self.K).float().to(device)
            filter_uc = torch.exp(-K_tensor * z_physical)

            # Use out-of-place accumulation to keep autograd behavior explicit.
            field_surface = field_surface + field_fft * filter_uc

        # Transform back to the spatial domain.
        field_uc = torch.fft.ifft2(field_surface).real  # (B, 64, 64)

        return field_uc.unsqueeze(1)  # (B, 1, 64, 64)


class FocalLoss(nn.Module):
    """
    Focal-style L1 loss.

    High-amplitude target regions receive larger weights to reduce amplitude
    underestimation near compact anomaly bodies.
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target, voxel_weight=None):
        """
        pred: Predicted field.
        target: Reference field.
        """
        # Absolute prediction error.
        diff = torch.abs(pred - target)

        target_magnitude = torch.abs(target)
        flat_magnitude = target_magnitude.view(target_magnitude.size(0), -1)
        max_magnitude = flat_magnitude.max(dim=1)[0].view(-1, 1, 1, 1, 1)
        normalized_magnitude = target_magnitude / (max_magnitude + 1e-8)
        weight = self.alpha + normalized_magnitude ** self.gamma

        if voxel_weight is not None:
            voxel_weight = voxel_weight.to(device=pred.device, dtype=pred.dtype)
            weight = weight * voxel_weight

        reduce_dims = tuple(range(1, weight.ndim))
        weight = weight / (weight.mean(dim=reduce_dims, keepdim=True) + 1e-8)

        return torch.mean(weight * diff)


class PoissonConstraint(nn.Module):
    """
    Laplacian regularization for source-free regions.

    The term encourages smooth potential-field behavior away from sources.
    """

    def __init__(self):
        super().__init__()

        # 3D finite-difference Laplacian kernel.
        laplacian_kernel = torch.tensor([
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ], dtype=torch.float32).view(1, 1, 3, 3, 3)

        self.register_buffer('laplacian_kernel', laplacian_kernel)

    def forward(self, field_3d):
        """
        Compute the Laplacian.

        field_3d: (B, 1, 64, 64, 64)
        Returns: Laplacian values.
        """
        padded = F.pad(field_3d, (1, 1, 1, 1, 1, 1), mode='replicate')
        laplacian = F.conv3d(padded, self.laplacian_kernel)
        return laplacian


class AdaptiveWeightScheduler:
    """
    Adaptive loss-weight scheduler driven by validation loss.

    The update rule follows the spirit of GradNorm-style balancing while
    keeping the implementation lightweight.
    """

    def __init__(self, initial_alpha=0.3, initial_beta=0.8, initial_gamma=0.15):
        self.alpha = initial_alpha
        self.beta = initial_beta
        self.gamma = initial_gamma

        self.prev_val_loss = float('inf')
        self.loss_history = []

    def step(self, current_val_loss, epoch):
        """Update loss weights from the validation-loss trend."""
        self.loss_history.append(current_val_loss)

        # Relative validation-loss change.
        if len(self.loss_history) > 1:
            loss_change = (self.loss_history[-2] - self.loss_history[-1]) / (self.loss_history[-2] + 1e-8)
        else:
            loss_change = 0

        # Increase regularization when validation progress slows.
        if loss_change < 0.01:
            self.alpha = min(self.alpha * 1.05, 0.5)
            self.beta = min(self.beta * 1.05, 1.0)
            self.gamma = min(self.gamma * 1.05, 0.3)
        elif loss_change > 0.05:
            self.alpha = max(self.alpha * 0.95, 0.1)
            self.beta = max(self.beta * 0.95, 0.5)
            self.gamma = max(self.gamma * 0.95, 0.05)

        return self.alpha, self.beta, self.gamma


class FullyCorrectHybridPhysicsLoss(nn.Module):
    """
    Hybrid physics-guided objective used to train the network.

    It combines L1 data fidelity, Sobel-gradient matching, surface
    consistency, localization losses, auxiliary masks, depth-profile matching,
    anti-stripe penalties, amplitude constraints, and Poisson-style smoothing.
    """

    def __init__(
        self,
        nx=64,
        ny=64,
        nz=64,
        dx=50,
        dy=50,
        dz=50,
        input_mode='gz_amp',
        loss_profile='localization_first'
    ):
        super().__init__()

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.input_mode = input_mode
        self.loss_profile = loss_profile
        self.input_channel_names = get_input_channel_names(input_mode)

        self.sobel_3d = Sobel3D()
        self.upward_continuation = CorrectUpwardContinuation(nx, ny, nz, dx, dy, dz)
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.poisson_constraint = PoissonConstraint()
        self.weight_scheduler = AdaptiveWeightScheduler()
        self.use_adaptive_weight_scheduler = loss_profile == 'balanced_physics'
        self.depth_weight_gain = 0.60
        self.depth_profile_cdf_weight = 0.35
        self.z_center_weight = 1.75
        self.boundary_shell_width = 3

        self.register_buffer('depth_voxel_weight', self._build_depth_weight())
        self.register_buffer(
            'center_axis_weight',
            torch.tensor([1.0, 1.0, self.z_center_weight], dtype=torch.float32)
        )
        self.register_buffer(
            'boundary_shell_mask',
            self._build_boundary_shell_mask(shell_width=self.boundary_shell_width)
        )

        self.alpha = 0.0
        self.beta = 0.0
        self.gamma = 0.0
        self.delta = 0.0
        self.epsilon = 0.0
        self.zeta = 0.0
        self.eta = 0.0
        self.theta = 0.0
        self.iota = 0.0
        self.kappa = 0.0
        self.amp_weight = 0.0
        self.edge_weight = 0.0
        self.lambda_body_aux = 0.0
        self.lambda_center_aux = 0.0
        self.lambda_signed_body_aux = 0.0
        self.lambda_axis_aux = 0.0
        self._set_epoch_weights(epoch=0)

    def forward(self, Y_pred, Y_true, X_input, epoch=0, val_loss=None, location_targets=None, aux_predictions=None):
        """
        Compute the hybrid loss with depth-aware compensation.
        """
        if not self.use_adaptive_weight_scheduler:
            self._set_epoch_weights(epoch)

        depth_weight = self.depth_voxel_weight.to(device=Y_pred.device, dtype=Y_pred.dtype)
        loss_l1 = self.focal_loss(Y_pred, Y_true, voxel_weight=depth_weight)

        grad_pred = self.sobel_3d(Y_pred)
        grad_true = self.sobel_3d(Y_true)
        loss_grad = F.l1_loss(grad_pred, grad_true)

        loss_phys, loss_surface_aux = self._compute_surface_consistency_loss(Y_pred, X_input)

        laplacian_pred = self.poisson_constraint(Y_pred)
        target_scale = torch.abs(Y_true)
        max_target_scale = target_scale.view(target_scale.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1, 1)
        source_mask = target_scale / (max_target_scale + 1e-8)
        background_weight = 1.0 - torch.clamp(source_mask, 0.0, 1.0)
        loss_poisson = torch.mean((laplacian_pred * background_weight) ** 2)

        loss_sharpness = self._compute_sharpness_loss(Y_pred, Y_true)
        loss_projection = self._compute_projection_loss(Y_pred, Y_true)
        loss_center = self._compute_center_of_mass_loss(Y_pred, Y_true)
        loss_location = Y_pred.new_tensor(0.0)
        loss_depth_profile = self._compute_depth_profile_loss(Y_pred, Y_true)
        loss_artifact = self._compute_anti_stripe_loss(Y_pred, Y_true)
        loss_amplitude = self._compute_amplitude_loss(Y_pred, Y_true)
        loss_boundary = self._compute_boundary_suppression_loss(
            Y_pred,
            Y_true,
            grad_pred=grad_pred,
            grad_true=grad_true
        )
        loss_body_aux = Y_pred.new_tensor(0.0)
        loss_center_aux = Y_pred.new_tensor(0.0)
        loss_signed_body_aux = Y_pred.new_tensor(0.0)
        loss_axis_aux = Y_pred.new_tensor(0.0)

        if location_targets is not None and 'location_mask' in location_targets:
            location_mask = location_targets['location_mask']
            center_heatmap = location_targets.get('center_heatmap', location_mask)
            positive_body_mask = location_targets.get('positive_body_mask')
            negative_body_mask = location_targets.get('negative_body_mask')
            axis_heatmap = location_targets.get('axis_heatmap', location_mask)
            pred_location = self._compute_source_localization_response(Y_pred)
            target_location = self._build_location_supervision_target(location_mask, center_heatmap)
            loss_location = self._compute_location_loss(pred_location, target_location, center_heatmap)
            if aux_predictions is not None:
                pred_body_mask = aux_predictions.get('body_mask')
                if pred_body_mask is not None:
                    loss_body_aux = self._compute_body_mask_aux_loss(pred_body_mask, location_mask, center_heatmap)

                pred_center_heatmap = aux_predictions.get('center_heatmap')
                if pred_center_heatmap is not None:
                    loss_center_aux = self._compute_center_heatmap_aux_loss(pred_center_heatmap, center_heatmap)

                pred_positive_body_mask = aux_predictions.get('positive_body_mask')
                pred_negative_body_mask = aux_predictions.get('negative_body_mask')
                if (
                    pred_positive_body_mask is not None and
                    pred_negative_body_mask is not None and
                    positive_body_mask is not None and
                    negative_body_mask is not None
                ):
                    loss_signed_body_aux = self._compute_signed_body_mask_aux_loss(
                        pred_positive_body_mask,
                        pred_negative_body_mask,
                        positive_body_mask,
                        negative_body_mask,
                        axis_heatmap,
                    )

                pred_axis_heatmap = aux_predictions.get('axis_heatmap')
                if pred_axis_heatmap is not None:
                    loss_axis_aux = self._compute_axis_heatmap_aux_loss(
                        pred_axis_heatmap,
                        axis_heatmap,
                        location_mask,
                    )
            loss_projection = 0.5 * (
                loss_projection + self._compute_projection_loss(pred_location, target_location)
            )
            loss_center = 0.5 * (
                loss_center + self._compute_center_of_mass_loss(pred_location, center_heatmap)
            )
            loss_depth_profile = 0.5 * (
                loss_depth_profile + self._compute_depth_profile_loss(pred_location, target_location)
            )
            loss_artifact = 0.5 * (
                loss_artifact + self._compute_anti_stripe_loss(pred_location, target_location)
            )

        if val_loss is not None:
            self.update_weights(val_loss, epoch)

        loss_total = (loss_l1 +
                     self.alpha * loss_grad +
                     self.beta * loss_phys +
                     self.eta * loss_surface_aux +
                     self.theta * loss_location +
                     self.gamma * loss_sharpness +
                     self.delta * loss_poisson +
                     self.epsilon * loss_projection +
                     self.zeta * loss_center +
                     self.iota * loss_depth_profile +
                     self.kappa * loss_artifact +
                     self.amp_weight * loss_amplitude +
                     self.edge_weight * loss_boundary +
                     self.lambda_body_aux * loss_body_aux +
                     self.lambda_center_aux * loss_center_aux +
                     self.lambda_signed_body_aux * loss_signed_body_aux +
                     self.lambda_axis_aux * loss_axis_aux)

        return {
            'total': loss_total,
            'l1': loss_l1,
            'grad': loss_grad,
            'phys': loss_phys,
            'surface_aux': loss_surface_aux,
            'location': loss_location,
            'depth_profile': loss_depth_profile,
            'artifact': loss_artifact,
            'amplitude': loss_amplitude,
            'boundary': loss_boundary,
            'body_aux': loss_body_aux,
            'center_aux': loss_center_aux,
            'signed_body_aux': loss_signed_body_aux,
            'axis_aux': loss_axis_aux,
            'poisson': loss_poisson,
            'sharpness': loss_sharpness,
            'projection': loss_projection,
            'center': loss_center,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'delta': self.delta,
            'epsilon': self.epsilon,
            'zeta': self.zeta,
            'eta': self.eta,
            'theta': self.theta,
            'iota': self.iota,
            'kappa': self.kappa,
            'amp_weight': self.amp_weight,
            'edge_weight': self.edge_weight,
            'lambda_body_aux': self.lambda_body_aux,
            'lambda_center_aux': self.lambda_center_aux,
            'lambda_signed_body_aux': self.lambda_signed_body_aux,
            'lambda_axis_aux': self.lambda_axis_aux,
        }

    def _build_depth_weight(self):
        """Use a softened inverse-distance compensation so deep voxels are not ignored."""
        depth_index = torch.arange(self.nz, dtype=torch.float32)
        softened_compensation = torch.sqrt(1.0 + depth_index)
        weight = 1.0 + self.depth_weight_gain * (softened_compensation - 1.0)
        return (weight / weight.mean()).view(1, 1, 1, 1, self.nz)

    def _weighted_l1_loss(self, pred, target, weight):
        weight = weight.to(device=pred.device, dtype=pred.dtype)
        weight = weight / (weight.mean() + 1e-8)
        return torch.mean(weight * torch.abs(pred - target))

    def _relative_scale_loss(self, pred_scale, true_scale):
        return torch.mean(torch.abs(pred_scale - true_scale) / (torch.abs(true_scale) + 1e-6))

    def _log_scale_loss(self, pred_scale, true_scale):
        pred_scale = torch.clamp(pred_scale, min=0.0)
        true_scale = torch.clamp(true_scale, min=0.0)
        return F.smooth_l1_loss(torch.log1p(pred_scale), torch.log1p(true_scale))

    def _build_boundary_shell_mask(self, shell_width=3):
        mask = torch.zeros((self.nx, self.ny, self.nz), dtype=torch.float32)
        shell_width = max(1, int(shell_width))

        mask[:shell_width, :, :] = 1.0
        mask[-shell_width:, :, :] = 1.0
        mask[:, :shell_width, :] = 1.0
        mask[:, -shell_width:, :] = 1.0
        mask[:, :, :shell_width] = 1.0
        mask[:, :, -shell_width:] = 1.0

        return mask.view(1, 1, self.nx, self.ny, self.nz)

    def _compute_surface_consistency_loss(self, Y_pred, X_input):
        """Match the predicted surface response to whichever channels were actually provided."""
        primary_losses = []
        aux_losses = []

        if 'gz' in self.input_channel_names:
            gz_channel_idx = self.input_channel_names.index('gz')
            gz_obs = X_input[:, gz_channel_idx:gz_channel_idx + 1, :, :]
            pred_surface_gz = Y_pred[:, :, :, :, 0]
            primary_losses.append(self._compute_normalized_surface_alignment(
                self._detrend_surface_map(pred_surface_gz),
                self._detrend_surface_map(gz_obs)
            ))

        if 'gzz' in self.input_channel_names:
            gzz_channel_idx = self.input_channel_names.index('gzz')
            obs_surface_gzz = X_input[:, gzz_channel_idx:gzz_channel_idx + 1, :, :]
            pred_surface_gzz = self._compute_surface_vertical_gradient(Y_pred)
            target_bucket = primary_losses if 'gz' not in self.input_channel_names else aux_losses
            target_bucket.append(self._compute_normalized_surface_alignment(pred_surface_gzz, obs_surface_gzz))

        if 'amp' in self.input_channel_names:
            amp_channel_idx = self.input_channel_names.index('amp')
            obs_surface_amp = X_input[:, amp_channel_idx:amp_channel_idx + 1, :, :]
            pred_surface_amp = self._compute_surface_analytic_amplitude(Y_pred)
            target_bucket = primary_losses if not primary_losses else aux_losses
            target_bucket.append(self._compute_normalized_surface_alignment(pred_surface_amp, obs_surface_amp))

        loss_primary = torch.stack(primary_losses).mean() if primary_losses else Y_pred.new_tensor(0.0)
        loss_aux = torch.stack(aux_losses).mean() if aux_losses else Y_pred.new_tensor(0.0)
        return loss_primary, loss_aux

    def _compute_amplitude_loss(self, Y_pred, Y_true):
        """Keep the model from collapsing toward a near-zero volume without letting single spikes dominate."""
        reduce_dims = tuple(range(1, Y_pred.ndim))
        surface_dims = tuple(range(1, Y_pred[:, :, :, :, 0].ndim))

        pred_abs = torch.abs(Y_pred)
        true_abs = torch.abs(Y_true)

        pred_rms = torch.sqrt(torch.mean(Y_pred ** 2, dim=reduce_dims) + 1e-8)
        true_rms = torch.sqrt(torch.mean(Y_true ** 2, dim=reduce_dims) + 1e-8)

        pred_mean_abs = torch.mean(pred_abs, dim=reduce_dims)
        true_mean_abs = torch.mean(true_abs, dim=reduce_dims)
        pred_energy = torch.mean(pred_abs ** 2, dim=reduce_dims)
        true_energy = torch.mean(true_abs ** 2, dim=reduce_dims)

        pred_surface = Y_pred[:, :, :, :, 0]
        true_surface = Y_true[:, :, :, :, 0]
        pred_surface_rms = torch.sqrt(torch.mean(pred_surface ** 2, dim=surface_dims) + 1e-8)
        true_surface_rms = torch.sqrt(torch.mean(true_surface ** 2, dim=surface_dims) + 1e-8)
        pred_surface_mean_abs = torch.mean(torch.abs(pred_surface), dim=surface_dims)
        true_surface_mean_abs = torch.mean(torch.abs(true_surface), dim=surface_dims)

        flat_pred = pred_abs.view(pred_abs.size(0), -1)
        flat_true = true_abs.view(true_abs.size(0), -1)
        pred_p995 = torch.quantile(flat_pred, 0.995, dim=1)
        true_p995 = torch.quantile(flat_true, 0.995, dim=1)
        pred_peak = torch.amax(flat_pred, dim=1)
        true_peak = torch.amax(flat_true, dim=1)

        flat_pred_surface = torch.abs(pred_surface).view(pred_surface.size(0), -1)
        flat_true_surface = torch.abs(true_surface).view(true_surface.size(0), -1)
        pred_surface_p995 = torch.quantile(flat_pred_surface, 0.995, dim=1)
        true_surface_p995 = torch.quantile(flat_true_surface, 0.995, dim=1)

        volume_cap = torch.clamp(4.0 * true_p995 + 0.05, min=0.05)
        surface_cap = torch.clamp(4.0 * true_surface_p995 + 0.03, min=0.03)
        overflow_penalty = torch.mean(F.relu(pred_peak - volume_cap) / (volume_cap + 1e-6))
        surface_overflow_penalty = torch.mean(
            F.relu(torch.amax(flat_pred_surface, dim=1) - surface_cap) / (surface_cap + 1e-6)
        )

        return (
            0.22 * self._log_scale_loss(pred_rms, true_rms) +
            0.18 * self._log_scale_loss(pred_mean_abs, true_mean_abs) +
            0.18 * self._log_scale_loss(pred_energy, true_energy) +
            0.16 * self._log_scale_loss(pred_p995, true_p995) +
            0.10 * self._log_scale_loss(pred_surface_rms, true_surface_rms) +
            0.08 * self._log_scale_loss(pred_surface_mean_abs, true_surface_mean_abs) +
            0.04 * self._log_scale_loss(pred_surface_p995, true_surface_p995) +
            0.02 * self._log_scale_loss(pred_peak, true_peak) +
            0.01 * overflow_penalty +
            0.01 * surface_overflow_penalty
        )

    def _compute_surface_vertical_gradient(self, field_3d):
        """Approximate Gzz at the surface from the predicted 3D gz volume."""
        surface = field_3d[:, :, :, :, 0]

        if field_3d.shape[-1] >= 3:
            next_1 = field_3d[:, :, :, :, 1]
            next_2 = field_3d[:, :, :, :, 2]
            return (-3.0 * surface + 4.0 * next_1 - next_2) / (2.0 * self.dz)

        next_1 = field_3d[:, :, :, :, 1]
        return (next_1 - surface) / self.dz

    def _compute_surface_analytic_amplitude(self, field_3d):
        """Approximate analytic-signal amplitude from the predicted surface gz."""
        surface = field_3d[:, :, :, :, 0]
        grad_x = self._central_difference_2d(surface, dim='x', spacing=self.dx)
        grad_y = self._central_difference_2d(surface, dim='y', spacing=self.dy)
        grad_z = self._compute_surface_vertical_gradient(field_3d)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2 + 1e-8)

    def _central_difference_2d(self, field_2d, dim, spacing):
        """Compute a simple central difference on a 2D surface map."""
        if dim == 'x':
            padded = F.pad(field_2d, (0, 0, 1, 1), mode='replicate')
            return (padded[:, :, 2:, :] - padded[:, :, :-2, :]) / (2.0 * spacing)
        if dim == 'y':
            padded = F.pad(field_2d, (1, 1, 0, 0), mode='replicate')
            return (padded[:, :, :, 2:] - padded[:, :, :, :-2]) / (2.0 * spacing)
        raise ValueError(f"Unsupported derivative dimension: {dim}")

    def _compute_normalized_surface_alignment(self, pred_surface, obs_surface):
        """Match derivative-like channels by normalized shape rather than raw amplitude."""
        pred_norm = self._normalize_surface_map(pred_surface)
        obs_norm = self._normalize_surface_map(obs_surface)

        loss_l1 = F.l1_loss(pred_norm, obs_norm)
        pred_flat = pred_norm.flatten(start_dim=1)
        obs_flat = obs_norm.flatten(start_dim=1)
        loss_corr = 1.0 - F.cosine_similarity(pred_flat, obs_flat, dim=1).mean()

        return 0.5 * (loss_l1 + loss_corr)

    def _normalize_surface_map(self, surface_map):
        surface_centered = surface_map - surface_map.mean(dim=(-2, -1), keepdim=True)
        surface_scale = torch.sqrt(torch.mean(surface_centered ** 2, dim=(-2, -1), keepdim=True) + 1e-8)
        return surface_centered / surface_scale

    def _detrend_surface_map(self, surface_map):
        """Suppress broad regional trends so the loss focuses on anomaly location."""
        low_freq = F.avg_pool2d(surface_map, kernel_size=15, stride=1, padding=7)
        return surface_map - low_freq

    def _compute_sharpness_loss(self, Y_pred, Y_true):
        """Preserve anomaly edges so the recovered body outline stays visible."""
        laplacian_pred = self.poisson_constraint(Y_pred)
        laplacian_true = self.poisson_constraint(Y_true)

        grad_true = self.sobel_3d(Y_true)
        weight = torch.clamp(grad_true, 0, 1)

        loss_sharpness = F.l1_loss(laplacian_pred * weight, laplacian_true * weight)
        return loss_sharpness

    def _compute_projection_loss(self, Y_pred, Y_true):
        """Match energy projections so anomaly position stays visible in slices."""
        pred_energy = torch.abs(Y_pred)
        true_energy = torch.abs(Y_true)

        pred_xz = torch.sqrt(torch.mean(pred_energy ** 2, dim=3) + 1e-8)
        true_xz = torch.sqrt(torch.mean(true_energy ** 2, dim=3) + 1e-8)
        pred_yz = torch.sqrt(torch.mean(pred_energy ** 2, dim=2) + 1e-8)
        true_yz = torch.sqrt(torch.mean(true_energy ** 2, dim=2) + 1e-8)
        pred_xy = torch.sqrt(torch.mean(pred_energy ** 2, dim=4) + 1e-8)
        true_xy = torch.sqrt(torch.mean(true_energy ** 2, dim=4) + 1e-8)

        return (
            F.l1_loss(pred_xz, true_xz) +
            F.l1_loss(pred_yz, true_yz) +
            0.5 * F.l1_loss(pred_xy, true_xy)
        ) / 2.5

    def _compute_center_of_mass_loss(self, Y_pred, Y_true):
        """Penalize large offsets in anomaly energy centroid."""
        pred_mass = torch.abs(Y_pred).squeeze(1) + 1e-6
        true_mass = torch.abs(Y_true).squeeze(1) + 1e-6

        pred_center = self._compute_normalized_center(pred_mass)
        true_center = self._compute_normalized_center(true_mass)
        axis_weight = self.center_axis_weight.to(device=pred_center.device, dtype=pred_center.dtype)
        return F.smooth_l1_loss(pred_center * axis_weight, true_center * axis_weight)

    def _normalize_location_response(self, field_3d):
        """
        Normalize source-like responses with a robust interior quantile scale instead
        of a global max, so boundary spikes cannot collapse the real anomaly signal.
        """
        response = torch.abs(field_3d)
        interior_weight = 1.0 - self.boundary_shell_mask.to(device=response.device, dtype=response.dtype)
        flat_response = response.view(response.size(0), -1)
        flat_interior = interior_weight.view(1, -1).expand(response.size(0), -1) > 0.5
        interior_values = flat_response[:, flat_interior[0]]

        low_q = torch.quantile(interior_values, 0.50, dim=1, keepdim=True)
        high_q = torch.quantile(interior_values, 0.995, dim=1, keepdim=True)
        robust_scale = torch.clamp(high_q - low_q, min=1e-6)

        low_q = low_q.view(-1, 1, 1, 1, 1)
        robust_scale = robust_scale.view(-1, 1, 1, 1, 1)

        normalized = (response - low_q) / robust_scale
        normalized = torch.clamp(normalized, 0.0, 1.0)

        # Softly attenuate the outer shell during localization supervision without
        # completely discarding edge-near anomalies.
        attenuated_weight = 0.25 + 0.75 * interior_weight
        return normalized * attenuated_weight

    def _compute_source_localization_response(self, field_3d):
        """
        Build a source-focused response from gz using gradient/analytic-signal style cues.

        For gz, the 3D gradient magnitude is a better proxy for anomaly boundaries than
        the Laplacian, whose extrema tend to emphasize top/bottom surfaces because
        Laplacian(gz) is related to the vertical derivative of density rather than density itself.
        """
        gradient_response = self._normalize_location_response(self.sobel_3d(field_3d))
        envelope_response = self._normalize_location_response(
            F.avg_pool3d(torch.abs(field_3d), kernel_size=5, stride=1, padding=2)
        )
        source_response = 0.78 * gradient_response + 0.22 * envelope_response
        return self._normalize_location_response(source_response)

    def _build_location_supervision_target(self, location_mask, center_heatmap):
        """Transform density-derived labels into a boundary-aware target matched to analytic-signal style responses."""
        boundary_response = self._normalize_location_response(self.sobel_3d(location_mask))
        body_response = self._normalize_location_response(location_mask)
        center_response = self._normalize_location_response(center_heatmap)
        target_response = (
            0.58 * boundary_response +
            0.24 * body_response +
            0.18 * center_response
        )
        return self._normalize_location_response(target_response)

    def _soft_dice_loss(self, pred, target, voxel_weight=None):
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        if voxel_weight is not None:
            voxel_weight = voxel_weight.to(device=pred.device, dtype=pred.dtype)
            weight_flat = (torch.ones_like(pred) * voxel_weight).view(pred.size(0), -1)
        else:
            weight_flat = torch.ones_like(pred_flat)

        intersection = (weight_flat * pred_flat * target_flat).sum(dim=1)
        denominator = (weight_flat * pred_flat).sum(dim=1) + (weight_flat * target_flat).sum(dim=1)
        dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
        return 1.0 - dice.mean()

    def _compute_location_loss(self, pred_location, target_location, center_heatmap):
        """Match a source-response volume to a boundary-aware localization target."""
        depth_weight = self.depth_voxel_weight.to(device=pred_location.device, dtype=pred_location.dtype)
        structure_loss = 0.5 * self._weighted_l1_loss(pred_location, target_location, depth_weight) + 0.5 * self._soft_dice_loss(
            pred_location,
            target_location,
            voxel_weight=depth_weight
        )
        projection_loss = self._compute_projection_loss(pred_location, target_location)
        center_loss = self._compute_center_of_mass_loss(pred_location, center_heatmap)

        return (
            0.45 * structure_loss +
            0.35 * projection_loss +
            0.20 * center_loss
        )

    def _compute_body_mask_aux_loss(self, pred_body_mask, location_mask, center_heatmap):
        """Directly supervise anomaly-body support so range estimation does not rely only on postprocessing."""
        pred_body_mask = torch.clamp(pred_body_mask, 1e-4, 1.0 - 1e-4)
        location_mask = torch.clamp(location_mask, 0.0, 1.0)
        center_heatmap = torch.clamp(center_heatmap, 0.0, 1.0)

        depth_weight = self.depth_voxel_weight.to(device=pred_body_mask.device, dtype=pred_body_mask.dtype)
        importance = depth_weight * (1.0 + 2.8 * location_mask + 1.6 * center_heatmap)
        reduce_dims = tuple(range(1, importance.ndim))
        importance = importance / (importance.mean(dim=reduce_dims, keepdim=True) + 1e-8)

        bce_loss = F.binary_cross_entropy(pred_body_mask, location_mask, weight=importance)
        dice_loss = self._soft_dice_loss(pred_body_mask, location_mask, voxel_weight=depth_weight)
        projection_loss = self._compute_projection_loss(pred_body_mask, location_mask)
        center_loss = self._compute_center_of_mass_loss(pred_body_mask, center_heatmap)
        depth_profile_loss = self._compute_depth_profile_loss(pred_body_mask, location_mask)
        boundary_loss = self._compute_boundary_suppression_loss(pred_body_mask, location_mask)

        return (
            0.30 * bce_loss +
            0.24 * dice_loss +
            0.18 * projection_loss +
            0.12 * center_loss +
            0.10 * depth_profile_loss +
            0.06 * boundary_loss
        )

    def _compute_signed_body_mask_aux_loss(self, pred_positive, pred_negative, positive_target, negative_target, axis_heatmap):
        """Supervise positive and negative bodies separately so opposite-sign bars do not collapse into one blob."""
        pred_positive = torch.clamp(pred_positive, 1e-4, 1.0 - 1e-4)
        pred_negative = torch.clamp(pred_negative, 1e-4, 1.0 - 1e-4)
        positive_target = torch.clamp(positive_target, 0.0, 1.0)
        negative_target = torch.clamp(negative_target, 0.0, 1.0)
        axis_heatmap = torch.clamp(axis_heatmap, 0.0, 1.0)

        depth_weight = self.depth_voxel_weight.to(device=pred_positive.device, dtype=pred_positive.dtype)

        def _single_branch_loss(pred_branch, target_branch):
            importance = depth_weight * (1.0 + 2.6 * target_branch + 1.2 * axis_heatmap)
            reduce_dims = tuple(range(1, importance.ndim))
            importance = importance / (importance.mean(dim=reduce_dims, keepdim=True) + 1e-8)
            bce_loss = F.binary_cross_entropy(pred_branch, target_branch, weight=importance)
            dice_loss = self._soft_dice_loss(pred_branch, target_branch, voxel_weight=depth_weight)
            projection_loss = self._compute_projection_loss(pred_branch, target_branch)
            depth_profile_loss = self._compute_depth_profile_loss(pred_branch, target_branch)
            return 0.34 * bce_loss + 0.28 * dice_loss + 0.22 * projection_loss + 0.16 * depth_profile_loss

        positive_loss = _single_branch_loss(pred_positive, positive_target)
        negative_loss = _single_branch_loss(pred_negative, negative_target)
        pred_union = torch.clamp(pred_positive + pred_negative, 0.0, 1.0)
        target_union = torch.clamp(positive_target + negative_target, 0.0, 1.0)
        union_projection_loss = self._compute_projection_loss(pred_union, target_union)
        overlap_penalty = torch.mean(pred_positive * pred_negative)

        return (
            0.42 * positive_loss +
            0.42 * negative_loss +
            0.10 * union_projection_loss +
            0.06 * overlap_penalty
        )

    def _compute_axis_heatmap_aux_loss(self, pred_axis_heatmap, axis_heatmap, location_mask):
        """Keep the anomaly response elongated along the true body axis instead of collapsing to a compact blob."""
        pred_axis_heatmap = torch.clamp(pred_axis_heatmap, 1e-4, 1.0 - 1e-4)
        axis_heatmap = torch.clamp(axis_heatmap, 0.0, 1.0)
        location_mask = torch.clamp(location_mask, 0.0, 1.0)

        depth_weight = self.depth_voxel_weight.to(device=pred_axis_heatmap.device, dtype=pred_axis_heatmap.dtype)
        importance = depth_weight * (1.0 + 4.0 * axis_heatmap + 1.5 * location_mask)
        reduce_dims = tuple(range(1, importance.ndim))
        importance = importance / (importance.mean(dim=reduce_dims, keepdim=True) + 1e-8)

        l1_loss = torch.mean(importance * torch.abs(pred_axis_heatmap - axis_heatmap))
        bce_loss = F.binary_cross_entropy(pred_axis_heatmap, axis_heatmap, weight=importance)
        projection_loss = self._compute_projection_loss(pred_axis_heatmap, axis_heatmap)
        depth_profile_loss = self._compute_depth_profile_loss(pred_axis_heatmap, axis_heatmap)

        return (
            0.34 * l1_loss +
            0.28 * bce_loss +
            0.22 * projection_loss +
            0.16 * depth_profile_loss
        )

    def _compute_center_heatmap_aux_loss(self, pred_center_heatmap, center_heatmap):
        """Keep the predicted anomaly center compact and correctly placed in depth."""
        pred_center_heatmap = torch.clamp(pred_center_heatmap, 1e-4, 1.0 - 1e-4)
        center_heatmap = torch.clamp(center_heatmap, 0.0, 1.0)

        depth_weight = self.depth_voxel_weight.to(device=pred_center_heatmap.device, dtype=pred_center_heatmap.dtype)
        importance = depth_weight * (1.0 + 5.0 * center_heatmap)
        reduce_dims = tuple(range(1, importance.ndim))
        importance = importance / (importance.mean(dim=reduce_dims, keepdim=True) + 1e-8)

        l1_loss = torch.mean(importance * torch.abs(pred_center_heatmap - center_heatmap))
        bce_loss = F.binary_cross_entropy(pred_center_heatmap, center_heatmap, weight=importance)
        projection_loss = self._compute_projection_loss(pred_center_heatmap, center_heatmap)
        center_loss = self._compute_center_of_mass_loss(pred_center_heatmap, center_heatmap)
        depth_profile_loss = self._compute_depth_profile_loss(pred_center_heatmap, center_heatmap)

        return (
            0.34 * l1_loss +
            0.28 * bce_loss +
            0.18 * center_loss +
            0.12 * projection_loss +
            0.08 * depth_profile_loss
        )

    def _compute_depth_profile_loss(self, Y_pred, Y_true):
        """Match how anomaly energy is distributed along depth to suppress shallow cheats."""
        pred_energy = torch.abs(Y_pred).squeeze(1)
        true_energy = torch.abs(Y_true).squeeze(1)

        pred_profile = torch.sqrt(torch.mean(pred_energy ** 2, dim=(1, 2)) + 1e-8)
        true_profile = torch.sqrt(torch.mean(true_energy ** 2, dim=(1, 2)) + 1e-8)
        depth_weight = self.depth_voxel_weight.to(device=Y_pred.device, dtype=Y_pred.dtype).view(1, self.nz)

        profile_loss = self._weighted_l1_loss(pred_profile, true_profile, depth_weight)

        pred_distribution = pred_profile / (pred_profile.sum(dim=1, keepdim=True) + 1e-8)
        true_distribution = true_profile / (true_profile.sum(dim=1, keepdim=True) + 1e-8)
        pred_cdf = torch.cumsum(pred_distribution, dim=1)
        true_cdf = torch.cumsum(true_distribution, dim=1)
        cdf_loss = self._weighted_l1_loss(pred_cdf, true_cdf, depth_weight)

        return (
            (1.0 - self.depth_profile_cdf_weight) * profile_loss +
            self.depth_profile_cdf_weight * cdf_loss
        )

    def _compute_anti_stripe_loss(self, Y_pred, Y_true):
        """Match projected curvature so the model cannot explain anomalies with narrow stripe bands."""
        pred_energy = torch.abs(Y_pred)
        true_energy = torch.abs(Y_true)

        pred_xz = torch.sqrt(torch.mean(pred_energy ** 2, dim=3) + 1e-8)
        true_xz = torch.sqrt(torch.mean(true_energy ** 2, dim=3) + 1e-8)
        pred_yz = torch.sqrt(torch.mean(pred_energy ** 2, dim=2) + 1e-8)
        true_yz = torch.sqrt(torch.mean(true_energy ** 2, dim=2) + 1e-8)

        pred_x_curvature = torch.diff(pred_xz, n=2, dim=2)
        true_x_curvature = torch.diff(true_xz, n=2, dim=2)
        pred_y_curvature = torch.diff(pred_yz, n=2, dim=2)
        true_y_curvature = torch.diff(true_yz, n=2, dim=2)

        pred_z_curvature = torch.diff(pred_xz, n=2, dim=3)
        true_z_curvature = torch.diff(true_xz, n=2, dim=3)

        return (
            F.l1_loss(pred_x_curvature, true_x_curvature) +
            F.l1_loss(pred_y_curvature, true_y_curvature) +
            0.5 * F.l1_loss(pred_z_curvature, true_z_curvature)
        ) / 2.5

    def _compute_boundary_suppression_loss(self, Y_pred, Y_true, grad_pred=None, grad_true=None):
        """Suppress artificial edge energy that later dominates derivative visualizations."""
        shell_weight = self.boundary_shell_mask.to(device=Y_pred.device, dtype=Y_pred.dtype)
        shell_weight = shell_weight / (shell_weight.mean() + 1e-8)

        field_loss = torch.mean(shell_weight * torch.abs(Y_pred - Y_true))

        if grad_pred is None:
            grad_pred = self.sobel_3d(Y_pred)
        if grad_true is None:
            grad_true = self.sobel_3d(Y_true)

        grad_loss = torch.mean(shell_weight * torch.abs(grad_pred - grad_true))
        return 0.65 * field_loss + 0.35 * grad_loss

    def _set_epoch_weights(self, epoch):
        """Use a physics-first warm-up so the model learns the gz field before strong localization constraints."""
        if self.loss_profile == 'localization_first':
            if epoch < 25:
                weights = {
                    'alpha': 0.18,
                    'beta': 0.12,
                    'gamma': 0.06,
                    'delta': 0.0002,
                    'epsilon': 0.08,
                    'zeta': 0.03,
                    'eta': 0.08,
                    'theta': 0.05,
                    'iota': 0.06,
                    'kappa': 0.05,
                    'amp_weight': 0.05,
                    'edge_weight': 0.06,
                    'lambda_body_aux': 0.08,
                    'lambda_center_aux': 0.05,
                    'lambda_signed_body_aux': 0.12,
                    'lambda_axis_aux': 0.06,
                }
            elif epoch < 80:
                weights = {
                    'alpha': 0.13,
                    'beta': 0.10,
                    'gamma': 0.05,
                    'delta': 0.0006,
                    'epsilon': 0.16,
                    'zeta': 0.08,
                    'eta': 0.10,
                    'theta': 0.10,
                    'iota': 0.10,
                    'kappa': 0.07,
                    'amp_weight': 0.14,
                    'edge_weight': 0.10,
                    'lambda_body_aux': 0.14,
                    'lambda_center_aux': 0.09,
                    'lambda_signed_body_aux': 0.18,
                    'lambda_axis_aux': 0.10,
                }
            else:
                weights = {
                    'alpha': 0.10,
                    'beta': 0.09,
                    'gamma': 0.05,
                    'delta': 0.0012,
                    'epsilon': 0.28,
                    'zeta': 0.12,
                    'eta': 0.10,
                    'theta': 0.16,
                    'iota': 0.12,
                    'kappa': 0.08,
                    'amp_weight': 0.20,
                    'edge_weight': 0.10,
                    'lambda_body_aux': 0.20,
                    'lambda_center_aux': 0.12,
                    'lambda_signed_body_aux': 0.24,
                    'lambda_axis_aux': 0.14,
                }
        elif self.loss_profile == 'balanced_physics':
            weights = {
                'alpha': self.alpha or 0.20,
                'beta': self.beta or 0.15,
                'gamma': self.gamma or 0.25,
                'delta': 0.02,
                'epsilon': 0.60,
                'zeta': 0.15,
                'eta': 0.18,
                'theta': 0.30,
                'iota': 0.25,
                'kappa': 0.10,
                'amp_weight': 0.22,
                'edge_weight': 0.08,
                'lambda_body_aux': 0.22,
                'lambda_center_aux': 0.14,
                'lambda_signed_body_aux': 0.28,
                'lambda_axis_aux': 0.18,
            }
        else:
            raise ValueError(
                f"Unsupported loss_profile '{self.loss_profile}'. "
                f"Available options: ['localization_first', 'balanced_physics']"
            )

        for key, value in weights.items():
            setattr(self, key, value)

    def _compute_normalized_center(self, mass):
        mass = mass / mass.sum(dim=(1, 2, 3), keepdim=True)
        _, nx, ny, nz = mass.shape

        x_coords = torch.linspace(0.0, 1.0, nx, device=mass.device, dtype=mass.dtype).view(1, nx, 1, 1)
        y_coords = torch.linspace(0.0, 1.0, ny, device=mass.device, dtype=mass.dtype).view(1, 1, ny, 1)
        z_coords = torch.linspace(0.0, 1.0, nz, device=mass.device, dtype=mass.dtype).view(1, 1, 1, nz)

        center_x = (mass * x_coords).sum(dim=(1, 2, 3))
        center_y = (mass * y_coords).sum(dim=(1, 2, 3))
        center_z = (mass * z_coords).sum(dim=(1, 2, 3))

        return torch.stack([center_x, center_y, center_z], dim=1)

    def update_weights(self, val_loss, epoch):
        if self.use_adaptive_weight_scheduler:
            self.alpha, self.beta, self.gamma = self.weight_scheduler.step(val_loss, epoch)
        else:
            self._set_epoch_weights(epoch)


class CosineAnnealingWarmRestarts:
    """Cosine-annealing learning-rate scheduler with warmup."""

    def __init__(self, optimizer, T_0=30, T_mult=2, eta_min=0, warmup_epochs=10):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.base_lr = optimizer.defaults['lr']

    def step(self):
        """Update and return the current learning rate."""
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            epoch_in_cycle = self.current_epoch - self.warmup_epochs
            T_cur = epoch_in_cycle % self.T_0
            lr = self.eta_min + (self.base_lr - self.eta_min) * (1 + np.cos(np.pi * T_cur / self.T_0)) / 2

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr
