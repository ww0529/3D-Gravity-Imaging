"""
Training entry point for the 3D gravity-imaging neural network.

The script trains the localization-aware hybrid U-Net with the physics-guided
losses used in the manuscript. Command-line options are provided so reviewers
can reproduce short verification runs or full training runs without editing the
source code.
"""

import argparse
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from data_generator import create_dataloader, get_input_channel_names
from network import LocalizationAwareHybridUNet
from loss_functions_fully_corrected import FullyCorrectHybridPhysicsLoss, CosineAnnealingWarmRestarts


DEFAULT_CONFIG = {
    'batch_size': 2,
    'num_epochs': 300,
    'learning_rate': 3e-4,
    'weight_decay': 1e-5,
    'num_workers': 0,
    'num_train_samples': 1000,
    'num_val_samples': 200,
    'save_interval': 20,
    'noise_level': 0.02,
    'add_regional': True,
    'input_mode': 'gzz',
    'model_capacity': 'large',
    'use_multimodal_stem': False,
    'use_location_targets': True,
    'loss_profile': 'localization_first',
    'selection_metric': 'localization_score',
    'depth_sampling_mode': 'focused_shallow',
    'structured_case_probability': 0.60,
    'tilted_pair_probability': 0.78,
    'wide_pair_probability': 0.82,
    'seed': 2026,
    'deterministic': False,
    'log_dir': None,
    'checkpoint_dir': None
}


def set_reproducible_seed(seed, deterministic=False):
    """Seed Python, NumPy, and PyTorch random number generators."""
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def str_to_bool(value):
    """Parse command-line boolean values in a reviewer-friendly way."""
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if value in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got '{value}'.")


def load_config_file(config_path):
    """Load a JSON configuration file and return an empty dict when not used."""
    if config_path is None:
        return {}
    with open(config_path, 'r', encoding='utf-8') as file_obj:
        return json.load(file_obj)


def build_arg_parser():
    """Create the command-line interface used for reproducible runs."""
    parser = argparse.ArgumentParser(
        description="Train the 3D gravity-imaging network with reproducible options."
    )
    parser.add_argument('--config', type=str, default=None, help='Optional JSON config file.')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs.')
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--train-samples', type=int, default=None)
    parser.add_argument('--val-samples', type=int, default=None)
    parser.add_argument('--noise-level', type=float, default=None)
    parser.add_argument('--add-regional', type=str_to_bool, default=None)
    parser.add_argument(
        '--input-mode',
        type=str,
        default=None,
        choices=['gzz', 'gz_amp', 'gz_gzz', 'gz_gzz_amp']
    )
    parser.add_argument('--model-capacity', type=str, default=None, choices=['small', 'medium', 'large'])
    parser.add_argument('--use-multimodal-stem', type=str_to_bool, default=None)
    parser.add_argument('--use-location-targets', type=str_to_bool, default=None)
    parser.add_argument(
        '--loss-profile',
        type=str,
        default=None,
        choices=['localization_first', 'balanced_physics']
    )
    parser.add_argument(
        '--selection-metric',
        type=str,
        default=None,
        choices=['localization_score', 'val_loss']
    )
    parser.add_argument(
        '--depth-sampling-mode',
        type=str,
        default=None,
        choices=['balanced', 'shallow_biased', 'focused_shallow']
    )
    parser.add_argument('--structured-case-probability', type=float, default=None)
    parser.add_argument('--tilted-pair-probability', type=float, default=None)
    parser.add_argument('--wide-pair-probability', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--deterministic', type=str_to_bool, default=None)
    parser.add_argument('--log-dir', type=str, default=None)
    parser.add_argument('--checkpoint-dir', type=str, default=None)
    return parser


def config_from_args(args):
    """Merge defaults, an optional JSON config, and command-line overrides."""
    config = DEFAULT_CONFIG.copy()
    config.update(load_config_file(args.config))

    arg_to_config = {
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'num_workers': args.num_workers,
        'num_train_samples': args.train_samples,
        'num_val_samples': args.val_samples,
        'noise_level': args.noise_level,
        'add_regional': args.add_regional,
        'input_mode': args.input_mode,
        'model_capacity': args.model_capacity,
        'use_multimodal_stem': args.use_multimodal_stem,
        'use_location_targets': args.use_location_targets,
        'loss_profile': args.loss_profile,
        'selection_metric': args.selection_metric,
        'depth_sampling_mode': args.depth_sampling_mode,
        'structured_case_probability': args.structured_case_probability,
        'tilted_pair_probability': args.tilted_pair_probability,
        'wide_pair_probability': args.wide_pair_probability,
        'seed': args.seed,
        'deterministic': args.deterministic,
        'log_dir': args.log_dir,
        'checkpoint_dir': args.checkpoint_dir,
    }
    config.update({key: value for key, value in arg_to_config.items() if value is not None})
    return config


class FullyCorrectGravityTrainer:
    """Trainer for 3D gravity-field reconstruction."""

    def __init__(self, config_dict=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")

        self.config = DEFAULT_CONFIG.copy()

        if config_dict:
            self.config.update(config_dict)

        set_reproducible_seed(
            self.config.get('seed'),
            deterministic=self.config.get('deterministic', False)
        )

        self.input_channel_names = get_input_channel_names(self.config['input_mode'])
        if self.config['use_multimodal_stem'] is None:
            self.config['use_multimodal_stem'] = (
                len(self.input_channel_names) > 1 and 'gzz' in self.input_channel_names
            )

        if self.config['log_dir'] is None:
            self.config['log_dir'] = (
                f"./logs_localization_{self.config['model_capacity']}_{self.config['input_mode']}"
            )
        if self.config['checkpoint_dir'] is None:
            self.config['checkpoint_dir'] = (
                f"./checkpoints_localization_{self.config['model_capacity']}_{self.config['input_mode']}"
            )

        Path(self.config['log_dir']).mkdir(exist_ok=True)
        Path(self.config['checkpoint_dir']).mkdir(exist_ok=True)

        self.model = LocalizationAwareHybridUNet(
            capacity=self.config['model_capacity'],
            input_channels=len(self.input_channel_names),
            input_mode=self.config['input_mode'],
            use_multimodal_stem=self.config['use_multimodal_stem']
        ).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        # Use the physics-guided loss reported in the manuscript.
        self.loss_fn = FullyCorrectHybridPhysicsLoss(
            input_mode=self.config['input_mode'],
            loss_profile=self.config['loss_profile']
        ).to(self.device)

        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=30,
            T_mult=2,
            eta_min=1e-6,
            warmup_epochs=10
        )

        self.writer = SummaryWriter(self.config['log_dir'])

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_l1': [],
            'train_grad': [],
            'train_phys': [],
            'train_surface_aux': [],
            'train_location': [],
            'train_depth_profile': [],
            'train_artifact': [],
            'train_amplitude': [],
            'train_boundary': [],
            'train_poisson': [],
            'train_body_aux': [],
            'train_center_aux': [],
            'train_signed_body_aux': [],
            'train_axis_aux': [],
            'val_projection': [],
            'val_center': [],
            'val_location': [],
            'val_surface_aux': [],
            'val_depth_profile': [],
            'val_artifact': [],
            'val_amplitude': [],
            'val_boundary': [],
            'val_body_aux': [],
            'val_center_aux': [],
            'val_signed_body_aux': [],
            'val_axis_aux': [],
            'selection_score': [],
            'learning_rate': [],
            'best_val_loss': float('inf'),
            'best_selection_score': float('inf')
        }

    def _extract_location_targets(self, meta):
        if not isinstance(meta, dict):
            return None
        return {
            key: value.to(self.device)
            for key, value in meta.items()
            if key != 'target_scale'
        }

    def _extract_aux_predictions(self, output_dict):
        return {
            key: output_dict[key]
            for key in (
                'body_mask',
                'center_heatmap',
                'positive_body_mask',
                'negative_body_mask',
                'axis_heatmap',
            )
            if key in output_dict
        }

    def train_epoch(self, train_loader, epoch):
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        loss_l1_sum = 0
        loss_grad_sum = 0
        loss_phys_sum = 0
        loss_surface_aux_sum = 0
        loss_location_sum = 0
        loss_depth_profile_sum = 0
        loss_artifact_sum = 0
        loss_amplitude_sum = 0
        loss_boundary_sum = 0
        loss_poisson_sum = 0
        loss_body_aux_sum = 0
        loss_center_aux_sum = 0
        loss_signed_body_aux_sum = 0
        loss_axis_aux_sum = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch_idx, batch_data in enumerate(pbar):
            # Keep compatibility with older batches that included a separate amplitude tensor.
            location_targets = None
            if len(batch_data) == 4:
                X_batch, Y_batch, A_batch, meta = batch_data
            else:
                X_batch, Y_batch, meta = batch_data
                A_batch = None

            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)
            location_targets = self._extract_location_targets(meta)

            self.optimizer.zero_grad()
            output_dict = self.model(X_batch)
            Y_pred = output_dict['output']
            aux_predictions = self._extract_aux_predictions(output_dict)

            loss_dict = self.loss_fn(
                Y_pred,
                Y_batch,
                X_batch,
                epoch=epoch,
                val_loss=None,
                location_targets=location_targets,
                aux_predictions=aux_predictions
            )
            loss = loss_dict['total']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            loss_l1_sum += loss_dict['l1'].item()
            loss_grad_sum += loss_dict['grad'].item()
            loss_phys_sum += loss_dict['phys'].item()
            loss_surface_aux_sum += loss_dict['surface_aux'].item()
            loss_location_sum += loss_dict['location'].item()
            loss_depth_profile_sum += loss_dict['depth_profile'].item()
            loss_artifact_sum += loss_dict['artifact'].item()
            loss_amplitude_sum += loss_dict['amplitude'].item()
            loss_boundary_sum += loss_dict['boundary'].item()
            loss_poisson_sum += loss_dict['poisson'].item()
            loss_body_aux_sum += loss_dict['body_aux'].item()
            loss_center_aux_sum += loss_dict['center_aux'].item()
            loss_signed_body_aux_sum += loss_dict['signed_body_aux'].item()
            loss_axis_aux_sum += loss_dict['axis_aux'].item()

            pbar.set_postfix({
                'loss': loss.item(),
                'l1': loss_dict['l1'].item(),
                'grad': loss_dict['grad'].item(),
                'phys': loss_dict['phys'].item(),
                'aux': loss_dict['surface_aux'].item(),
                'loc': loss_dict['location'].item(),
                'depth': loss_dict['depth_profile'].item(),
                'art': loss_dict['artifact'].item(),
                'amp': loss_dict['amplitude'].item(),
                'bnd': loss_dict['boundary'].item(),
                'poisson': loss_dict['poisson'].item(),
                'body': loss_dict['body_aux'].item(),
                'ctrh': loss_dict['center_aux'].item(),
                'sgnb': loss_dict['signed_body_aux'].item(),
                'axis': loss_dict['axis_aux'].item(),
            })

        lr = self.scheduler.step()

        avg_loss = total_loss / len(train_loader)
        avg_l1 = loss_l1_sum / len(train_loader)
        avg_grad = loss_grad_sum / len(train_loader)
        avg_phys = loss_phys_sum / len(train_loader)
        avg_surface_aux = loss_surface_aux_sum / len(train_loader)
        avg_location = loss_location_sum / len(train_loader)
        avg_depth_profile = loss_depth_profile_sum / len(train_loader)
        avg_artifact = loss_artifact_sum / len(train_loader)
        avg_amplitude = loss_amplitude_sum / len(train_loader)
        avg_boundary = loss_boundary_sum / len(train_loader)
        avg_poisson = loss_poisson_sum / len(train_loader)
        avg_body_aux = loss_body_aux_sum / len(train_loader)
        avg_center_aux = loss_center_aux_sum / len(train_loader)
        avg_signed_body_aux = loss_signed_body_aux_sum / len(train_loader)
        avg_axis_aux = loss_axis_aux_sum / len(train_loader)

        self.history['train_loss'].append(avg_loss)
        self.history['train_l1'].append(avg_l1)
        self.history['train_grad'].append(avg_grad)
        self.history['train_phys'].append(avg_phys)
        self.history['train_surface_aux'].append(avg_surface_aux)
        self.history['train_location'].append(avg_location)
        self.history['train_depth_profile'].append(avg_depth_profile)
        self.history['train_artifact'].append(avg_artifact)
        self.history['train_amplitude'].append(avg_amplitude)
        self.history['train_boundary'].append(avg_boundary)
        self.history['train_poisson'].append(avg_poisson)
        self.history['train_body_aux'].append(avg_body_aux)
        self.history['train_center_aux'].append(avg_center_aux)
        self.history['train_signed_body_aux'].append(avg_signed_body_aux)
        self.history['train_axis_aux'].append(avg_axis_aux)
        self.history['learning_rate'].append(lr)

        self.writer.add_scalar('Loss/train_total', avg_loss, epoch)
        self.writer.add_scalar('Loss/train_l1', avg_l1, epoch)
        self.writer.add_scalar('Loss/train_grad', avg_grad, epoch)
        self.writer.add_scalar('Loss/train_phys', avg_phys, epoch)
        self.writer.add_scalar('Loss/train_surface_aux', avg_surface_aux, epoch)
        self.writer.add_scalar('Loss/train_location', avg_location, epoch)
        self.writer.add_scalar('Loss/train_depth_profile', avg_depth_profile, epoch)
        self.writer.add_scalar('Loss/train_artifact', avg_artifact, epoch)
        self.writer.add_scalar('Loss/train_amplitude', avg_amplitude, epoch)
        self.writer.add_scalar('Loss/train_boundary', avg_boundary, epoch)
        self.writer.add_scalar('Loss/train_poisson', avg_poisson, epoch)
        self.writer.add_scalar('Loss/train_body_aux', avg_body_aux, epoch)
        self.writer.add_scalar('Loss/train_center_aux', avg_center_aux, epoch)
        self.writer.add_scalar('Loss/train_signed_body_aux', avg_signed_body_aux, epoch)
        self.writer.add_scalar('Loss/train_axis_aux', avg_axis_aux, epoch)
        self.writer.add_scalar('Learning_Rate', lr, epoch)

        return avg_loss

    @torch.no_grad()
    def validate(self, val_loader, epoch):
        """Evaluate one validation pass."""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        total_rmse = 0
        total_projection = 0
        total_center = 0
        total_location = 0
        total_surface_aux = 0
        total_depth_profile = 0
        total_artifact = 0
        total_amplitude = 0
        total_boundary = 0
        total_body_aux = 0
        total_center_aux = 0
        total_signed_body_aux = 0
        total_axis_aux = 0

        for batch_data in val_loader:
            # Keep compatibility with older batches that included a separate amplitude tensor.
            location_targets = None
            if len(batch_data) == 4:
                X_batch, Y_batch, A_batch, meta = batch_data
            else:
                X_batch, Y_batch, meta = batch_data
                A_batch = None

            X_batch = X_batch.to(self.device)
            Y_batch = Y_batch.to(self.device)
            location_targets = self._extract_location_targets(meta)

            output_dict = self.model(X_batch)
            Y_pred = output_dict['output']
            aux_predictions = {
                key: output_dict[key]
                for key in ('body_mask', 'center_heatmap', 'positive_body_mask', 'negative_body_mask', 'axis_heatmap')
                if key in output_dict
            }
            aux_predictions = self._extract_aux_predictions(output_dict)
            loss_dict = self.loss_fn(
                Y_pred,
                Y_batch,
                X_batch,
                epoch=epoch,
                val_loss=None,
                location_targets=location_targets,
                aux_predictions=aux_predictions
            )
            total_loss += loss_dict['total'].item()
            total_projection += loss_dict['projection'].item()
            total_center += loss_dict['center'].item()
            total_location += loss_dict['location'].item()
            total_surface_aux += loss_dict['surface_aux'].item()
            total_depth_profile += loss_dict['depth_profile'].item()
            total_artifact += loss_dict['artifact'].item()
            total_amplitude += loss_dict['amplitude'].item()
            total_boundary += loss_dict['boundary'].item()
            total_body_aux += loss_dict['body_aux'].item()
            total_center_aux += loss_dict['center_aux'].item()
            total_signed_body_aux += loss_dict['signed_body_aux'].item()
            total_axis_aux += loss_dict['axis_aux'].item()

            mae = torch.mean(torch.abs(Y_pred - Y_batch)).item()
            rmse = torch.sqrt(torch.mean((Y_pred - Y_batch)**2)).item()
            total_mae += mae
            total_rmse += rmse

        avg_loss = total_loss / len(val_loader)
        avg_mae = total_mae / len(val_loader)
        avg_rmse = total_rmse / len(val_loader)
        avg_projection = total_projection / len(val_loader)
        avg_center = total_center / len(val_loader)
        avg_location = total_location / len(val_loader)
        avg_surface_aux = total_surface_aux / len(val_loader)
        avg_depth_profile = total_depth_profile / len(val_loader)
        avg_artifact = total_artifact / len(val_loader)
        avg_amplitude = total_amplitude / len(val_loader)
        avg_boundary = total_boundary / len(val_loader)
        avg_body_aux = total_body_aux / len(val_loader)
        avg_center_aux = total_center_aux / len(val_loader)
        avg_signed_body_aux = total_signed_body_aux / len(val_loader)
        avg_axis_aux = total_axis_aux / len(val_loader)
        selection_score = (
            0.16 * avg_projection +
            0.18 * avg_location +
            0.11 * avg_center +
            0.11 * avg_depth_profile +
            0.10 * avg_amplitude +
            0.08 * avg_boundary +
            0.07 * avg_body_aux +
            0.05 * avg_center_aux +
            0.08 * avg_signed_body_aux +
            0.04 * avg_axis_aux +
            0.01 * avg_artifact +
            0.01 * avg_surface_aux
        )

        self.history['val_loss'].append(avg_loss)
        self.history['val_projection'].append(avg_projection)
        self.history['val_center'].append(avg_center)
        self.history['val_location'].append(avg_location)
        self.history['val_surface_aux'].append(avg_surface_aux)
        self.history['val_depth_profile'].append(avg_depth_profile)
        self.history['val_artifact'].append(avg_artifact)
        self.history['val_amplitude'].append(avg_amplitude)
        self.history['val_boundary'].append(avg_boundary)
        self.history['val_body_aux'].append(avg_body_aux)
        self.history['val_center_aux'].append(avg_center_aux)
        self.history['val_signed_body_aux'].append(avg_signed_body_aux)
        self.history['val_axis_aux'].append(avg_axis_aux)
        self.history['selection_score'].append(selection_score)
        self.writer.add_scalar('Loss/val_total', avg_loss, epoch)
        self.writer.add_scalar('Loss/val_projection', avg_projection, epoch)
        self.writer.add_scalar('Loss/val_center', avg_center, epoch)
        self.writer.add_scalar('Loss/val_location', avg_location, epoch)
        self.writer.add_scalar('Loss/val_surface_aux', avg_surface_aux, epoch)
        self.writer.add_scalar('Loss/val_depth_profile', avg_depth_profile, epoch)
        self.writer.add_scalar('Loss/val_artifact', avg_artifact, epoch)
        self.writer.add_scalar('Loss/val_amplitude', avg_amplitude, epoch)
        self.writer.add_scalar('Loss/val_boundary', avg_boundary, epoch)
        self.writer.add_scalar('Loss/val_body_aux', avg_body_aux, epoch)
        self.writer.add_scalar('Loss/val_center_aux', avg_center_aux, epoch)
        self.writer.add_scalar('Loss/val_signed_body_aux', avg_signed_body_aux, epoch)
        self.writer.add_scalar('Loss/val_axis_aux', avg_axis_aux, epoch)
        self.writer.add_scalar('Metrics/selection_score', selection_score, epoch)
        self.writer.add_scalar('Metrics/val_mae', avg_mae, epoch)
        self.writer.add_scalar('Metrics/val_rmse', avg_rmse, epoch)

        return {
            'loss': avg_loss,
            'mae': avg_mae,
            'rmse': avg_rmse,
            'projection': avg_projection,
            'center': avg_center,
            'location': avg_location,
            'surface_aux': avg_surface_aux,
            'depth_profile': avg_depth_profile,
            'artifact': avg_artifact,
            'amplitude': avg_amplitude,
            'boundary': avg_boundary,
            'body_aux': avg_body_aux,
            'center_aux': avg_center_aux,
            'signed_body_aux': avg_signed_body_aux,
            'axis_aux': avg_axis_aux,
            'selection_score': selection_score,
        }

    def train(self):
        """Run the full training workflow."""
        self._write_run_config()
        print(f"Input mode: {self.config['input_mode']} -> {self.input_channel_names}")
        print(f"Model capacity: {self.config['model_capacity']}")
        print(f"Multimodal stem: {'enabled' if self.model.use_multimodal_stem else 'disabled'}")
        print(f"Localization targets: {'enabled' if self.config['use_location_targets'] else 'disabled'}")
        print(f"Loss profile: {self.config['loss_profile']}")
        print(f"Selection metric: {self.config['selection_metric']}")
        print(f"Depth sampling mode: {self.config['depth_sampling_mode']}")
        print(f"Seed: {self.config.get('seed')}")
        print(f"Deterministic algorithms: {self.config.get('deterministic', False)}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        print(f"Training configuration: {self.config}")
        print(
            "structured bars: "
            f"p={self.config['structured_case_probability']:.2f}, "
            f"tilted_pair={self.config['tilted_pair_probability']:.2f}, "
            f"wide_pair={self.config['wide_pair_probability']:.2f}"
        )
        print("=" * 60)
        print("Physics-guided training terms:")
        print("- Upward-continuation surface consistency over all depth levels")
        print("- Focal weighting for high-amplitude target regions")
        print("- Depth-profile compensation to reduce shallow artifacts")
        print("- Anti-stripe regularization for vertical-slice artifacts")
        print("- Body-mask auxiliary supervision")
        print("- Center-heatmap auxiliary supervision")
        print("- Poisson/Laplacian regularization")
        print("- Optional adaptive loss-weight scheduling")
        print("=" * 60)

        train_loader = create_dataloader(
            batch_size=self.config['batch_size'],
            num_samples=self.config['num_train_samples'],
            num_workers=self.config['num_workers'],
            noise_level=self.config['noise_level'],
            add_regional=self.config['add_regional'],
            input_mode=self.config['input_mode'],
            use_location_targets=self.config['use_location_targets'],
            depth_sampling_mode=self.config['depth_sampling_mode'],
            structured_case_probability=self.config['structured_case_probability'],
            tilted_pair_probability=self.config['tilted_pair_probability'],
            wide_pair_probability=self.config['wide_pair_probability']
        )

        val_loader = create_dataloader(
            batch_size=self.config['batch_size'],
            num_samples=self.config['num_val_samples'],
            num_workers=self.config['num_workers'],
            noise_level=self.config['noise_level'],
            add_regional=self.config['add_regional'],
            input_mode=self.config['input_mode'],
            use_location_targets=self.config['use_location_targets'],
            depth_sampling_mode=self.config['depth_sampling_mode'],
            structured_case_probability=self.config['structured_case_probability'],
            tilted_pair_probability=self.config['tilted_pair_probability'],
            wide_pair_probability=self.config['wide_pair_probability']
        )

        best_val_loss = float('inf')
        best_selection_score = float('inf')

        for epoch in range(self.config['num_epochs']):
            train_loss = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader, epoch)
            val_loss = val_metrics['loss']
            val_mae = val_metrics['mae']
            val_rmse = val_metrics['rmse']
            selection_score = val_metrics['selection_score']
            self.loss_fn.update_weights(val_loss, epoch)

            print(f"Epoch {epoch+1}/{self.config['num_epochs']} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"MAE: {val_mae:.6f} | RMSE: {val_rmse:.6f} | "
                  f"Proj: {val_metrics['projection']:.6f} | "
                  f"Loc: {val_metrics['location']:.6f} | "
                  f"Ctr: {val_metrics['center']:.6f} | "
                  f"Depth: {val_metrics['depth_profile']:.6f} | "
                  f"Art: {val_metrics['artifact']:.6f} | "
                  f"Amp: {val_metrics['amplitude']:.6f} | "
                  f"Bnd: {val_metrics['boundary']:.6f} | "
                  f"Body: {val_metrics['body_aux']:.6f} | "
                  f"CtrH: {val_metrics['center_aux']:.6f} | "
                  f"SBody: {val_metrics['signed_body_aux']:.6f} | "
                  f"Axis: {val_metrics['axis_aux']:.6f} | "
                  f"Score: {selection_score:.6f} | "
                  f"eps={self.loss_fn.epsilon:.3f} theta={self.loss_fn.theta:.3f} "
                  f"zeta={self.loss_fn.zeta:.3f} iota={self.loss_fn.iota:.3f} "
                  f"kappa={self.loss_fn.kappa:.3f} "
                  f"ampw={self.loss_fn.amp_weight:.3f} edgew={self.loss_fn.edge_weight:.3f} "
                  f"bodyw={self.loss_fn.lambda_body_aux:.3f} ctrw={self.loss_fn.lambda_center_aux:.3f} "
                  f"sgbw={self.loss_fn.lambda_signed_body_aux:.3f} axisw={self.loss_fn.lambda_axis_aux:.3f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.history['best_val_loss'] = best_val_loss
            is_better_model = False
            if self.config['selection_metric'] == 'localization_score':
                if selection_score < best_selection_score:
                    best_selection_score = selection_score
                    self.history['best_selection_score'] = best_selection_score
                    is_better_model = True
            else:
                if val_loss <= best_val_loss:
                    is_better_model = True

            if is_better_model:
                self.save_checkpoint(epoch, is_best=True, selection_score=selection_score)
                print(
                    f"Saved best model "
                    f"(Score: {selection_score:.6f}, Val Loss: {val_loss:.6f})"
                )

        self.writer.close()
        print("Training complete.")
        self.plot_training_history()

    def _write_run_config(self):
        """Save the resolved configuration for reproducibility."""
        config_path = Path(self.config['checkpoint_dir']) / 'run_config.json'
        with open(config_path, 'w', encoding='utf-8') as file_obj:
            json.dump(self.config, file_obj, indent=2, sort_keys=True)

    def save_checkpoint(self, epoch, is_best=False, selection_score=None):
        """Save a PyTorch checkpoint with architecture and training metadata."""
        checkpoint = {
            'epoch': epoch,
            'config': self.config,
            'seed': self.config.get('seed'),
            'model_capacity': self.config['model_capacity'],
            'input_mode': self.config['input_mode'],
            'use_multimodal_stem': self.model.use_multimodal_stem,
            'use_location_targets': self.config['use_location_targets'],
            'loss_profile': self.config['loss_profile'],
            'selection_metric': self.config['selection_metric'],
            'depth_sampling_mode': self.config['depth_sampling_mode'],
            'structured_case_probability': self.config['structured_case_probability'],
            'tilted_pair_probability': self.config['tilted_pair_probability'],
            'wide_pair_probability': self.config['wide_pair_probability'],
            'selection_score': selection_score,
            'model_architecture_version': 'localization_multimodal_stem_antistripe_v8_signed_axis_structuredbars',
            'input_normalization_mode': 'fixed_physical_v1',
            'input_channel_names': self.input_channel_names,
            'input_channels': len(self.input_channel_names),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }

        if is_best:
            path = Path(self.config['checkpoint_dir']) / 'best_model.pth'
        else:
            path = Path(self.config['checkpoint_dir']) / f'checkpoint_epoch_{epoch}.pth'

        torch.save(checkpoint, path)

    def plot_training_history(self):
        """Write a compact training-history figure to the log directory."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        axes[0, 0].plot(self.history['train_loss'], label='Train', marker='o')
        axes[0, 0].plot(self.history['val_loss'], label='Val', marker='s')
        if self.history['selection_score']:
            axes[0, 0].plot(self.history['selection_score'], label='Selection Score', marker='^')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Total loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(self.history['train_l1'], label='L1 Loss', marker='o')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('L1 Loss')
        axes[0, 1].set_title('L1 regression loss')
        axes[0, 1].grid(True)

        axes[0, 2].plot(self.history['train_grad'], label='Grad Loss', marker='o')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Gradient Loss')
        axes[0, 2].set_title('Gradient loss')
        axes[0, 2].grid(True)

        axes[1, 0].plot(self.history['train_phys'], label='Gz Surface Loss', marker='o')
        if self.history['train_surface_aux']:
            axes[1, 0].plot(self.history['train_surface_aux'], label='Aux Surface Loss', marker='s')
        if self.history['train_location']:
            axes[1, 0].plot(self.history['train_location'], label='Location Loss', marker='^')
        if self.history['train_artifact']:
            axes[1, 0].plot(self.history['train_artifact'], label='Artifact Loss', marker='d')
        if self.history['train_boundary']:
            axes[1, 0].plot(self.history['train_boundary'], label='Boundary Loss', marker='x')
        if self.history['train_body_aux']:
            axes[1, 0].plot(self.history['train_body_aux'], label='Body Aux', marker='v')
        if self.history['train_signed_body_aux']:
            axes[1, 0].plot(self.history['train_signed_body_aux'], label='Signed Body Aux', marker='<')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Physics Loss')
        axes[1, 0].set_title('Surface and auxiliary losses')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        axes[1, 1].plot(self.history['train_poisson'], label='Poisson Loss', marker='o')
        if self.history['train_depth_profile']:
            axes[1, 1].plot(self.history['train_depth_profile'], label='Train Depth Loss', marker='s')
        if self.history['val_depth_profile']:
            axes[1, 1].plot(self.history['val_depth_profile'], label='Val Depth Loss', marker='^')
        if self.history['train_amplitude']:
            axes[1, 1].plot(self.history['train_amplitude'], label='Train Amp Loss', marker='d')
        if self.history['val_amplitude']:
            axes[1, 1].plot(self.history['val_amplitude'], label='Val Amp Loss', marker='x')
        if self.history['train_center_aux']:
            axes[1, 1].plot(self.history['train_center_aux'], label='Train Center Aux', marker='v')
        if self.history['val_center_aux']:
            axes[1, 1].plot(self.history['val_center_aux'], label='Val Center Aux', marker='<')
        if self.history['train_axis_aux']:
            axes[1, 1].plot(self.history['train_axis_aux'], label='Train Axis Aux', marker='>')
        if self.history['val_axis_aux']:
            axes[1, 1].plot(self.history['val_axis_aux'], label='Val Axis Aux', marker='1')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Constraint Loss')
        axes[1, 1].set_title('Depth and Poisson constraints')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        axes[1, 2].plot(self.history['learning_rate'], label='Learning Rate', marker='o')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Learning Rate')
        axes[1, 2].set_title('Learning-rate schedule')
        axes[1, 2].set_yscale('log')
        axes[1, 2].grid(True)

        plt.tight_layout()
        plt.savefig(Path(self.config['log_dir']) / 'training_history.png', dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    trainer = FullyCorrectGravityTrainer(config_from_args(args))
    trainer.train()
