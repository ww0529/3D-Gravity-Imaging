# 3D Gravity Imaging

This repository contains the reproducibility package for **3D Gravity Imaging via 3D FFT Gravity Forward Modeling and Deep Learning**. The project reconstructs a volumetric subsurface gravity field from surface gravity-gradient observations.

<p align="center">
  <img src="figures/manuscript_outputs/synthetic_prism_model/predicted_3d_gravity_field/predicted_3d_gravity_field.png" width="45%" alt="Predicted 3D gravity field for the synthetic prism model">
  <img src="figures/manuscript_outputs/vinton_field_data/gradient_magnitude_3d_field/gradient_magnitude_3d_field.png" width="45%" alt="Magnitude of the gradient of the 3D gravity field for Vinton field data">
</p>

## Repository Structure

```text
3D Gravity Imaging/
  Source Code/
    data_generator.py
    loss_functions_fully_corrected.py
    network.py
    train_fully_corrected.py
  Test Code/
    generate_manuscript_figures.py
    test_model.py
  data/
    synthetic_prism_model/
    synthetic_two_prisms_model/
    synthetic_irregular_model/
    vinton_field_data/
    manifest_selected.json
  figures/
    manuscript_outputs/
  best_model.pth
  requirements.txt
  README.md
```

## Description

The method integrates **3D FFT gravity forward modeling** with a deep learning network to learn the nonlinear mapping from surface gravity-gradient observations to a full **predicted 3D gravity field** inside the source volume.

The workflow is:

1. Generate diverse synthetic geological models and compute gravity and gravity-gradient fields using 3D FFT gravity forward modeling.
2. Train an encoder-decoder deep learning network to reconstruct the 3D gravity field from observed gravity-gradient data.
3. Use the **magnitude of the gradient of the 3D gravity field** to enhance structural boundaries and delineate anomalous bodies.
4. Validate the trained framework on three synthetic examples and airborne vertical gravity-gradient data from the **Vinton salt dome** area in Louisiana, USA.

The training objective includes depth weighting constraints, Poisson's equation constraints, and upward continuation constraints.

## Installation

Create a Python environment and install the required packages:

```bash
python -m pip install -r requirements.txt
```

For GPU acceleration, install the PyTorch build that matches your CUDA driver before installing the remaining dependencies.

## Usage

### Training

Run a short verification training job from `Source Code`:

```bash
cd "Source Code"
python train_fully_corrected.py ^
  --epochs 1 ^
  --train-samples 4 ^
  --val-samples 2 ^
  --batch-size 1 ^
  --model-capacity small ^
  --input-mode gzz ^
  --use-multimodal-stem false ^
  --seed 2026 ^
  --log-dir ../review_logs_quick ^
  --checkpoint-dir ../review_checkpoints_quick
```

On Linux or macOS, replace `^` with `\` for line continuation.

### Manuscript Figure Generation

Run the manuscript figure exporter from `Test Code`:

```bash
cd "Test Code"
python generate_manuscript_figures.py
```

The default inputs are:

- `best_model.pth`
- `data/synthetic_prism_model/`
- `data/synthetic_two_prisms_model/`
- `data/synthetic_irregular_model/`
- `data/vinton_field_data/vinton_observed_gzz.txt`

To generate only selected cases:

```bash
python generate_manuscript_figures.py --cases synthetic_prism_model vinton_field_data
```

## Run Test Codes

The packaged test code reproduces the paper-style visualization outputs:

- observed gravity-gradient maps
- predicted 3D gravity field isosurfaces
- 2D slices of the predicted gravity field
- 3D slices of the predicted gravity field
- magnitude of the gradient of the 3D gravity field
- synthetic model slices for the three synthetic examples

The output root is:

```text
figures/manuscript_outputs/
```

## Results

### Synthetic Prism Model

| Observed gravity-gradient | Predicted 3D gravity field | Magnitude of the gradient |
| --- | --- | --- |
| ![](figures/manuscript_outputs/synthetic_prism_model/observed_gravity_gradient/observed_gravity_gradient_gzz.png) | ![](figures/manuscript_outputs/synthetic_prism_model/predicted_3d_gravity_field/predicted_3d_gravity_field.png) | ![](figures/manuscript_outputs/synthetic_prism_model/gradient_magnitude_3d_field/gradient_magnitude_3d_field.png) |

| Synthetic model slice | Predicted gravity field slice | Gradient magnitude slice |
| --- | --- | --- |
| ![](figures/manuscript_outputs/synthetic_prism_model/synthetic_model/synthetic_model_slices/2d_slices/synthetic_model_2d_slice_xy.png) | ![](figures/manuscript_outputs/synthetic_prism_model/predicted_gravity_field_slices_2d/predicted_gravity_field_slice_xy.png) | ![](figures/manuscript_outputs/synthetic_prism_model/gradient_magnitude_slices_2d/gradient_magnitude_slice_xy.png) |

### Synthetic Two Prisms Model

| Observed gravity-gradient | Predicted 3D gravity field | Magnitude of the gradient |
| --- | --- | --- |
| ![](figures/manuscript_outputs/synthetic_two_prisms_model/observed_gravity_gradient/observed_gravity_gradient_gzz.png) | ![](figures/manuscript_outputs/synthetic_two_prisms_model/predicted_3d_gravity_field/predicted_3d_gravity_field.png) | ![](figures/manuscript_outputs/synthetic_two_prisms_model/gradient_magnitude_3d_field/gradient_magnitude_3d_field.png) |

| Synthetic model slice | Predicted gravity field slice | Gradient magnitude slice |
| --- | --- | --- |
| ![](figures/manuscript_outputs/synthetic_two_prisms_model/synthetic_model/synthetic_model_slices/2d_slices/synthetic_model_2d_slice_xy.png) | ![](figures/manuscript_outputs/synthetic_two_prisms_model/predicted_gravity_field_slices_2d/predicted_gravity_field_slice_xy.png) | ![](figures/manuscript_outputs/synthetic_two_prisms_model/gradient_magnitude_slices_2d/gradient_magnitude_slice_xy.png) |

### Synthetic Irregular Model

| Observed gravity-gradient | Predicted 3D gravity field | Magnitude of the gradient |
| --- | --- | --- |
| ![](figures/manuscript_outputs/synthetic_irregular_model/observed_gravity_gradient/observed_gravity_gradient_gzz.png) | ![](figures/manuscript_outputs/synthetic_irregular_model/predicted_3d_gravity_field/predicted_3d_gravity_field.png) | ![](figures/manuscript_outputs/synthetic_irregular_model/gradient_magnitude_3d_field/gradient_magnitude_3d_field.png) |

| Synthetic model slice | Predicted gravity field slice | Gradient magnitude slice |
| --- | --- | --- |
| ![](figures/manuscript_outputs/synthetic_irregular_model/synthetic_model/synthetic_model_slices/2d_slices/synthetic_model_2d_slice_xy.png) | ![](figures/manuscript_outputs/synthetic_irregular_model/predicted_gravity_field_slices_2d/predicted_gravity_field_slice_xy.png) | ![](figures/manuscript_outputs/synthetic_irregular_model/gradient_magnitude_slices_2d/gradient_magnitude_slice_xy.png) |

### Vinton Field Data

The Vinton example uses airborne vertical gravity-gradient data acquired over the Vinton salt dome area in Louisiana, USA.

| Observed airborne vertical gravity-gradient | Predicted 3D gravity field | Magnitude of the gradient |
| --- | --- | --- |
| ![](figures/manuscript_outputs/vinton_field_data/observed_gravity_gradient/observed_gravity_gradient_gzz.png) | ![](figures/manuscript_outputs/vinton_field_data/predicted_3d_gravity_field/predicted_3d_gravity_field.png) | ![](figures/manuscript_outputs/vinton_field_data/gradient_magnitude_3d_field/gradient_magnitude_3d_field.png) |

| Predicted gravity field slice | Gradient magnitude slice |
| --- | --- |
| ![](figures/manuscript_outputs/vinton_field_data/predicted_gravity_field_slices_2d/predicted_gravity_field_slice_xy.png) | ![](figures/manuscript_outputs/vinton_field_data/gradient_magnitude_slices_2d/gradient_magnitude_slice_xy.png) |

## Data Cases

- `synthetic_prism_model`: synthetic prism model with observed gravity-gradient data containing Gaussian noise.
- `synthetic_two_prisms_model`: synthetic two prisms model with positive and negative density contrasts.
- `synthetic_irregular_model`: synthetic irregular model for complex inclined anomalous bodies.
- `vinton_field_data`: airborne vertical gravity-gradient data from the Vinton salt dome area.

## Notes

- `best_model.pth` is larger than GitHub's normal 100 MB file limit and should be tracked with Git LFS.
- The visual outputs in `figures/manuscript_outputs/` follow the paper terminology: observed gravity-gradient, predicted 3D gravity field, and magnitude of the gradient of the 3D gravity field.

## Citation

If you use this package, please cite the manuscript associated with `MS(1).doc`:

```text
3D Gravity Imaging via 3D FFT Gravity Forward Modeling and Deep Learning
```
