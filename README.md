# VFINN

**VFINN**: Video Flow-Informed Neural Network 

## Description

**VFINN** is a deep learning model designed for high-density particle tracking using dense optical flow estimation. The model estimates particle velocity fields without requiring ground-truth data, enabling robust tracking in complex biological systems.

VFINN leverages a flow-informed loss function for dense optical flow estimation, allowing pixel-level velocity field prediction to track high-density particles. The method incorporates the following features:

- **High-density tracking**: Achieves robust tracking at densities up to 1.0 localizations μm⁻² per frame.
- **Pre-trained model**: A large pre-trained vision model fine-tuned on a small set of SPT samples, with a diffusion loss to enhance smoothness.
- **Application**: Suitable for applications like high density tracking, fluorescence blinking and liquid-liquid phase separation (LLPS).

Details of the method are described in Zhang et al. ("Large vision model supported tracking of high-density particles based on optical flow learning").

## Installation and dependencies

### Downloading the source

The official distribution is on GitHub, and you can clone the repository using

```
git clone https://github.com/EdwardZX/VFINN.git
```

### Dependencies

**VPINN** is tested on PyTorch 2.0.0 with Python 3.8 (Ubuntu 20.04) and CUDA 11.8.

Install PyTorch and Related Packages:

```
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

Install other dependencies (NumPy, Matplotlib, Pandas, etc.):

```
pip install -r requirements.txt
```

Download crocoV2 pretrained weights from or visit the Zenodo repository linked in our paper:

```
wget https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo_V2_ViTLarge_BaseDecoder.pth -P pretrained_models/
```

## Usage

### Prepare the Dataset

Ensure your dataset includes .tif image pairs (for particle tracking) and .csv files (for localization center data, optional for piv), as shown in the example below:

```
data/
├── <path_to_data>/  # example your tracking tif and csv data files here
│   ├── example_00001_img1.tif
│   ├── example_00001_img2.tif
│   ├── example_00001_img1.csv
│   ├── example_00001_img2.csv
│   ├── example_00002_img1.tif
│   ├── example_00002_img2.tif
│   ├── example_00002_img1.csv
│   ├── example_00002_img2.csv
│   └── ...

```

To ensure the dataset is properly formatted for VFINN, make sure your CSV follows the same structure with each entry containing:

1. **POSITION_T**: The time step (frame number).
2. **POSITION_X**: The X-coordinate of the particle.
3. **POSITION_Y**: The Y-coordinate of the particle.

Create an **annotation file** in the `assets/annotations/` folder. This text file (e.g., `<path_to_data>_annotation.txt`) should specify the path to your dataset folder.

Example:

```
data/<path_to_data>
```

### Run the VFINN

**Fine-tune the Model:** To fine-tune the model with your dataset, use the following command:

```
python main_spt_flow_diffL.py --filename <path_to_data>_Ld --epochs 50 --beta_diffL 1.0 --beta_photo 1e-3  --is_spt_pt --is_train
```

Replace `<path_to_data>` with the path to your dataset folder.

**Track Particles:** Once the model is trained, you can track particles in your data by running:

```
python main_spt_flow_diffL.py --filename <path_to_data>_Ld --is_spt_pt
```

The results of initial single-particle tracking will be saved in the `model_dir` folder.

### Help

For details about additional arguments and options, run:

```
python main_spt_flow_diffL.py --help
```

### Example Colab

For a step-by-step guide, use the [VFINN Colab Notebook](https://colab.research.google.com/drive/1eO5eFm4-NIDArsWpcBN9y_YEUUxd4TPV?usp=sharing).

## Reference

TODO: add citeable reference

## License

**VFINN** is [MIT-licensed](https://opensource.org/licenses/MIT); refer to the LICENSE file for more information.
