# Processing-Aware Generative Inverse Design of Steels from Strength and Hardness Targets — Code

This repository contains the code used in the paper "Processing-Aware Generative Inverse Design of Fe-Based Alloys". The code implements a shared-attributed VAE and a diffusion/denoising model that operate in a learned latent space to perform inverse design for Fe-based alloys.

**Dataset**
- **ANSYS Granta (proprietary):** The main alloy dataset used in the experiments is the Fe alloys ANSYS GRANTA dataset. This dataset is proprietary and owned by ANSYS and is not included in this repository. Users must obtain access to the ANSYS dataset directly from ANSYS/Granta.

**Included files**
- `denoise_model.py`: diffusion / denoising model utilities and network definitions.
- `VAE_Model.py`: shared-attributed VAE and transformer-based encoder/decoder code.
- `Token_generation.py`: simple tokenization and encoding helpers for sequential/discrete data.
- `Main_Demo.ipynb`: demo notebook showing how to run the pipeline and reproduce example workflows.
- `files/P_model.pt`, `files/VAE_model.pt` and `files/denoise_model.pth.tar` : model checkpoints (if present) used for demos.

**Idea / Approach**
- The project learns a latent representation using a shared-attributed VAE that preserves information from both continuous numeric composition vectors and discrete sequential token data. A diffusion (denoising) model is trained / run inside that latent space to generate candidate alloy compositions and associated token sequences conditioned on processing-aware information.

**Requirements / Libraries**
Minimum recommended environment:

- Python 3.8+
- PyTorch (CPU or GPU): `torch` (tested with recent 1.x releases)
- `numpy`
- `jupyter` / `notebook` (to run `Main_Demo.ipynb`)

Install example (basic):
```bash
python -m pip install torch numpy pandas 
```


**Usage (quick)**
- Open the demo: [Main_Demo.ipynb](Main_Demo.ipynb) and run the cells to reproduce examples.
- Place any proprietary ANSYS GRANTA data in a local folder and adapt the data-loading cells in the notebook or scripts to the dataset location and format.
- If pre-trained checkpoints are available in `files/`, point the demo to `files/VAE_model.pt` and `files/P_model.pt` when loading models.

**Notes and attribution**
- The dataset (ANSYS Granta) is proprietary and must be acquired separately from ANSYS.
- Some code portions are adapted from publicly available implementations (see comments in source files for references).

