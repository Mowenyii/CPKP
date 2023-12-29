# CPKP

## How to Install
This code is built on top of the awesome toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch).

### Install `dassl` as well as PyTorch. 
```bash
# Create a conda environment
conda create -n capkp python=3.7

# Activate the environment
conda activate capkp

cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install torch (version >= 1.7.1) and torchvision
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
```
### Install any additional packages that `CPKP` needs
```bash
cd CoOp-main/

pip install -r requirements.txt
```

### Install the datasets.
Follow [DATASETS.md](CoOp-main/DATASETS.md) to install the datasets.

## How to Run

We provide the running scripts in `scripts/`. Make sure you change the path in `DATA` and run the commands under `CoOp-main/scripts/`.

### Few-Shot Learning
For detailed parameter settings, see `CoOp-main/scripts/main_capkp.sh`, which contains nine input arguments.

`DATASET` takes as input a dataset name, like `imagenet` or `food101`. The valid names are the files' names in `CoOp-main/configs/datasets/`.

`CFG` means which config file to use, such as `rn50`, `rn101` or `vit_b32` (see `CoOp-main/configs/trainers/CPKP/`). Note that for ImageNet, we use `CoOp-main/configs/trainers/CPKP/*_ep50.yaml` for all settings (please follow the implementation details shown in the paper).

Below we provide examples on how to run CoOp-main on Food101. 

**CPKP (M=16, SHR)**:
- 1 shot: `bash main.sh food101 rn50_ep50 16 1 False 1e-1 0 True 101`




**CPKP(M=1, SPE)**:
- 1 shot: `bash main.sh food101 rn50_ep50 1 1 True 1e-1 0 True 101 `





**How to visualize nearest words for the learned context tokens?** The learned tokens are saved in `a/b/c/prompt_learner/model.pth.tar` and you would like to see the top-3 nearest words for each token. In this case, run `python interpret_prompt.py a/b/c/prompt_learner/model.pth.tar 3`

**CPKP (M=16, SHR)**:

`python interpret_prompt.py a/b/c/nctx16_cscFalse/d/prompt_learner/model.pth.tar-200 1 `

**CPKP (M=1, SPE)**:

`python interpret_prompt.py a/b/c/nctx1_cscTrue/d/prompt_learner/model.pth.tar-200 1 `



### Zero-Shot CLIP
See `CoOp-main/scripts/zeroshot.sh`.