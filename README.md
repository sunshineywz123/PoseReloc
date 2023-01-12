# OnePose++: Keypoint-Free One-Shot Object Pose Estimation without CAD Models
### [Project Page](https://zju3dv.github.io/onepose_plus_plus) | [Paper]()
<br/>

> OnePose++: Keypoint-Free One-Shot Object Pose Estimation without CAD Models
> [Xingyi He](https://github.com/hxy-123/)<sup>\*</sup>, [Jiaming Sun](https://jiamingsun.ml)<sup>\*</sup>,[Yu'ang Wang](https://github.com/angshine), [Di Huang](https://github.com/dihuangdh), [Hujun Bao](http://www.cad.zju.edu.cn/home/bao/), [Xiaowei Zhou](https://xzhou.me)
> NeurIPS 2022

![demo_vid](assets/demo.gif)

## TODO List
- [x] Training, inference and demo code.
- [x] Pipeline to reproduce the evaluation results on the OnePose dataset and proposed OnePose_LowTexture dataset.
- [ ] Use multiple GPUs for parallelized reconstruction and evaluation.
- [ ] `OnePose Cap` app: we are preparing for the release of the data capture app to the App Store (iOS only), please stay tuned.

## Installation

```shell
conda env create -f environment.yaml
conda activate oneposeplus
```

LoFTR and DeepLM are used in this project, thanks for their great work and we appreciate their contribution to the community. Please follow their installation instructions and LICENSE:
```shell
git submodule update --init --recursive

# Install DeepLM
cd submodules/DeepLM
sh example.sh
```
Note that the efficient optimizer DeepLM is used in our SfM refinement phase. If you face difficulty in installing DeepLM, don't worry. You can still run the code by using our first-order optimizer which is a little slower.

[COLMAP](https://colmap.github.io/) is also used in this project for Structure-from-Motion. Please refer to the official [instructions](https://colmap.github.io/install.html) for the installation.

Download the [pretrained models]() including our 2D-3D model and LoFTR model. Then move them to `${REPO_ROOT}/weights`.

[Optional] You may optionally try out our web-based 3D visualization tool [Wis3D](https://github.com/zju3dv/Wis3D) for convenient and interactive visualizations of feature matches. We also provide many other cool visualization features in Wis3D, welcome to try it out.

```bash
# Working in progress, should be ready very soon, only available on test-pypi now.
pip install -i https://test.pypi.org/simple/ wis3d
```
## Demo
After the installation, please refer to [this page](doc/demo.md) for running demo with your custom data.


## Training and Evaluation
### Dataset setup 
1. Download OnePose from [onedrive storage](https://zjueducn-my.sharepoint.com/:f:/g/personal/zihaowang_zju_edu_cn/ElfzHE0sTXxNndx6uDLWlbYB-2zWuLfjNr56WxF11_DwSg?e=GKI0Df) and and OnePose_LowTexture dataset from [onedrive storage](https://zjueducn-my.sharepoint.com/:f:/g/personal/zihaowang_zju_edu_cn/ElfzHE0sTXxNndx6uDLWlbYB-2zWuLfjNr56WxF11_DwSg?e=GKI0Df), extract them into `$/your/path/to/onepose_datasets`. 
The directory should be organized in the following structure:
    ```
    |--- /your/path/to/datasets
    |       |--- train_data
    |       |--- val_data
    |       |--- test_data
    |       |--- lowtexture_test_data
    ```

2. Build the dataset symlinks
    ```shell
    REPO_ROOT=/path/to/OnePose_Plus_Plus
    ln -s /your/path/to/datasets $REPO_ROOT/data/datasets
    ```

### Inference
1. Run Structure-from-Motion for the data sequences

    Reconstructed the semi-dense object point cloud and 2D-3D correspondences are needed for both training and test objects:
    ```python
    python run.py +preprocess=sfm_spp_spg_train.yaml # for training data
    python run.py +preprocess=sfm_spp_spg_test.yaml # for testing data
    python run.py +preprocess=sfm_spp_spg_val.yaml # for val data
    ```

2. Inference:

    ```python
    python inference.py +experiment=test_GATsSPG
    ```

### Training
1. Prepare ground-truth annotations. Merge annotations of training/val data:
    ```python
    python run.py +preprocess=merge_anno task_name=onepose split=train
    python run.py +preprocess=merge_anno task_name=onepose split=val
    ```
   
2. Begin training
    ```python
    python train.py +experiment=train_GATsSPG task_name=onepose exp_name=training_onepose
    ```
   
All model weights will be saved under `${REPO_ROOT}/models/checkpoints/${exp_name}` and logs will be saved under `${REPO_ROOT}/logs/${exp_name}`.
<!-- You can visualize the training process by tensorboard:
```shell
tensorboard xx
``` -->

## Citation
If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@inproceedings{
    he2022oneposeplusplus,
    title={OnePose++: Keypoint-Free One-Shot Object Pose Estimation without {CAD} Models},
    author={Xingyi He and Jiaming Sun and Yuang Wang and Di Huang and Hujun Bao and Xiaowei Zhou},
    booktitle={Advances in Neural Information Processing Systems},
    year={2022}
}
```


## Acknowledgement
Part of our code is borrowed from [hloc](https://github.com/cvg/Hierarchical-Localization) and [LoFTR](https://github.com/zju3dv/LoFTR). Thanks to their authors for the great works.