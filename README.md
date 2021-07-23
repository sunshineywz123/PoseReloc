# 1 Prepare Data

Most scanning data and its reconstruction results are under the path: 

```
Scanning data: /data/wangzihao/code/PoseReloc/data
Reconstruction results: /data/wangzihao/code/PoseReloc/data/sfm_model
```

Some useful model:

```
SuperPoint pretrained model: /data/wangzihao/code/sort_PoseReloc/data/models/extractors/SuperPoint/superpoint_v1.pth
SuperGlue pretrained model: /data/wangzihao/code/sort_PoseReloc/data/models/matchers/SuperGlue/superglue_outdoor.pth 
```


## 1.1 SFM

Firstly, we need to run sparse reconstruction on scanning data to get sparse point clouds and 3d features. For each object, we may have multi-sequences under its directory. We can assign which sequences to run sparse reconstruction.

```python
python run.py +preprocess=sfm_spp_spg data_dir=/root_dir_of_an_object/and/seqs_to_reconstruct
```

## 1.2 Merge multi-objects data for training

To train a model on multi-objects, we need to merge different objects' annotations.

```python
python run.py +preprocess=merge_anno names=object_names_to_merge task_name=task_name
```



# 2 Training

```python
python train.py +experiment=train_PoseReloc.yaml task_name=task_name exp_name=exp_name
```



# 3 Inference


```python
python inference.py +experiment=test_PoseReloc task_name=task_name model.pretrain_model_path=/path/to/checkpoint/ input.data_dir=/path/to/test_sequence/ input.data_dir=/path/to/test_object_reconstruction_model
```

