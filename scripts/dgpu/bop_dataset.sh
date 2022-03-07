#!/bin/bash
export SRC=http://ptak.felk.cvut.cz/6DB/public/bop_datasets
for dataset_name in lm
# for dataset_name in lmo tless tudl icbin itodd hb ycbv
do
    wget $SRC/${dataset_name}_base.zip         # Base archive with dataset info, camera parameters, etc.
    wget $SRC/${dataset_name}_models.zip       # 3D object models.
    wget $SRC/${dataset_name}_test_all.zip     # All test images ("_bop19" for a subset used in the BOP Challenge 2019/2020).
    wget $SRC/${dataset_name}_train_pbr.zip    # PBR training images (rendered with BlenderProc4BOP).

    unzip ${dataset_name}_base.zip             # Contains folder "lm".
    unzip ${dataset_name}_models.zip -d $dataset_name     # Unpacks to "lm".
    unzip ${dataset_name}_test_all.zip -d $dataset_name   # Unpacks to "lm".
    unzip ${dataset_name}_train_pbr.zip -d $dataset_name  # Unpacks to "lm".
done