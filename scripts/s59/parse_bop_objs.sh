#!/bin/bash
data_base_dir=/cephfs-mvs/3dv-research/hexingyi/bop
output_data_dir=/cephfs-mvs/3dv-research/hexingyi/arscan_aligned/arscan_data
dataset_name=ycbv
# for obj_id in 11 12 13 14 15 16 17 18 19 20 21
for obj_id in 1 2 3 4 5 6 7 8 9 10
    do
        echo "obj_id:$obj_id"
        python tools/data_prepare/parse_bop_data_for_onepose.py \
            --data_base_dir $data_base_dir \
            --dataset_name $dataset_name \
            --obj_id $obj_id \
            --assign_onepose_id 07${obj_id} \
            --output_data_dir $output_data_dir
    done