#!/bin/bash
data_base_dir=/nas/users/hexingyi/bop
output_data_dir=/nas/users/hexingyi/onepose_hard_data
dataset_name=lmo
# for obj_id in 11 12 13 14 15 16 17 18 19 20 21
for obj_id in 01 05 06 08 09 10 11 12
    do
        echo "obj_id:$obj_id"
        python tools/data_prepare/parse_bop_data_for_onepose.py \
            --data_base_dir $data_base_dir \
            --dataset_name $dataset_name \
            --obj_id $obj_id \
            --assign_onepose_id 08${obj_id} \
            --output_data_dir $output_data_dir \
            --split val
    done