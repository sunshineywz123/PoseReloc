#!/bin/bash
data_base_dir=/nas/users/hexingyi/lm_full
output_data_dir=/nas/users/hexingyi/yolo_real_data
dataset_name=lm
for obj_id in 02 04 05 06 08 09 10 11 12 13 14 15
# for obj_id in 04 05 06 08 09 10 11 13 14 15
# for obj_id in 05 06 08 10 11 13 14 15
# for obj_id in 12
    do
        echo "obj_id:$obj_id"
        python tools/data_prepare/parse_lm_real_data_for_yolo.py \
            --data_base_dir $data_base_dir \
            --obj_id $obj_id \
            --assign_onepose_id 08${obj_id} \
            --output_data_dir $output_data_dir \
            --split train
        python tools/data_prepare/parse_lm_real_data_for_yolo.py \
            --data_base_dir $data_base_dir \
            --obj_id $obj_id \
            --assign_onepose_id 08${obj_id} \
            --output_data_dir $output_data_dir \
            --split val
    done