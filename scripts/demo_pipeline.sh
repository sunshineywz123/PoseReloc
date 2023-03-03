#!/bin/bash
PROJECT_DIR="$(pwd)"
OBJ_NAME=$1
echo "Current work dir: $PROJECT_DIR"

echo '-------------------'
echo 'Parse scanned data:'
echo '-------------------'
# Parse scanned annotated & test sequence:
python $PROJECT_DIR/parse_scanned_data.py \
    --scanned_object_path \
    "$PROJECT_DIR/data/demo/$OBJ_NAME"

echo '--------------------------------------------------------------'
echo 'Run Keypoint-Free SfM to reconstruct object point cloud for pose estimation:'
echo '--------------------------------------------------------------'
# Run SfM to reconstruct object sparse point cloud from $OBJ_NAME-annotate sequence:
time python $PROJECT_DIR/run.py \
    +preprocess="sfm_demo" \
    dataset.data_dir="[$PROJECT_DIR/data/demo/$OBJ_NAME $OBJ_NAME-annotate-1]" \
    dataset.outputs_dir="$PROJECT_DIR/data/demo/sfm_model" \

echo "-----------------------------------"
echo "Run inference and output demo video:"
echo "-----------------------------------"

# # Run inference on $OBJ_NAME-test and output demo video:
time python $PROJECT_DIR/demo.py +experiment="inference_demo" data_base_dir="$PROJECT_DIR/data/demo/$OBJ_NAME $OBJ_NAME-test\
 $OBJ_NAME-test-1 $OBJ_NAME-test-2 $OBJ_NAME-test-3 \
 $OBJ_NAME-test-4 $OBJ_NAME-test-5 $OBJ_NAME-test-6 \
 $OBJ_NAME-test-7 $OBJ_NAME-test-8 $OBJ_NAME-test-9 \
 $OBJ_NAME-test-10 \
 $OBJ_NAME-test-11"\
 sfm_base_dir="$PROJECT_DIR/data/demo/sfm_model/outputs_softmax_loftr_loftr/$OBJ_NAME"