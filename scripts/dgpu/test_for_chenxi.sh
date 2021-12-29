#!/bin/bash
set -x
set -u
set -e

now=$(date +"%Y%m%d_%H%M%S")
jobname="$1-$now"

# scenes=brandenburg_gate lincoln_memorial palacio_de_bellas_artes pantheon_exterior trevi_fountain
for scene in brandenburg_gate lincoln_memorial palacio_de_bellas_artes pantheon_exterior trevi_fountain
do
    echo "processing scene ${scene}"
    rm -rf ./data/${scene}
    mkdir ./data/${scene}
    ln -s /nas/datasets/IMC/phototourism/training_set/${scene}/dense/images ./data/${scene}
    python run_neuralSfM.py --work_dir ./data/${scene} --use_ray --n_images 10 2>&1|tee log/${jobname}_${scene}.log
done