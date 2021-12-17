import os
import subprocess
import numpy as np

opencv32_dlib_path = "/mnt/env/opencv-3.2.0/build/lib"


def readkp(filename):
    def is_float(n):
        try:
            float(n)
            return True
        except:
            return False
    kp_temp = []
    counter = 0
    num_kp = 0
    with open(filename, "r") as file:
        for line in file.readlines():
            counter += 1

            if (counter == 2):
                num_kp = int(line)
            if (counter > 2):
                kp_temp.append([float(n) for n in line.split(',') if is_float(n)])

    kp_temp = np.array(kp_temp, dtype=np.float32)

    assert (kp_temp.shape[0] == num_kp)
    assert (kp_temp.shape[1] == 4)

    return kp_temp, num_kp


def FFD_detect(image_path, num_level=3, max_keypoints=-1, contrast_threshold=0.05, curvature_ratio=10, time_cost=-1):
    os.environ['LD_LIBRARY_PATH'] = opencv32_dlib_path
    exe = "/mnt/Code/svcnn-matcher/src/extractors/FFD/FFD"
    tmp_dir = "/mnt/Code/svcnn-matcher/src/extractors/FFD/tmp"
    cmd = f"{exe} {image_path} {tmp_dir} {num_level} {max_keypoints} {contrast_threshold} {curvature_ratio} {time_cost}"

    process = subprocess.Popen(cmd, shell=True)
    process.wait()

    txt_path = os.path.join(tmp_dir, 'FFD_'+image_path.split('/')[-1]+'.txt')
    keypoints, _ = readkp(txt_path)

    process = subprocess.Popen(f'rm {txt_path}', shell=True)
    process.wait()

    return keypoints
