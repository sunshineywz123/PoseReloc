import sys
sys.path.insert(0, '/home/hexingyi/code/PoseReloc')

import os
import cv2
import json
import tqdm
import numpy as np
import os.path as osp

from src.utils.vis_utils import draw_3d_box
from src.utils import data_utils
from pathlib import Path
from transforms3d import affines, quaternions

def get_gt_pose_path_by_color(color_path):
    return color_path.replace('/color/', '/poses_ba/').replace(
        '.png', '.txt'
    )

def get_intrin_path_by_color(color_path):
    return color_path.replace('/color/', '/intrin_ba/').replace(
        '.png', '.txt'
    )

def get_test_seq_path(obj_root, last_n_seq_as_test=1):
    seq_names = os.listdir(obj_root)
    seq_ids = [int(seq_name.split('-')[-1]) for seq_name in seq_names if '-' in seq_name]
    
    test_obj_name = seq_names[0].split('-')[0]
    test_seq_ids = sorted(seq_ids)[(-1 * last_n_seq_as_test):]
    test_seq_paths = [osp.join(obj_root, test_obj_name + '-' + str(test_seq_id)) for test_seq_id in test_seq_ids]
    return test_seq_paths

def get_refine_box(box_file, trans_box_file):
    def read_transformation(trans_box_file):
        with open(trans_box_file, 'r') as f:
            line = f.readlines()[1]

        data = [float(var) for var in line.split(' ')]
        scale = np.array(data[0])
        rot_vec = np.array(data[1:4])
        trans_vec = np.array(data[4:])
        
        return scale, rot_vec, trans_vec

    box3d, box3d_homo = get_bbox3d(box_file)
    scale, rot_vec, trans_vec = read_transformation(trans_box_file) 
    
    transformation = np.eye(4)
    rotation = cv2.Rodrigues(rot_vec)[0]
    transformation[:3, :3] = rotation
    transformation[:3, 3:] = trans_vec.reshape(3, 1)

    box3d_homo *= scale
    refine_box = transformation @ box3d_homo.T 
    refine_box[:3] /= refine_box[3:]

    return refine_box[:3].T

def get_arkit_default_path(data_dir):
    video_file = osp.join(data_dir, 'Frames.m4v')
    
    color_dir = osp.join(data_dir, 'color')
    Path(color_dir).mkdir(parents=True, exist_ok=True)

    # box_file = osp.join(data_dir, 'RefinedBox.txt')
    box_file = osp.join(data_dir, 'Box.txt')
    assert Path(box_file).exists()

    out_pose_dir = osp.join(data_dir, 'poses')
    Path(out_pose_dir).mkdir(parents=True, exist_ok=True)
    pose_file = osp.join(data_dir, 'ARposes.txt')
    assert Path(pose_file).exists()
    
    intrin_file = osp.join(data_dir, 'Frames.txt')
    assert Path(intrin_file).exists()
    
    reproj_box_dir = osp.join(data_dir, 'reproj_box')
    Path(reproj_box_dir).mkdir(parents=True, exist_ok=True)
    out_box_dir = osp.join(data_dir, 'bbox')
    Path(out_box_dir).mkdir(parents=True, exist_ok=True)

    orig_intrin_file = osp.join(data_dir, 'Frames.txt')
    final_intrin_file = osp.join(data_dir, 'intrinsics.txt')

    intrin_dir = osp.join(data_dir, 'intrin')
    Path(intrin_dir).mkdir(parents=True, exist_ok=True)

    M_dir = osp.join(data_dir, 'M')
    Path(M_dir).mkdir(parents=True, exist_ok=True)

    paths = {
        'video_file': video_file,
        'color_dir': color_dir,
        'box_path': box_file,
        'pose_file': pose_file,
        'out_box_dir': out_box_dir,
        'reproj_box_dir': reproj_box_dir,
        'out_pose_dir': out_pose_dir,
        'orig_intrin_file': orig_intrin_file,
        'final_intrin_file': final_intrin_file,
        'intrin_dir': intrin_dir,
        'M_dir': M_dir
    }
    
    return paths 
    

def get_bbox3d(box_path):
    assert Path(box_path).exists()
    with open(box_path, 'r') as f:
        lines = f.readlines()
    box_data = [float(e) for e in lines[1].strip().split(',')]
    ex, ey, ez = box_data[3: 6]
    bbox_3d = np.array([
        [ex,   ey,  ez],
        [ex,  -ey,  ez],
        [ex,   ey, -ez],
        [ex,  -ey, -ez],
        [-ex,  ey,  ez],
        [-ex, -ey,  ez],
        [-ex,  ey, -ez],
        [-ex, -ey, -ez]
    ]) * 0.5
    bbox_3d_homo = np.concatenate([bbox_3d, np.ones((8, 1))], axis=1)
    return bbox_3d, bbox_3d_homo


def get_K(intrin_file):
    assert Path(intrin_file).exists()
    with open(intrin_file, 'r') as f:
        lines = f.readlines()
    intrin_data = [line.rstrip('\n').split(':')[1] for line in lines]
    fx, fy, cx, cy = list(map(float, intrin_data))

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])
    K_homo = np.array([
        [fx, 0, cx, 0],
        [0, fy, cy, 0],
        [0,  0,  1, 0]
    ])
    return K, K_homo


def parse_box(box_path):
    with open(box_path, 'r') as f:
        lines = f.readlines()
    data = [float(e) for e in lines[1].strip().split(',')]
    position = data[:3]
    quaternion = data[6:]
    rot_mat = quaternions.quat2mat(quaternion)
    T_ow = affines.compose(position, rot_mat, np.ones(3))
    return T_ow


def reproj(K_homo, pose, points3d_homo):
    assert K_homo.shape == (3, 4)
    assert pose.shape == (4, 4)
    assert points3d_homo.shape[0] == 4 # [4 ,n]

    reproj_points = K_homo @ pose @ points3d_homo
    reproj_points = reproj_points[:]  / reproj_points[2:]
    reproj_points = reproj_points[:2, :].T
    return reproj_points # [n, 2]


def parse_video(paths, downsample_rate=5, bbox_3d_homo=None, hw=512):
    orig_intrin_file = paths['final_intrin_file']
    K, K_homo = get_K(orig_intrin_file)
    
    intrin_dir = paths['intrin_dir']
    cap = cv2.VideoCapture(paths['video_file'])
    index = 0
    while True:
        ret, image = cap.read()
        if not ret:
            break
        # if index != 0 and index % downsample_rate == 0: # TODO: check index == 0
        if index % downsample_rate == 0:
            img_name = osp.join(paths['color_dir'], '{}.png'.format(index))
            save_intrin_path = osp.join(intrin_dir, '{}.txt'.format(index))

            # x0, y0, x1, y1 = np.loadtxt(osp.join(paths['out_box_dir'], '{}.txt'.format(index))).astype(int)
            # x0, y0, x1, y1 = np.loadtxt(osp.join(paths['reproj_box_dir'], '{}.txt'.format(index))).astype(int)
            reproj_box3d_file = osp.join(paths['reproj_box_dir'], '{}.txt'.format(index))
            if not osp.isfile(reproj_box3d_file):
                continue
            reproj_box3d = np.loadtxt(osp.join(paths['reproj_box_dir'], '{}.txt'.format(index))).astype(int)
            x0, y0 = reproj_box3d.min(0)
            x1, y1 = reproj_box3d.max(0)

            box = np.array([x0, y0, x1, y1])
            resize_shape = np.array([y1 - y0, x1 - x0])
            K_crop, K_crop_homo = data_utils.get_K_crop_resize(box, K, resize_shape)
            image_crop, trans1 = data_utils.get_image_crop_resize(image, box, resize_shape)

            box_new = np.array([0, 0, x1-x0, y1-y0])  
            resize_shape = np.array([hw, hw])
            K_crop, K_crop_homo = data_utils.get_K_crop_resize(box_new, K_crop, resize_shape)
            try:
                image_crop, trans2 = data_utils.get_image_crop_resize(image_crop, box_new, resize_shape)
            except:
                import ipdb; ipdb.set_trace()

            trans_full_to_crop = trans2 @ trans1 
            trans_crop_to_full = np.linalg.inv(trans_full_to_crop)

            np.savetxt(osp.join(paths['M_dir'], '{}.txt'.format(index)), trans_crop_to_full)
            
            # cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 3)
            pose = np.loadtxt(osp.join(paths['out_pose_dir'], '{}.txt'.format(index)))
            reproj_crop = reproj(K_crop_homo, pose, bbox_3d_homo.T)
            x0_new, y0_new = reproj_crop.min(0)
            x1_new, y1_new = reproj_crop.max(0)
            box_new = np.array([x0_new, y0_new, x1_new, y1_new])
            
            np.savetxt(osp.join(paths['out_box_dir'], '{}.txt'.format(index)), box_new)
            cv2.imwrite(img_name, image_crop)
            full_img_dir = paths['color_dir'] + '_full'
            Path(full_img_dir).mkdir(exist_ok=True, parents=True)
            cv2.imwrite(osp.join(full_img_dir, '{}.png'.format(index)), image)
            np.savetxt(save_intrin_path, K_crop)
            
        index += 1
    cap.release()


def data_process(data_dir, downsample_rate=5, hw=512):
    paths = get_arkit_default_path(data_dir)
    with open(paths['orig_intrin_file'], 'r') as f:
        lines = [l.strip() for l in f.readlines() if len(l) > 0 and l[0] != '#']
    eles = [[float(e) for e in l.split(',')] for l in lines]
    data = np.array(eles)
    fx, fy, cx, cy = np.average(data, axis=0)[2:]
    with open(paths['final_intrin_file'], 'w') as f:
        f.write('fx: {0}\nfy: {1}\ncx: {2}\ncy: {3}'.format(fx, fy, cx, cy))
    
    bbox_3d, bbox_3d_homo = get_bbox3d(paths['box_path'])
    K_homo = np.array([
        [fx, 0, cx, 0],
        [0, fy, cy, 0],
        [0,  0,  1, 0]
    ])

    with open(paths['pose_file'], 'r') as f:
        lines = [l.strip() for l in f.readlines()]
        index = 0
        for line in tqdm.tqdm(lines):
            if len(line) == 0 or line[0] == '#':
                continue

            if index % downsample_rate == 0:
                eles = line.split(',') 
                data = [float(e) for e in eles]

                position = data[1:4]
                quaternion = data[4:]
                rot_mat = quaternions.quat2mat(quaternion)
                rot_mat = rot_mat @ np.array([
                    [1,  0,  0],
                    [0, -1,  0],
                    [0,  0, -1]
                ])

                T_ow = parse_box(paths['box_path'])
                T_cw = affines.compose(position, rot_mat, np.ones(3))
                T_wc = np.linalg.inv(T_cw)
                T_oc = T_wc @ T_ow
                pose_save_path = osp.join(paths['out_pose_dir'], '{}.txt'.format(index))
                box_save_path = osp.join(paths['reproj_box_dir'], '{}.txt'.format(index))
                # box_save_path = osp.join(paths['out_box_dir'], '{}.txt'.format(index))
                reproj_box3d = reproj(K_homo, T_oc, bbox_3d_homo.T)

                x0, y0 = reproj_box3d.min(0)
                x1, y1 = reproj_box3d.max(0)
                
                if x0 < -1000 or y0 < -1000 or x1 > 3000 or y1 > 3000:
                    continue                
                
                reproj_box2d = np.array([x0, y0, x1, y1])
                np.savetxt(pose_save_path, T_oc)
                np.savetxt(box_save_path, reproj_box3d)
            index += 1
    
    parse_video(paths, downsample_rate, bbox_3d_homo, hw=hw)


def ln_data():
    data_root = './data/scan_data'
    orig_data_root = '/data/IKEA_Obj'

    orig_datasets = os.listdir(orig_data_root)
    for orig_dataset in orig_datasets:
        obj_data_root = osp.join(data_root, orig_dataset)
        sub_dirs = os.listdir(osp.join(orig_data_root, orig_dataset))
        obj_name = orig_dataset.split('-')[1]
        if not osp.isdir(obj_data_root):
            os.mkdir(obj_data_root)

            for sub_dir in sub_dirs:
                if obj_name not in sub_dir:
                    continue
                orig_path = osp.join(orig_data_root, orig_dataset, sub_dir)
                sub_dir_path = osp.join(obj_data_root, sub_dir)
                os.mkdir(sub_dir_path)

                os.system(f'ln -s {orig_path}/* {sub_dir_path}')
                print(f'=> ln {sub_dir_path}')
                

if __name__ == "__main__":
    # ln_data()
    # data_root = './data/scan_data'
    data_root = '/nas/users/hexingyi/onepose_hard_data'
    seq_dirs = [
        # '0408-colorbox-box', '0409-aptamil-box', '0410-huiyuan-box',
        # '0411-doveshampoo-others', '0412-pikachubowl-others', '0413-juliecookies-box',
        # '0414-babydiapers-others', '0415-captaincup-others', '0416-tongyinoodles-others',
        # '0417-kangshifunoodles-others', '0418-cookies1-others', '0419-cookies2-others',
        # '0420-liuliumei-others', '0421-cannedfish-others', '0422-qvduoduo-box', '0423-oreo-box'
        # '0424-chocbox-box', '0425-vitacare-others', '0426-prunes-others', '0427-raisins-others',

        # '0428-carrotmeat-others', '0429-cranberry-others', '0430-newtolmeat-others', '0431-hawthorntea-others',
        # '0432-sesamesedge-others', '0433-badamsedge-others', '0434-plum-others', '0435-chips-others',
        # '0436-sedge-others', '0437-twinings-others', '0438-royal-others', '0439-nongfuspring-others',
        # '0440-goatmilk-others', '0441-baci-others', '0442-hersheys-others', '0443-wheatbear-others',
        # '0444-weizen-others', '0445-pistonhead-others', '0446-blanc-others', '0447-nabati-box', 
        # '0448-soyabeancookies-bottle'

        # "0449-ybtoothpaste-box", "0450-hlychocpie-box", "0451-originaloreo-box", 
        # "0452-hlymatchapie-box", "0453-hlyyolkpie-box", "0454-darlietoothpaste-box",
        # "0455-strawberryoreo-box", "0456-chocoreo-box", "0457-nabati-bottle", 
        # "0458-hetaocakes-box",

        # "0459-jzhg-box", "0460-bhdsoyabeancookies-bottle", "0461-cranberrycookies-bottle",
        # "0462-ylmilkpowder-bottle", "0463-camelmilk-bottle", "0464-mfchoccake-box",
        # "0465-mfcreamcake-box", "0466-mfmilkcake-box", "0467-peppapigcookies-box",
        # "0468-minipuff-box", "0469-diycookies-box", "0470-eggrolls-box", "0471-hlyormosiapie-box",
        # "0472-chocoreo-bottle", "0473-twgrassjelly1-box", "0474-twgrassjelly2-box",
        # "0475-dinosaurcup-bottle", "0476-giraffecup-bottle", "0477-cutlet-bottle",
        # "0478-seasedgecutlet-bottle", "0479-ggbondcutlet-others", "0480-ljcleaner-others", 
        # "0481-yxcleaner-others", "0482-gagaduck-others", "0483-ambrosial-box", "0484-bigroll-box",
        # "0485-ameria-box", "0486-sanqitoothpaste-box", "0487-jindiantoothpaste-box",
        # "0448-soyabeancookies-bottle", "0489-taipingcookies-others", "0490-haochidiancookies-others",
        # "0491-jiashilicookies-others", "0492-tuccookies-box", "0493-haochidianeggroll-box", 
        # "0494-qvduoduocookies-box", "0495-fulingstapler-box", '0496-delistapler-box',
        # "0497-delistaplerlarger-box", "0498-yousuanru-box", "0499-tiramisufranzzi-box", 

        # "0500-chocfranzzi-box", "0501-matchafranzzi-box", "0502-shufujia-box", "0503-shufujiawhite-box",
        # "0504-lux-box", "0505-ksfbeefnoodles-others", "0506-sauerkrautnoodles-others",
        # "0507-hotsournoodles-others", "0508-yqsl-others", "0509-bscola-others", "0510-yqslmilk-others"
        # "0541-apromilk-others", '0542-bueno-box', '0543-brownhouse-others',
        # '0544-banana-others', '0545-book-others', '0546-can-bottle',
        # '0547-cubebox-box', '0548-duck-others', '0549-footballcan-bottle',
        # '0550-greenbox-box', '0551-milk-others', '0552-mushroom-others',
        # '0553-swifferbox-box', '0554-papercups-others', '0555-pig-others',
        # '0556-pinkbox-box', '0557-santachoc-others', '0558-teddychoc-others',
        # '0559-tissuebox-box', '0560-tofubox-box', 
        # '0561-yellowbottle-bottle', '0562-yellowbox-box'
        # "0511-policecar-others", "0512-ugreenhub-box", '0513-busbox-box'
        # "0407-clock-others", '0408-colorbox-box', '0409-aptamil-box',
        # '0410-huiyuan-box', '0411-doveshampoo-others', '0412-pikachubowl-others',
        # '0413-juliecookies-box', '0414-babydiapers-others', '0415-captaincup-others',
        # '0416-tongyinoodles-others', '0417-kangshifunoodles-others', '0418-cookies1-others',
        # '0419-cookies2-others', '0420-liuliumei-others', '0421-cannedfish-others',
        # '0422-qvduoduo-box', '0423-oreo-box', '0424-chocbox-box', '0425-vitacare-others',
        # '0426-prunes-others', '0427-raisins-others', '0428-carrotmeat-others',
        # '0429-cranberry-others', '0430-newtolmeat-others', '0431-hawthorntea-others',
        # '0432-sesamesedge-others', '0433-badamsedge-others', '0434-plum-others', 
        # '0435-chips-others', '0436-sedge-others', '0437-twinings-others',
        # '0438-royal-others', '0439-nongfuspring-others', '0440-goatmilk-others',
        # '0442-hersheys-others', '0443-wheatbear-others',
        # '0444-weizen-others', '0445-pistonhead-others', "0446-blanc-others",
        # '0447-nabati-box', '0448-soyabeancookies-bottle', "0449-ybtoothpaste-box", 
        # '0450-hlychocpie-box', '0451-originaloreo-box', '0452-hlymatchapie-box',
        # "0453-hlyyolkpie-box", "0454-darlietoothpaste-box", '0455-strawberryoreo-box',
        # '0456-chocoreo-box', '0457-nabati-bottle', '0458-hetaocakes-box', 
        # '0459-jzhg-box', '0460-bhdsoyabeancookies-bottle', '0461-cranberrycookies-bottle', 
        # '0462-ylmilkpowder-bottle', '0463-camelmilk-bottle', '0464-mfchoccake-box', 
        # '0465-mfcreamcake-box', '0466-mfmilkcake-box', '0467-peppapigcookies-box', 
        # '0468-minipuff-box', '0469-diycookies-box', '0470-eggrolls-box', '0471-hlyormosiapie-box', 
        # '0472-chocoreo-bottle', '0473-twgrassjelly1-box', '0474-twgrassjelly2-box', 
        # '0475-dinosaurcup-bottle', '0476-giraffecup-bottle', '0477-cutlet-bottle', 
        # '0478-seasedgecutlet-bottle', '0479-ggbondcutlet-others', '0480-ljcleaner-others', 
        # '0481-yxcleaner-others', '0482-gagaduck-others', '0483-ambrosial-box', '0484-bigroll-box',
        # "0485-ameria-box", "0486-sanqitoothpaste-box", '0487-jindiantoothpaste-box',
        # '0488-jijiantoothpaste-box', '0489-taipingcookies-others', '0490-haochidiancookies-others',
        # '0491-jiashilicookies-others', '0492-tuccookies-box', '0493-haochidianeggroll-box',
        # '0494-qvduoduocookies-box', '0495-fulingstapler-box', '0496-delistapler-box',
        # '0497-delistaplerlarger-box', '0498-yousuanru-box', '0499-tiramisufranzzi-box',
        # '0500-chocfranzzi-box', '0501-matchafranzzi-box', '0502-shufujia-box', '0503-shufujiawhite-box',
        # '0504-lux-box', '0505-ksfbeefnoodles-others', '0506-sauerkrautnoodles-others',
        # '0507-hotsournoodles-others', '0508-yqsl-others', '0509-bscola-others', '0510-yqslmilk-others',
        # '0511-policecar-others', '0512-ugreenhub-box', '0513-busbox-box',
        # '0514-hmbb-others', '0515-porcelain-others',
        # '0516-wewarm-box', '0517-nationalgeo-box', '0518-jasmine-box',
        # '0519-backpack1-box', '0520-lipault-box', '0521-ranova-box',
        # '0522-milkbox-box', '0523-edibleoil-others', '0524-skimmilkpowder-bottle',
        # '0525-toygrab-others', '0526-toytable-others', '0527-spalding-others',
        # '0528-crunchy-box', '0529-onionnoodles-box', '0530-trufflenoodles-box',
        # '0531-whiskware-box', '0532-delonghi-box', '0533-shiramyun-box',
        # '0534-tonkotsuramen-box', '0535-odbmilk-box', '0536-ranovarect-box',
        # '0537-petsnack-box', '0538-winterquilt-box', '0539-spamwrapper-others', '0585-ecobox-box'
        # "0563-applejuice-box", '0564-biatee-others', 
        # '0565-biscuits-box', '0566-chillisauce-box',
        # '0567-coffeebox-box', '0568-cornflakes-box', '0569-greentea-bottle', '0570-kasekuchen-box',
        # '0571-cakebox-box', '0572-milkbox-others', '0573-redchicken-others', '0574-rubberduck-others',
        # '0575-saltbottle-bottle', '0576-saltbox-box', '0577-schoko-box', '0578-tee-others',
        # '0579-tomatocan-bottle', '0580-xmaxbox-others', '0581-yogurt-bottle', '0582-yogurtlarge-others',
        # '0583-yogurtmedium-others', '0584-yogurtsmall-others'
        # '0600-toyrobot-others'
        # '0601-oldtea-box', '0602-sensesheep-others', '0603-fakebanana-others', '0604-catmodel-others'
        # '0605-pingpangball-ball', '0606-yellowduck-others', '0607-oringe-others'
        # '0608-greenteapot-others', '0609-blackteapot-others', '0610-lecreusetcup-others', '0611-bosecolumnaudio-others'
        # "0612-insta-others", '0613-batterycharger-others', '0614-hhkbhandrest-others', '0615-logimouse-others', '0616-sensezong-others', '0617-miehuoqixiang-others'
        # "0618-huangjinyatea-others", 
        '0619-blueyellowbox-box', '0620-shuixiantea-others','0621-headphonecontainer-others', '0622-ugreenbox-others', '0623-camera-others', '0624-weishengsubox-others', '0625-dogairpods-others'
    ]
    deal_first = True
    deal_last = True
    for seq_dir in seq_dirs:
        data_dir = osp.join(data_root, seq_dir)
        subdirs = os.listdir(data_dir)
        subdir_ids = []
        # Find last seq for an object
        for subdir in subdirs:
            if not subdir.startswith('.') and 'visual' not in subdir and '.' not in subdir:
                if '-' in subdir:
                    obj_name, subdir_id = subdir.split('-')
                elif '_' in subdir:
                    obj_name, subdir_id = subdir.split('_')
                else:
                    obj_name, subdir_id = subdir[:-1], subdir[-1]
                subdir_ids.append(int(subdir_id))
        subdir_ids.sort()
        max_id = subdir_ids[-1]
        min_id = subdir_ids[0]
        
        for subdir in subdirs: 
            if not subdir.startswith('.') and "visual" not in subdir and '.' not in subdir:
                if '-' in subdir:
                    obj_name, subdir_id = subdir.split('-')
                elif '_' in subdir:
                    obj_name, subdir_id = subdir.split('_')
                else:
                    obj_name, subdir_id = subdir[:-1], subdir[-1]

                print('=> processing: ', subdir)
                if int(subdir_id) == max_id and deal_last:
                    data_process(osp.join(data_dir, subdir), downsample_rate=1, hw=512)
                elif int(subdir_id) == min_id and deal_first:
                    data_process(osp.join(data_dir, subdir), downsample_rate=1, hw=512)
                else:
                    data_process(osp.join(data_dir, subdir), downsample_rate=5, hw=512)
                 