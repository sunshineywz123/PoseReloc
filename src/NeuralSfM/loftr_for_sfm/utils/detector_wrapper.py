from time import sleep
from matplotlib.pyplot import tight_layout
import torch
import torch.nn as nn
import h5py
import os.path as osp
import os
import numpy as np
from src.utils.comm import gather


class DetectorWrapper(nn.Module):
    def __init__(self,
                 detector=None,
                 detector_type="OnGrid",
                 fullcfg=None):
        super().__init__()
        assert detector_type in ['OnGrid', 'SuperPoint', 'SuperPointEC', 'SIFT'] \
            or 'and grid' in detector_type
        self.detector_type = detector_type

        if detector_type == 'OnGrid':
            assert detector is None
            self.detector = None
        elif detector_type in ['SuperPoint', 'SuperPointEC', 'SIFT'] or 'and grid' in detector_type:
            if not fullcfg['feats_hdf5']:
                assert detector is not None
                self.detector = detector.requires_grad_(False)
            else:
                self.detector = None
        else:
            raise NotImplementedError
            
        self.save_keypoints= fullcfg['save_keypoints']
        self.read_keypoints=fullcfg['feats_hdf5']

        #self.root_dir="/home/hexingyi"
        self.root_dir='/mnt/lustre/hexingyi'

    def draw_detection(self, image, kpts):
        """to visualize the detection results"""
        import numpy as np
        import matplotlib.pyplot as plt
        image = (image[0] * 255).cpu().numpy().astype(np.uint8)
        kpts = kpts.cpu().numpy()
        plt.close("all")
        plt.imshow(image, cmap='gray')
        plt.scatter(kpts[:, 0], kpts[:, 1], s=1)
        plt.savefig("detection.png", dpi=100)
    
    def save_keypoints_multiprocess(self,total_path,keypoints,img_size):
        try:
            with h5py.File(total_path,"w") as f:
                f.create_dataset("keypoints",data=keypoints)
                f.create_dataset("img_size",data=img_size)
        except:
            import time
            import random
            sleep_time=random.random()
            print(f"sleep:{sleep_time}")
            time.sleep(sleep_time)
            self.save_keypoints_multiprocess(total_path,keypoints,img_size)

    @torch.no_grad()
    def forward(self, batch):
        """ Extract keypoints on all **left** images.
        Update:
            batch (dict):{
                "detector_kpts0": [M, 2]    (Optional)
                "detector_b_ids": [M]       (Optional)
            }
        """
        if self.detector_type == 'OnGrid':
            pass
        elif self.detector_type in ['SuperPoint', 'SuperPointEC', 'SIFT'] or 'and grid' in self.detector_type:
            # update batch
            device = batch['image0'].device

            if not self.read_keypoints:
                # compute kpts on the image0
                self.detector.eval()
                ret_dict = self.detector({'image': batch['image0']}, mode='eval')
            else:
                assert "keypoints0" in batch, "keypoints are not loaded!"
                ret_dict = {"keypoints":[batch['keypoints0'][0]]}
                del batch['keypoints0']
            
            '''
            # self.detector.config['keypoint_threshold'] = 0.003  # DEBUG
            if hasattr(self, "feats_saved"):
                # make image_name from batch['pair_names'] -> read keypoints
                # keypoints = self.feats_saved[image_name]['keypoints'].__array__().astype(np.float32)
                raise NotImplementedError
            else:
                ret_dict = self.detector({'image': batch['image0']}, mode='eval')
            # self.draw_detection(batch['image0'][0], ret_dict['keypoints'][0])  # DEBUG 0.005->686/0.003->759
            # indices = torch.randint(len(ret_dict['keypoints'][0]), (100,), device=batch['image0'][0].device)
            # self.draw_detection(batch['image0'][0], ret_dict['keypoints'][0][indices])
            '''
            #TODO: change multiprocess write to gather to gpu1 and sequencely write
            if self.save_keypoints:
                save_path=osp.join(batch['pair_names'][0][0].rsplit("/",2)[0],"keypoints")
                if self.root_dir is not None:
                    save_path=osp.join(self.root_dir,save_path.split('//',1)[-1])
                file_path=osp.basename(batch['pair_names'][0][0]).rsplit(".",1)[0] + "." + "h5py"
                total_path=osp.join(save_path,file_path)
                os.makedirs(save_path,exist_ok=True)
                keypoints=ret_dict['keypoints'][0].clone().detach().cpu().numpy()
                self.save_keypoints_multiprocess(total_path,keypoints,batch['image0'].shape[-1])
            '''
            if self.save_keypoints:
                save_path=osp.join(batch['pair_names'][0][0].rsplit("/",2)[0],"keypoints")
                if self.root_dir is not None:
                    save_path=osp.join(self.root_dir,save_path.split('//',1)[-1])
                file_path=osp.basename(batch['pair_names'][0][0]).rsplit(".",1)[0] + "." + "h5py"
                total_path=osp.join(save_path,file_path)
                save_data={'file_path':total_path,'keypoints':ret_dict["keypoints"][0]}
            '''
            # remove kpts at masking border
            if 'mask0' in batch:
                valid_ws = batch['mask0'].max(1)[0].sum(1)
                valid_hs = batch['mask0'].max(2)[0].sum(1)
                coarse_downscale = batch['image0'].shape[-1] // batch['mask0'].shape[-1]  # e.g. 8
                downscaled_kpts = [kpt_ // coarse_downscale for kpt_ in ret_dict['keypoints']]
                in_mask = [(kpt_[:, 0] < valid_ws[i] - 1) * (kpt_[:, 1] < valid_hs[i] - 1)
                            for i, kpt_ in enumerate(downscaled_kpts)]
            else:
                # all true
                in_mask = [torch.ones((kpt_.shape[0],), device=kpt_.device).bool()
                            for kpt_ in ret_dict['keypoints']]
            kpts0 = torch.cat([kpt_[in_mask[i]]
                                for i, kpt_ in enumerate(ret_dict['keypoints'])],
                                dim=0)  # flatten batch dimension
            scores0 = torch.cat([scores_[in_mask[i]]
                                for i, scores_ in enumerate(ret_dict['scores'])],
                                dim=0)
            bids = torch.cat([torch.Tensor([bs]*sum(in_mask_))
                                for bs, in_mask_ in enumerate(in_mask)], dim=0)

            # kpts0 = torch.cat(ret_dict['keypoints'], dim=0)  # this is the old way.
            # bids = torch.cat([torch.Tensor([bs]*len(kpts)) for bs, kpts in enumerate(ret_dict['keypoints'])], dim=0)
            batch.update({
                'detector_kpts0': kpts0,  # [M, 2] - <x, y> (at input image scale)
                'detector_scores0': scores0,
                'detector_b_ids': bids.long().to(device)  # [M]
            })

        else:
            raise NotImplementedError

        batch.update({'detector_type': self.detector_type})

class DetectorWrapperTwoView(DetectorWrapper):
    @torch.no_grad()
    def forward(self, batch):
        """ Extract keypoints on both left and right images.
        Update:
            batch (dict):{
                "detector_kpts0": [M0, 2]
                "detector_b_ids1": [M0]
                "detector_scores0": [M0]
                "detector_kpts1": [M1, 2]
                "detector_b_ids1": [M1]
                "detector_scores1": [M1]
            }
        """
        batch0 = {'image0': batch['image0']}
        if 'mask0' in batch:
            batch0.update({'mask0': batch['mask0']})
        super().forward(batch0)
        
        batch1 = {'image0': batch['image1']}
        if 'mask1' in batch:
            batch1.update({'mask0': batch['mask1']})
        super().forward(batch1)
        batch.update({
            'detector_kpts0': batch0['detector_kpts0'],
            'detector_b_ids': batch0['detector_b_ids'],
            'detector_scores0': batch0['detector_scores0'],
            'detector_kpts1': batch1['detector_kpts0'],
            'detector_b_ids1': batch1['detector_b_ids'],
            'detector_scores1': batch1['detector_scores0']
        })
