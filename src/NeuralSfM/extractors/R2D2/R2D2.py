import torch
from .patchnet import *
from pathlib import Path

from .utils import common

class NonMaxSuppression (torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr
    
    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        maxima *= (reliability   >= self.rel_thr)

        return maxima.nonzero().t()[2:4]


class R2D2(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        self.version=config['version']
        self.max_keypoints=config['max_keypoints']
        self.config=config


        model_path=Path(__file__).parent/'models'/f"r2d2_{self.version}.pt"
        self.model=self.load_network(model_path)

        self.detector= NonMaxSuppression(rel_thr=config["rel_thr"],rep_thr=config["rep_thr"])
    
    def load_network(self,model_fn): 
        checkpoint = torch.load(model_fn,map_location="cpu")
        print("\n>> Creating net = " + checkpoint['net']) 
        net = eval(checkpoint['net'])
        nb_of_weights = common.model_size(net)
        print(f" ( Model size: {nb_of_weights/1000:.0f}K parameters )")

        # initialization
        weights = checkpoint['state_dict']
        net.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
        return net.eval()
    
    def forward(self,data,mode):
        device=data["image"].device
        images=data["image"]
        B,C,H,W=images.shape

        assert B==1,"only batch_size=1 is allow for R2D2"
        images=images.expand(-1,3,-1,-1) if C==1 else images

        #multi-scale extraction and detection
        min_size=246
        max_size=2000

        min_scale=self.config["min_scale"]
        max_scale=self.config["max_scale"]
        scale_f=self.config["scale_f"]

        s=1.0
        X,Y,S,C,Q,D = [],[],[],[],[],[]
        while  s+0.001 >= max(min_scale, min_size / max(H,W)):
            if s-0.001 <= min(max_scale, max_size / max(H,W)):
                nh, nw = images.shape[2:]
                # extract descriptors
                with torch.no_grad():
                    res = self.model(imgs=[images])
                    
                # get output and reliability map
                descriptors = res['descriptors'][0]
                reliability = res['reliability'][0]
                repeatability = res['repeatability'][0]

                # normalize the reliability for nms
                # extract maxima and descs
                y,x = self.detector(**res) # nms
                c = reliability[0,0,y,x]
                q = repeatability[0,0,y,x]
                d = descriptors[0,:,y,x]
                n = d.shape[0]

                # accumulate multiple scales
                X.append(x.float() * W/nw)
                Y.append(y.float() * H/nh)
                S.append((32/s) * torch.ones(n, dtype=torch.float32, device=device))
                C.append(c)
                Q.append(q)
                D.append(d)
            s /= scale_f

            # down-scale the image for next iteration
            nh, nw = round(H*s), round(W*s)
            images = F.interpolate(images, (nh,nw), mode='bilinear', align_corners=False)
        
        Y = torch.cat(Y)
        X = torch.cat(X)
        S = torch.cat(S) # scale
        scores = torch.cat(C) * torch.cat(Q) # scores = reliability * repeatability
        XY = torch.stack([X,Y], dim=-1)
        D = torch.cat(D)

        #get top k!
        n=self.max_keypoints+1
        
        #only for mnn
        n=min(n,scores.shape[0])
        if scores.shape[0]<self.max_keypoints+1:
            print(f"warning:not enough points,only have{scores.shape[0]},condition only allow for mnn")
            self.max_keypoints=n-1

        #constrains for spg
        #assert scores.shape[0] >= self.max_keypoints,f"not enough points,only have{scores.shape[0]}"
        minus_threshold, _indices = torch.kthvalue(-scores, n)
        mask = scores > -minus_threshold 
                
        if mask.float().sum() != self.max_keypoints:
            mask_equal = scores == -minus_threshold
            assert mask_equal.float().sum()!=1,"num of threshold is 1"
            diff=self.max_keypoints-mask.float().sum()
            assert mask_equal.float().sum()>=diff,"num of threhold smaller than diff"
            for i in range(mask_equal.numel()):
                if mask_equal[i]==True:
                    if diff!=0:
                        diff-=1
                    else:
                        mask_equal[i]=False
            mask=mask | mask_equal
            assert mask.float().sum() == self.max_keypoints,"still not equal"
        
        return {
            "keypoints": XY[mask].unsqueeze(0).long(),
            "scores": scores[mask].unsqueeze(0),
            "descriptors":D[:,mask].unsqueeze(0)
        }