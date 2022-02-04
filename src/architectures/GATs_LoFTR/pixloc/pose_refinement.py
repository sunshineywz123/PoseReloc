from turtle import forward
import torch
import torch.nn as nn

class PoseRefinement(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, data):
        '''
        Update:
            "pose_optimized": 4*4
        '''