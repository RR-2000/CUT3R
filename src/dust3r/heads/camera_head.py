import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from dust3r.heads.postprocess import (
    postprocess_pose,
)

# code adapted from 'https://github.com/yyfz/Pi3/blob/main/pi3/models/layers/camera_head.py'

class ResConvBlock(nn.Module):
    """
    1x1 convolution residual block
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head_skip = nn.Identity() if self.in_channels == self.out_channels else nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        # self.res_conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        # self.res_conv2 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)
        # self.res_conv3 = nn.Conv2d(self.out_channels, self.out_channels, 1, 1, 0)

        # change 1x1 convolution to linear
        self.res_conv1 = nn.Linear(self.in_channels, self.out_channels)
        self.res_conv2 = nn.Linear(self.out_channels, self.out_channels)
        self.res_conv3 = nn.Linear(self.out_channels, self.out_channels)

    def forward(self, res):
        x = F.relu(self.res_conv1(res))
        x = F.relu(self.res_conv2(x))
        x = F.relu(self.res_conv3(x))
        res = self.head_skip(res) + x
        return res

class CameraHead(nn.Module):
    def __init__(self, dim=512, pose_mode='linear'):
        super().__init__()
        output_dim = dim
        self.output_dim = output_dim
        self.res_conv = nn.ModuleList([deepcopy(ResConvBlock(output_dim, output_dim)) 
                for _ in range(2)])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.more_mlps = nn.Sequential(
            nn.Linear(output_dim + 768, output_dim + 768), # add dim for camera pose input
            nn.ReLU(),
            nn.Linear(output_dim + 768, output_dim + 768),
            nn.ReLU()
            )
        self.fc_pose = nn.Linear(output_dim + 768, 7)
        self.pose_mode = pose_mode

    def forward(self, feat, raft, yolo, patch_h, patch_w, cam_token):
        BN, hw, c = feat.shape
        input = [feat]
        if raft is not None:
            input.append(raft)
        if yolo is not None:
            input.append(yolo)

        input= torch.cat(input, dim=-1)  # concatenate along the channel dimension

        for i in range(2):
            feat = self.res_conv[i](input)

        # feat = self.avgpool(feat)
        input = self.avgpool(input.permute(0, 2, 1).reshape(BN, -1, patch_h, patch_w).contiguous())              ##########
        
        input = input.view(input.size(0),1,  -1)
        input_final = torch.cat([input, cam_token], dim=-1).view(input.size(0), -1)  # concatenate along the channel dimension

        input_final = self.more_mlps(input_final)  # [B, D_]
        with torch.amp.autocast(device_type='cuda', enabled=False):
            pose = self.fc_pose(input_final.float())  # [B,7]

        out_pose = postprocess_pose(pose, mode=self.pose_mode)

        return out_pose