from typing import Tuple

from mmcv.runner import force_fp32
from torch import nn

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2

from mmdet3d.models.builder import VTRANSFORMS

from .base import BaseTransform

__all__ = ["LSSTransform"]


@VTRANSFORMS.register_module()
class LSSTransform(BaseTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.depthnet = nn.Conv2d(in_channels, self.D + self.C, 1)
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    @force_fp32()
    def get_cam_feats(self, x):
        B, N, C, fH, fW = x.shape

        x = x.view(B * N, C, fH, fW)

        x = self.depthnet(x)
        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)

        # visualize the depth image
        # depth = depth.view(B, N, self.D, fH, fW)
        # final_depth_indices = torch.argmax(depth, dim=2)
        # batch_index = 0
        # view_index = 2
        # final_depth_map = final_depth_indices[batch_index, view_index].cpu().numpy()
        # final_depth_map = final_depth_map.astype(np.float32)
        # final_depth_map = (final_depth_map - np.min(final_depth_map)) / (np.max(final_depth_map) - np.min(final_depth_map))

        # # Step 8: Apply Gaussian smoothing
        # smoothed_depth_map = gaussian_filter(final_depth_map, sigma=1)

        # # Step 9: Upsample the depth map
        # upsampled_depth_map = cv2.resize(final_depth_map, (fW * 4, fH * 4), interpolation=cv2.INTER_LINEAR)

        # # Step 10: Apply colormap
        # colored_depth_map = cv2.applyColorMap((upsampled_depth_map * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # plt.imshow(colored_depth_map, cmap='gray')  # Use 'viridis' colormap for better visualization
        # plt.colorbar(label='Depth')
        # plt.title('Final Depth Map')
        # plt.xlabel('Width')
        # plt.ylabel('Height')
        # plt.show()

        return x

    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        if x == None:
            return x
        else:
            x = self.downsample(x)
            return x
