from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS


from .base import Base3DFusionModel

import math
import pickle
import numpy as np

from Transformer_Encoder import TransformerEncoder
import Spatial_info
import matplotlib.pyplot as plt
import time
import timm
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import ToPILImage
from torchvision import transforms

import torch.nn.functional as F
import torch
from torchvision.transforms import GaussianBlur

__all__ = ["BEVFusion"]


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        if encoders.get("radar") is not None:
            if encoders["radar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["radar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["radar"]["voxelize"])
            self.encoders["radar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["radar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["radar"].get("voxelize_reduce", True)

        # initialize the infrastructure part
        if encoders.get("infra") is not None:
            self.encoders["infra"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["infra"]["backbone"]),
                    "neck": build_neck(encoders["infra"]["neck"]),
                    "vtransform": build_vtransform(encoders["infra"]["vtransform"]),
                }
            )

        if fuser is not None:
            # initialize the transformer here
            self.input_cross_transformer = TransformerEncoder(num_layers=1, input_dim=128,dim_feedforward=128*3, num_heads=4,cross_label=True,bev_cross_label=False)
            self.input_cross_transformer_infra = TransformerEncoder(num_layers=1, input_dim=128,dim_feedforward=128*3, num_heads=4,cross_label=True,bev_cross_label=False)
            
            self.seg_model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
        
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None
        
        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        # If the camera's vtransform is a BEVDepth version, then we're using depth loss. 
        self.use_depth_loss = ((encoders.get('camera', {}) or {}).get('vtransform', {}) or {}).get('type', '') in ['BEVDepth', 'AwareBEVDepth', 'DBEVDepth', 'AwareDBEVDepth']

        # Initial neural network for BEV & img position embedding
        #self.img_position_embed = nn.Linear(3, 3)   # input_dim = coordinate_dim = 3, output_dim = embed_dim
        self.count = 0
        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights() 

    def extract_camera_features(
        self,
        x_all,
        points,
        radar_points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        gt_depths=None,
    ) -> torch.Tensor:
        start_time = time.time()
        x = x_all[:,0:6,:,:,:]
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)
        x_original = x.clone()
        # Sample edge pixels of the image
        #resized_x = F.interpolate(x, size=(640, 640), mode='bilinear', align_corners=False)

        #det_out, da_seg_out, ll_seg_out = self.seg_model(resized_x)
        #ll_seg_out = F.interpolate(ll_seg_out, size=(H,W), mode='bilinear',align_corners=False)
        #ll_seg_mask = (ll_seg_out > 0.5).to(torch.uint8).to(torch.float32)
        #da_seg_out = F.interpolate(da_seg_out, size=(H,W), mode='bilinear',align_corners=False)
        #da_seg_mask = (da_seg_out > 0.5).to(torch.uint8).to(torch.float32)
    
        #mask_map = torch.max(ll_seg_mask[:,1,:,:], da_seg_mask[:,1,:,:]) 
        # Randomly sample some pixels
        mask_map = torch.zeros(B,6,H,W).cuda()
        x_extract = x_all[:,10:16,0,:,:]
        mask_map[:,0,:,:] = x_extract[:,3,:,:]
        mask_map[:,1,:,:] = x_extract[:,5,:,:]
        mask_map[:,2,:,:] = x_extract[:,4,:,:]
        mask_map[:,3,:,:] = x_extract[:,0,:,:]
        mask_map[:,4,:,:] = x_extract[:,1,:,:]
        mask_map[:,5,:,:] = x_extract[:,2,:,:]
        mask_map = mask_map.view(B*N,H,W)
        mask_map = (mask_map <= 0).float()
        #plt.imshow(mask_map[0,:,:].detach().cpu().numpy())
        #plt.axis('off')
        #plt.show()
        # Do img transformation here
        # Randomly sample some pixels
        num_samples = 30000
        #total_elements = mask_map.numel()
        #indices = torch.randperm(total_elements, device=mask_map.device)[:num_samples]
        
        flat_mask_map = mask_map.reshape(-1)  # Ensure a view of the original tensor
        #flat_mask_map[indices] = 1
        #mask_map = flat_mask_map.view(mask_map.shape)
        #dilated_lane_map = mask_map.unsqueeze(1).expand(-1, 3, -1, -1)

        # Importance Sample
        nonzero_indices = mask_map.nonzero(as_tuple=True)
        center_y = nonzero_indices[1].float().mean()
        center_x = nonzero_indices[2].float().mean()
        y_indices, x_indices = torch.meshgrid(torch.arange(H), torch.arange(W))
        flat_y_indices = y_indices.flatten().cuda()
        flat_x_indices = x_indices.flatten().cuda()
        distance_map = torch.sqrt((flat_y_indices - center_y)**2 + (flat_x_indices - center_x)**2)
        sigma = 1.0
        weights = torch.exp(-distance_map**2 / (2*sigma**2))
        weights /= weights.sum()
        indices = torch.multinomial(weights, num_samples, replacement=False)
        flat_mask_map[indices] = 1
        mask_map = flat_mask_map.view(mask_map.shape)
        dilated_lane_map = mask_map.unsqueeze(1).expand(-1, 3, -1, -1)
       
        # Generate the position embedding
        pos_x = torch.arange(W, dtype=torch.float32, device='cuda').view(1, 1, 1, W)
        pos_y = torch.arange(H, dtype=torch.float32, device='cuda').view(1, 1, H, 1)

        # Precompute divisors
        dim = 256
        div_term = 10000 ** (torch.arange(0, dim, 2, dtype=torch.float32, device='cuda') / dim)

        # Compute sine and cosine positional encodings
        sin_y = torch.sin(pos_y / div_term.view(1, -1, 1, 1))
        cos_x = torch.cos(pos_x / div_term.view(1, -1, 1, 1))

        # Concatenate to form the positional encoding
        pos_enc = torch.zeros((B * N, dim, H, W), device='cuda')
        pos_enc[:, 0::2, :, :] = sin_y
        pos_enc[:, 1::2, :, :] = cos_x
        
        
        pos_enc = torch.tensor(pos_enc,dtype=torch.float16).cuda()
        img_pe = pos_enc.view(B*N,-1,H,W)
        
        # Do transformer fusion here
        transformer_features = []
        final_features = []
        # Add on a positional embedding here
        # Cross-attention here
        #mask_map = torch.ones(B*N,H,W).cuda()
        
        for i in range(len(x)):
            if torch.any(mask_map[i]!=0) == True:
                output = self.input_cross_transformer(x ,feat_tag=i, pe=img_pe,mask_map=mask_map[i])
            else:
                mask_map[i] = torch.ones(H,W)
                output = self.input_cross_transformer(x ,feat_tag=i, pe=img_pe,mask_map=mask_map[i])
            transformer_features.append(output)
        
        #print('transformer take:',end-start)

        final_features = torch.cat(transformer_features,dim=0)
        
        dilated_lane_map = F.max_pool2d(final_features, 3, stride=1, padding=1)
        
        x = 0.5*x_original + 0.5*dilated_lane_map
    
        #x = final_features+x_original
        #if self.count % 100 == 0:
        #plt.imshow(dilated_lane_map[0,:,:,:].permute(1,2,0).detach().cpu().numpy())
        #plt.axis('off')
        #plt.show()
        #plt.imshow(x[0,:,:,:].permute(1,2,0).detach().cpu().numpy())
        #plt.axis('off')
        #plt.show()
        
        x = self.encoders["camera"]["backbone"](x)  #[6,192,32,88]

        x = self.encoders["camera"]["neck"](x)  #[6,256,32,88]
        
        #print('Vehicle Feature Extract takes:',end_time - start_time)

        
        # Augment the neck features
        non_mask = dilated_lane_map!=0
        non_zero_lane_value = dilated_lane_map[non_mask]
        dilated_lane_map_min = non_zero_lane_value.min()
        dilated_lane_map_max = non_zero_lane_value.max()
        normalized_lane_value = (non_zero_lane_value - dilated_lane_map_min) / (dilated_lane_map_max - dilated_lane_map_min +1e-6)
        normalized_lane_map = dilated_lane_map.clone()
        normalized_lane_map[non_mask] = normalized_lane_value
        
        resize_dilated_map = F.interpolate(normalized_lane_map, size=(32,88), mode='bilinear', align_corners=False)
        resize_dilated_map = resize_dilated_map.float().cuda() 
        lane_map_expand = resize_dilated_map.repeat(1, 256 // resize_dilated_map.shape[1], 1, 1)
        if 256 % 3 != 0:
            extra_channels = 256 - (256 // 3) * 3
            lane_map_expand = torch.cat([lane_map_expand, resize_dilated_map[:, :extra_channels, :, :]], dim=1)
        
        x = list(x)  # Convert to list
        x[0] = x[0] + lane_map_expand  # Modify the tensor in place
        x = tuple(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x,depth_loss = self.encoders["camera"]["vtransform"](
            x,
            x_original,
            points,
            radar_points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
            depth_loss=self.use_depth_loss, 
            gt_depths=gt_depths,
        )
        end_time = time.time()
        #print("Vehicle viewtransform takes:",end_time - start_time)
        
        return x, depth_loss
    def extract_infra_features(
        self,
        x_all,
        points,
        radar_points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        gt_depths=None,
    ) -> torch.Tensor:
        start_time = time.time()
        x = x_all[:,0:4,:,:,:]
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)
        x_original = x.clone()
        # Sample edge pixels of the image
        # mask_map cost 0.04 - 0.1 s

        # Revert normalization
        mask_map = x_all[:,10:14,0,:,:]
        mask_map = mask_map.view(B*N,H,W)
        mask_map = (mask_map <= 0).float()

        # Randomly sample some pixels

        num_samples = 20000
        #total_elements = mask_map.numel()
        #indices = torch.randperm(total_elements, device=mask_map.device)[:num_samples]

        flat_mask_map = mask_map.reshape(-1)  # Ensure a view of the original tensor
        #flat_mask_map[indices] = 1
        #mask_map = flat_mask_map.view(mask_map.shape)
        #dilated_lane_map = mask_map.unsqueeze(1).expand(-1, 3, -1, -1)

        # Importance Sample
        nonzero_indices = mask_map.nonzero(as_tuple=True)
        center_y = nonzero_indices[1].float().mean()
        center_x = nonzero_indices[2].float().mean()
        y_indices, x_indices = torch.meshgrid(torch.arange(H), torch.arange(W))
        flat_y_indices = y_indices.flatten().cuda()
        flat_x_indices = x_indices.flatten().cuda()
        distance_map = torch.sqrt((flat_y_indices - center_y)**2 + (flat_x_indices - center_x)**2)
        sigma = 1.0
        weights = torch.exp(-distance_map**2 / (2*sigma**2))
        weights /= weights.sum()
        indices = torch.multinomial(weights, num_samples, replacement=False)
        flat_mask_map[indices] = 1
        mask_map = flat_mask_map.view(mask_map.shape)
        dilated_lane_map = mask_map.unsqueeze(1).expand(-1, 3, -1, -1)

        # Create a Gaussian kernel for local weighting
        
        # Generate the position embedding
        # pe generation cost 0.0007

        pos_x = torch.arange(W, dtype=torch.float32, device='cuda').view(1, 1, 1, W)
        pos_y = torch.arange(H, dtype=torch.float32, device='cuda').view(1, 1, H, 1)

        # Precompute divisors
        dim = 256
        div_term = 10000 ** (torch.arange(0, dim, 2, dtype=torch.float32, device='cuda') / dim)

        # Compute sine and cosine positional encodings
        sin_y = torch.sin(pos_y / div_term.view(1, -1, 1, 1))
        cos_x = torch.cos(pos_x / div_term.view(1, -1, 1, 1))

        # Concatenate to form the positional encoding
        pos_enc = torch.zeros((B * N, dim, H, W), device='cuda')
        pos_enc[:, 0::2, :, :] = sin_y
        pos_enc[:, 1::2, :, :] = cos_x
        
        
        pos_enc = torch.tensor(pos_enc,dtype=torch.float16).cuda()
        img_pe = pos_enc.view(B*N,-1,H,W)


        transformer_features = []
        # Do transformer fusion here
        # transformer take time around 0.1-0.2s

        # Cross-attention here
        final_features = []
        start = time.time()
        #mask_map = torch.ones(B*N,H,W).cuda()
        for i in range(len(x)):
            if torch.any(mask_map[i]!=0) == True:
                output = self.input_cross_transformer_infra(x,feat_tag=i, pe=img_pe, mask_map=mask_map[i])
            else:
                mask_map[i] = torch.ones(H,W)
                output = self.input_cross_transformer_infra(x,feat_tag=i, pe=img_pe, mask_map=mask_map[i])
                
            transformer_features.append(output)
        end = time.time()
        #print('infra transformer takes:',end - start)
        final_features = torch.cat(transformer_features,dim=0)
        
        dilated_lane_map = F.max_pool2d(final_features, 3, stride=1, padding=1)
        
        x = 0.5*x_original + 0.5*dilated_lane_map

        #plt.imshow(dilated_lane_map[0,:,:,:].permute(1,2,0).detach().cpu().numpy())
        #plt.axis('off')
        #plt.show()
        #plt.imshow(x[0,:,:,:].permute(1,2,0).detach().cpu().numpy())
        #plt.axis('off')
        #plt.show()

        x = self.encoders["infra"]["backbone"](x)
        x = self.encoders["infra"]["neck"](x)

        end_time = time.time()
        #print("Infrastructure feature extraction takes time:", end_time - start_time)
        # Augment the neck_features

        start_time = time.time()
        non_mask = dilated_lane_map!=0
        non_zero_lane_value = dilated_lane_map[non_mask]
        dilated_lane_map_min = non_zero_lane_value.min()
        dilated_lane_map_max = non_zero_lane_value.max()
        normalized_lane_value = (non_zero_lane_value - dilated_lane_map_min) / (dilated_lane_map_max - dilated_lane_map_min +1e-6)
        normalized_lane_map = dilated_lane_map.clone()
        normalized_lane_map[non_mask] = normalized_lane_value
        
        resize_dilated_map = F.interpolate(normalized_lane_map, size=(32,88), mode='bilinear', align_corners=False)
        resize_dilated_map = resize_dilated_map.float().cuda() 
        lane_map_expand = resize_dilated_map.repeat(1, 256 // resize_dilated_map.shape[1], 1, 1)
        if 256 % 3 != 0:
            extra_channels = 256 - (256 // 3) * 3
            lane_map_expand = torch.cat([lane_map_expand, resize_dilated_map[:, :extra_channels, :, :]], dim=1)
       
        x = list(x)  # Convert to list
        x[0] = x[0] + lane_map_expand  # Modify the tensor in place
        x = tuple(x)
        
        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        # vtransformer takes around 0.6s -1s
        
        x, depth_loss = self.encoders["infra"]["vtransform"](
            x,
            x_original,
            points,
            radar_points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
            depth_loss=self.use_depth_loss, 
            gt_depths=gt_depths,
        )   # [1,80,128, 128]
        #print(x.shape, torch.sum(x==0))
        #plt.imshow(x[0,0,:,:].detach().cpu().numpy())
        #plt.axis('off')
        #plt.show()
        end_time = time.time()
        #print('infrastructure view transform takes:',end_time - start_time)
        return x, depth_loss
    
    def extract_features(self, x, sensor) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x, sensor)
        batch_size = coords[-1, 0] + 1
        x = self.encoders[sensor]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x
    
    # def extract_lidar_features(self, x) -> torch.Tensor:
    #     feats, coords, sizes = self.voxelize(x)
    #     batch_size = coords[-1, 0] + 1
    #     x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
    #     return x

    # def extract_radar_features(self, x) -> torch.Tensor:
    #     feats, coords, sizes = self.radar_voxelize(x)
    #     batch_size = coords[-1, 0] + 1
    #     x = self.encoders["radar"]["backbone"](feats, coords, batch_size, sizes=sizes)
    #     return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points, sensor):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders[sensor]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    # @torch.no_grad()
    # @force_fp32()
    # def radar_voxelize(self, points):
    #     feats, coords, sizes = [], [], []
    #     for k, res in enumerate(points):
    #         ret = self.encoders["radar"]["voxelize"](res)
    #         if len(ret) == 3:
    #             # hard voxelize
    #             f, c, n = ret
    #         else:
    #             assert len(ret) == 2
    #             f, c = ret
    #             n = None
    #         feats.append(f)
    #         coords.append(F.pad(c, (1, 0), mode="constant", value=k))
    #         if n is not None:
    #             sizes.append(n)

    #     feats = torch.cat(feats, dim=0)
    #     coords = torch.cat(coords, dim=0)
    #     if len(sizes) > 0:
    #         sizes = torch.cat(sizes, dim=0)
    #         if self.voxelize_reduce:
    #             feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
    #                 -1, 1
    #             )
    #             feats = feats.contiguous()

    #     return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                depths,
                radar,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths=None,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        features = []
        auxiliary_losses = {}
        start_time = time.time()
        
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                # Extract on-board vehicle img
                
                img_veh = img[:,:16]
                camera2ego_veh = camera2ego[:,:6]
                lidar2camera_veh = lidar2camera[:,:6]
                lidar2image_veh = lidar2image[:,:6]
                camera_intrinsics_veh = camera_intrinsics[:,:6]
                camera2lidar_veh = camera2lidar[:,:6]
                img_aug_matrix_veh = img_aug_matrix[:,:6]
               
                feature, depth_loss = self.extract_camera_features(
                    img_veh,
                    points,
                    radar,
                    camera2ego_veh,
                    lidar2ego,
                    lidar2camera_veh,
                    lidar2image_veh,
                    camera_intrinsics_veh,
                    camera2lidar_veh,
                    img_aug_matrix_veh,
                    lidar_aug_matrix,
                    metas,
                    gt_depths=depths,
                )
                torch.cuda.empty_cache()
                if self.use_depth_loss:
                    feature, auxiliary_losses['depth'] = feature[0], feature[-1]

            elif sensor == "lidar":  ## take 0.1s
                feature = self.extract_features(points, sensor)
            elif sensor == "radar":
                feature = self.extract_features(radar, sensor)
            elif sensor == "infra":
                img_infra = img[:,6:]
                camera2ego_infra = camera2ego[:,6:10]
                lidar2camera_infra = lidar2camera[:,6:10]
                lidar2image_infra = lidar2image[:,6:10]
                camera_intrinsics_infra = camera_intrinsics[:,6:10]
                camera2lidar_infra = camera2lidar[:,6:10]
                img_aug_matrix_infra = img_aug_matrix[:,6:10]
                extra_trans_rot = torch.eye(4).unsqueeze(0).cuda()
                

                feature, depth_loss_infra = self.extract_infra_features(
                    img_infra,
                    points,
                    radar,
                    camera2ego_infra,
                    lidar2ego,
                    lidar2camera_infra,
                    lidar2image_infra,
                    camera_intrinsics_infra,
                    camera2lidar_infra,
                    img_aug_matrix_infra,
                    lidar_aug_matrix,
                    metas,
                    gt_depths=depths,
                )
                torch.cuda.empty_cache()
  
            else:
                raise ValueError(f"unsupported sensor: {sensor}")

            features.append(feature)

        if not self.training:
            # avoid OOM
            features = features[::-1]

        if self.fuser is not None:
            ## This part is for attention method
            
            None_check = any(x is None for x in features)
            if None_check == True:
                x = features[0]
            else:
                x = self.fuser(features)
            
        else:
            assert len(features) == 1, features
            x = features[0]
        #plt.imshow(x[0,0,:,:].detach().cpu().numpy())
        #plt.axis('on')
        #plt.show()
        
        batch_size = x.shape[0]
        
        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)
       
        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    #x[0] = x[0].float()
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                    
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                    depth_losses = depth_loss + depth_loss_infra
                    for key in losses:
                        losses[key] += depth_losses 
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            if self.use_depth_loss:
                if 'depth' in auxiliary_losses:
                    outputs["loss/depth"] = auxiliary_losses['depth']
                else:
                    raise ValueError('Use depth loss is true, but depth loss not found')
                
            ## This part is for gradient debug use
            #for idx, (name, param) in enumerate(self.named_parameters()):
                #if 384 <= idx <= 483:  # Check if index falls in the range
                    #print(f"Index: {idx}, Parameter Name: {name}, Shape: {param.shape}")
            #for name, param in self.named_parameters():
                #if name == 'input_cross_transformer.layers.0.self_attn.o_proj.weight' and param.grad is not None:
                    #print(param.grad.norm().item())
                #if param.grad is None:
                    #print(f"Parameter {name} did not receive a gradient")
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    #x[0] = x[0].float()
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs

