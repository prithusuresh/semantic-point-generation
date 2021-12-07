import numpy as np
from pcdet.config import cfg, cfg_from_yaml_file
from eval_utils.eval_utils import load_data_to_gpu
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from pcdet.utils import common_utils
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu, points_in_boxes_cpu
import plotvoxel
import sys

cfg = cfg_from_yaml_file("cfgs/kitti_models/spg.yaml", cfg)
model_cfg = cfg.MODEL
for p in cfg.DATA_CONFIG.DATA_PROCESSOR:
    if p.NAME == "transform_points_to_smaller_voxels":
        small_voxel_size = p.VOXEL_SIZE

model_info_dict = {'module_list': [], 
                   'num_rawpoint_features': 4, 
                   'num_point_features': 4, 
                   'grid_size': np.array([432, 496, 1]), 
                   'point_cloud_range': np.array([  0.  , -39.68,  -3.  ,  69.12,  39.68,   1.  ]), 
                   'voxel_size': [0.16, 0.16, 4],
                   "small_voxel_size": small_voxel_size, 
                   'z_voxel_size': small_voxel_size[-1],
                   'depth_downsample_factor': None}

model_info_dict["out_classes"] = int(abs(cfg.DATA_CONFIG.POINT_CLOUD_RANGE[2] - cfg.DATA_CONFIG.POINT_CLOUD_RANGE[5])/model_info_dict["z_voxel_size"])

class VFETemplate(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

    def get_output_feature_dim(self):
        raise NotImplementedError

    def forward(self, **kwargs):
        """
        Args:
            **kwargs:

        Returns:
            batch_dict:
                ...
                vfe_features: (num_voxels, C)
        """
        raise NotImplementedError

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict
    
class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict

class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x
        return data_dict



class ConvClassificationHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvClassificationHead, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 4, padding = 1, stride = 2)
        self.act = nn.Sigmoid()  #what should it be
    def forward(self, batch_dict):

        spatial_features_2d = batch_dict["spatial_features_2d"]
        output_prob = self.conv(spatial_features_2d)
        if not self.training:
            output_prob = self.act(output_prob)
        batch_dict["output_prob"] = output_prob
        return batch_dict
        
    
class SPG_CLASSIFICATION(nn.Module):
    def __init__(self):
        super(SPG_CLASSIFICATION, self).__init__()
        self.vfe_module = PillarVFE(
            model_cfg = model_cfg.VFE,
            num_point_features=model_info_dict['num_rawpoint_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            grid_size=model_info_dict['grid_size'],
            depth_downsample_factor=model_info_dict['depth_downsample_factor']
        )
        model_info_dict['num_point_features'] = self.vfe_module.get_output_feature_dim()
        model_info_dict['module_list'].append(self.vfe_module)
        
        self.map_to_bev_module = PointPillarScatter(
            model_cfg = model_cfg.MAP_TO_BEV,
            grid_size = model_info_dict['grid_size']
        )
        model_info_dict['module_list'].append(self.map_to_bev_module)
        model_info_dict['num_bev_features'] = self.map_to_bev_module.num_bev_features
        
        self.backbone_2d_module = BaseBEVBackbone(
            model_cfg=model_cfg.BACKBONE_2D,
            input_channels=model_info_dict['num_bev_features']
        )
        model_info_dict['module_list'].append(self.backbone_2d_module)
        model_info_dict['num_bev_features'] = self.backbone_2d_module.num_bev_features
        
        self.classification_head = ConvClassificationHead(model_info_dict["num_bev_features"],model_info_dict["out_classes"])
    def forward(self, batch_dict):
        batch_dict = self.vfe_module(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d_module(batch_dict)
        batch_dict = self.classification_head(batch_dict)
        return batch_dict

from train_utils import train_utils
import pandas as pd

if __name__ == "__main__":
    model = SPG_CLASSIFICATION()
    model.cuda()
    train_sample = pickle.load(open("voxelized_train_sample.p", "rb"))
    print("Train Sample = ", train_sample["points"].shape)
    #load data to gpu
    load_data_to_gpu(train_sample)
    ckpt = torch.load(sys.argv[1], map_location = "cuda")
    model.load_state_dict(ckpt["model_state"])
    #make forward pass
    # breakpoint()
    model.eval()
    out_dict = model(train_sample)
    gt,b = train_utils.prepare_ground_truth_and_weight_mask(out_dict,cfg.OPTIMIZATION)
    a_coords = torch.where(gt[0]>0.5)
    # print("a0 = = ",a.shape
    # breakpoint()
    gt_dict = {
        "z": a_coords[0].cpu().numpy(),
        "y": a_coords[1].cpu().numpy(),
        "x": a_coords[2].cpu().numpy(),
        "v": np.ones(a_coords[0].shape)
    }
    # print("GT ===== ",gt_dict["v"])
    # breakpoint()
    df_ground_truth = pd.DataFrame(gt_dict)
    print("GT ===== ",df_ground_truth.head())
    # breakpoint()
    df_ground_truth.to_csv("groundtruth.csv", index = False) 
    breakpoint()
    print("Ground Truth = ",a.shape)
    print("Weight Mask  = ",b.shape)
    # breakpoint()
    pred = out_dict["output_prob"]
    print("Pred shape = ",pred.shape)
    coords = torch.where(pred[0] > 0.5)
    print("Coords shape =",coords)
    breakpoint()
    df = {
        "z": coords[0].cpu().numpy(),
        "y": coords[1].cpu().numpy(),
        "x": coords[2].cpu().numpy(),
        "p": pred[0,coords[0], coords[1], coords[2]].detach().cpu().numpy()
    }
    print("DF ===== ",df.keys())
    df = pd.DataFrame(df)
    print("DF ===== ",df.head())

    breakpoint()

    df.to_csv("occupied_voxels.csv", index = False) 
    print("Keys = ",out_dict.keys())
    hidden_voxel = out_dict["hidden_voxel_coords"]
    
    
    
    
    plotvoxel.visualizeVoxels(hidden_voxel)
    print("Hidden Voxels Shape = ",hidden_voxel[0].shape)

    print("Hidden Voxels[0] Shape = ",hidden_voxel[1000])

    # plotvoxel.visualizeVoxels(out_dict["hidden_voxel_coords"][0])
    # breakpoint()

    # #find ALL foreground voxels
    # plotvoxel.visualizeVoxels(pred)

    # from kornia.losses import BinaryFocalLossWithLogits
    # criterion = BinaryFocalLossWithLogits(alpha = 0.25, gamma = 2.0, reduction = "none")
    # loss = (weight_mask*criterion(pred, ground_truth)).mean()
    # print (loss.item())
