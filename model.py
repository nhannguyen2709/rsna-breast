from typing import List, Tuple, Union

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from modeling import pad_tensor, unpad_tensor

# from timm.models.convnext import ConvNeXtBlock
# from timm.models.layers import LayerNorm


class Efficientnet(nn.Module):
    def __init__(self, model_name: str = "base"):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            in_chans=1,
            drop_path_rate=0.2,
        )

        feature_info = self.backbone.feature_info
        self.block_out_idx = [1, 2, 4]

        self.aux_block1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.BatchNorm1d(feature_info[1]["num_chs"])
        )
        self.aux_block2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.BatchNorm1d(feature_info[2]["num_chs"])
        )
        self.aux_block4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.BatchNorm1d(feature_info[3]["num_chs"])
        )

        self.aux_linear1 = nn.Linear(feature_info[1]["num_chs"], 1)
        self.aux_linear2 = nn.Linear(feature_info[2]["num_chs"], 1)
        self.aux_linear4 = nn.Linear(feature_info[3]["num_chs"], 1)
        self.num_features = (
            self.backbone.num_features
            + feature_info[1]["num_chs"]
            + feature_info[2]["num_chs"]
            + feature_info[3]["num_chs"]
        )
        self.linear = nn.Linear(self.num_features, 1)

    def forward_features(self, images: torch.Tensor):
        x = self.backbone.conv_stem(images)
        x = self.backbone.bn1(x)
        features = []
        for i, b in enumerate(self.backbone.blocks):
            x = b(x)
            if i in self.block_out_idx:
                features.append(x)

        features[0] = self.aux_block1(features[0])
        features[1] = self.aux_block2(features[1])
        features[2] = self.aux_block4(features[2])
        features.append(self.backbone.global_pool(self.backbone.bn2(self.backbone.conv_head(x))))
        return features

    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        features = self.forward_features(images)
        logits_block1 = self.aux_linear1(features[0])
        logits_block2 = self.aux_linear2(features[1])
        logits_block4 = self.aux_linear4(features[2])
        features = torch.cat(features, 1)
        logits = self.linear(features)
        loss = (
            F.binary_cross_entropy_with_logits(logits.view(-1), labels.float())
            + F.binary_cross_entropy_with_logits(logits_block1.view(-1), labels.float()) * 0.25
            + F.binary_cross_entropy_with_logits(logits_block2.view(-1), labels.float()) * 0.5
            + F.binary_cross_entropy_with_logits(logits_block4.view(-1), labels.float()) * 0.75
        )
        return loss, logits, features


class Convnext(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()

        self.backbone: nn.Module = timm.create_model(
            model_name,
            in_chans=1,
            pretrained=True,
        )
        self.backbone.reset_classifier(0, "")
        self.num_features = self.backbone.num_features
        self.linear = nn.Linear(self.num_features, 1)

    def forward_features(self, images: torch.Tensor):
        features = self.backbone(images)
        features = features.mean(dim=(2, 3))
        return features

    @autocast()
    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        features = self.forward_features(images)
        logits = self.linear(features)
        loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.float())
        return loss, logits, features


@autocast()
# @torch.inference_mode()
def extract_features(single_image_model, images):
    features = single_image_model.forward_features(images)
    if isinstance(features, list):
        features = torch.cat(features, 1)
    return features


def build_model(model_name: str):
    if "efficientnet" in model_name:
        return Efficientnet(model_name)
    elif "convnext" in model_name:
        return Convnext(model_name)


class MultiImageModel(nn.Module):
    def __init__(
        self,
        single_image_model: Union[Efficientnet, Convnext],
        transformer_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.25,
    ):
        super().__init__()

        for n, p in single_image_model.named_parameters():
            if n.startswith(
                ("backbone.stem", "backbone.stages.0", "backbone.stages.1")
            ):
                p.requires_grad = False
        self.single_image_model = single_image_model
        del self.single_image_model.linear

        self.projection = nn.Linear(self.single_image_model.num_features, transformer_dim)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            dim_feedforward=2 * transformer_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.linear = nn.Linear(self.single_image_model.num_features + transformer_dim, 1)

    @autocast()
    def forward(self, images: torch.Tensor, lengths: List[int], labels: torch.Tensor):
        features = extract_features(self.single_image_model, images)
        proj_features = self.projection(features)
        proj_features = torch.split(proj_features, lengths)
        proj_pad_features, pad_mask = pad_tensor(proj_features, lengths)
        trans_features = self.transformer(
            proj_pad_features, src_mask=None, src_key_padding_mask=pad_mask
        )
        trans_features = unpad_tensor(trans_features, lengths)
        features = torch.split(features, lengths)
        pooled_feats = torch.stack(
            [torch.cat([f, tf], dim=1).sum(0) for f, tf in zip(features, trans_features)]
        )
        logits = self.linear(pooled_feats)
        loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.float())
        return loss, logits, pooled_feats


# class NextViT(nn.Module):
#     def __init__(self, model_name: str = "nextvit_base"):
#         super().__init__()

#         from nextvit.nextvit import nextvit_base, nextvit_large, nextvit_small

#         if model_name == "nextvit_base":
#             self.backbone = nextvit_base(in_chans=1, use_checkpoint=True)
#             state_dict = torch.load("nextvit-checkpoints/nextvit_base_in1k6m_384.pth", "cpu").pop(
#                 "model"
#             )
#         state_dict["stem.0.conv.weight"] = state_dict["stem.0.conv.weight"].sum(
#             dim=1, keepdim=True
#         )
#         self.backbone.load_state_dict(state_dict)

#         stage_out_channels = self.backbone.stage_out_channels
#         self.aux_block0 = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.BatchNorm1d(stage_out_channels[0][-1])
#         )
#         self.aux_block1 = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.BatchNorm1d(stage_out_channels[1][-1])
#         )
#         self.aux_block2 = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.BatchNorm1d(stage_out_channels[2][-1])
#         )
#         self.aux_linear0 = nn.Linear(stage_out_channels[0][-1], 1)
#         self.aux_linear1 = nn.Linear(stage_out_channels[1][-1], 1)
#         self.aux_linear2 = nn.Linear(stage_out_channels[2][-1], 1)
#         self.linear = nn.Linear(
#             stage_out_channels[0][-1]
#             + stage_out_channels[1][-1]
#             + stage_out_channels[2][-1]
#             + stage_out_channels[3][-1],
#             1,
#         )

#     def forward(self, images, labels):
#         features = self.backbone.forward_features(images)
#         features[0] = self.aux_block0(features[0])
#         features[1] = self.aux_block1(features[1])
#         features[2] = self.aux_block2(features[2])
#         features[3] = torch.flatten(self.backbone.avgpool(features[3]), 1)
#         logits0 = self.aux_linear0(features[0])
#         logits1 = self.aux_linear1(features[1])
#         logits2 = self.aux_linear2(features[2])
#         features = torch.cat(features, 1)
#         logits = self.linear(features)
#         loss = (
#             F.binary_cross_entropy_with_logits(logits.view(-1), labels.float())
#             + F.binary_cross_entropy_with_logits(logits0.view(-1), labels.float()) * 0.25
#             + F.binary_cross_entropy_with_logits(logits1.view(-1), labels.float()) * 0.5
#             + F.binary_cross_entropy_with_logits(logits2.view(-1), labels.float()) * 0.75
#         )
#         return loss, logits, features
