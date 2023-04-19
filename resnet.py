import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class Resnet(nn.Module):
    def __init__(self, model_name: str = "base"):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            in_chans=1,
            drop_path_rate=0.2,
        )

        feature_info = self.backbone.feature_info

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
        features = []
        x = self.backbone.maxpool(
            self.backbone.act1(self.backbone.bn1(self.backbone.conv1(images)))
        )
        x = self.backbone.layer1(x)
        features.append(self.aux_block1(x))
        x = self.backbone.layer2(x)
        features.append(self.aux_block2(x))
        x = self.backbone.layer3(x)
        features.append(self.aux_block4(x))
        x = self.backbone.layer4(x)
        x = self.backbone.global_pool(x)
        features.append(x)
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


if __name__ == "__main__":
    model = Resnet("resnet50d")
    x = torch.randn(2, 1, 224, 224)
    y = torch.as_tensor([0, 1])
    model(x, y)
