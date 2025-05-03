import math

import timm
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from SAR_C2QM.modules.c2q_attention import (
    MultiModalPyramidAttentionFusion,
    PyramidAttentionFusion,
)


class ConditionEncoders_RHV(nn.Module):
    def __init__(
        self,
        model_name="resnet50",
        sar_in_ch=5,
        geo_in_ch=None,
        sar_out_ch=5,
        geo_out_ch=None,
        in_size=256,
        out_size=64,
        ckpt_path=None,
    ):
        """条件编码器模块
        :param sar_in_ch: sar模态编码器输入通道数（模态1）
        :param geo_in_ch: 地形模态编码器输入通道数（模态2）
        :param out_ch: 编码器输出通道数
        :param in_size: 输入特征图大小
        :param out_size: 输出特征图大小
        """

        super().__init__()

        self.geo_in_ch = geo_in_ch

        # 动态选择编码器的中间层特征图输出索引，不要超过 out_size
        idx_min = int(math.log2(in_size / out_size)) - 1  # 计算最小索引，
        # 索引 0 - 初始卷积层输出的特征图 尺寸1/2  128
        # 索引 1 - 第一个残差块输出特征图 尺寸1/4  64
        # 索引 2 - 第二个残差块输出特征图 尺寸1/8  32
        # 索引 3 - 第三个残差块输出特征图 尺寸1/16 16
        # 索引 4 - 第四个残差块输出特征图 尺寸1/32
        # out_indices = [i for i in range(idx_min, 5)]
        # out_indices = [i for i in range(idx_min, 4)]
        out_indices = [idx_min]

        # 独立编码sar模态
        self.sar_encoder = timm.create_model(
            model_name=model_name,
            features_only=True,
            out_indices=out_indices,  # 使用中间层特征图
            in_chans=sar_in_ch,
        )
        if geo_in_ch is not None:
            # 独立编码地形模态 (DEM+坡度+坡向+局部入射角)
            self.geo_encoder = timm.create_model(
                model_name=model_name,
                features_only=True,
                out_indices=out_indices,  # 使用中间层特征图
                in_chans=geo_in_ch,
            )

        # print(self.sar_encoder)

        # 获取ResNet中间层特征图的通道数
        with torch.no_grad():
            dummy_sar = torch.zeros(1, sar_in_ch, in_size, in_size)
            sar_feats = self.sar_encoder(dummy_sar)
            sar_channels = [f.shape[1] for f in sar_feats]

            if geo_in_ch is not None:
                dummy_geo = torch.zeros(1, geo_in_ch, in_size, in_size)
                geo_feats = self.geo_encoder(dummy_geo)
                geo_channels = [f.shape[1] for f in geo_feats]

        # 特征融合层
        self.sar_proj_out = nn.Conv2d(sar_channels[0], sar_out_ch, kernel_size=1)
        if geo_in_ch is not None:
            self.geo_proj_out = nn.Conv2d(geo_channels[0], geo_out_ch, kernel_size=1)

        if ckpt_path is not None:
            cache = torch.load(ckpt_path, map_location="cpu")
            sd = cache["cond_encoder_state_dict"]
            self.load_state_dict(sd, strict=True)
            print(f"Restored from {ckpt_path}")


    def encode(self, cond):
        """条件编码函数，适配LDM模型调用
        Args:
            cond: 包含条件输入的批次数据
        Returns:
            多尺度特征图列表
        """
        if isinstance(cond, dict):
            sar = cond.get("modal_1", None)
            if sar is None:
                raise ValueError("输入字典中必须包含'modal_1'键")
            if self.geo_in_ch is not None: 
                geo = cond.get("modal_2", None)
                if geo is None:
                    raise ValueError("输入字典中必须包含'modal_2'键")

            sar_feats = self.sar_encoder(sar)
            sar_feat = self.sar_proj_out(sar_feats[0])
            if self.geo_in_ch is not None: 
                geo_feats = self.geo_encoder(geo)
                geo_feat = self.geo_proj_out(geo_feats[0])
                cond_feat = torch.cat([sar_feat, geo_feat], dim=1)
            else:
                cond_feat = sar_feat
            
            return cond_feat
        else:
            raise ValueError("输入必须是字典格式")


class ConditionEncoders_RHV_imp(nn.Module):
    def __init__(
        self,
        model_name="resnet50",
        sar_in_ch=4,
        geo_in_ch=5,
        out_ch=9,
        in_size=256,
        out_size=64,
        ckpt_path=None,
    ):
        """条件编码器模块
        :param sar_in_ch: sar模态编码器输入通道数（模态1）
        :param geo_in_ch: 地形模态编码器输入通道数（模态2）
        :param out_ch: 编码器输出通道数
        :param in_size: 输入特征图大小
        :param out_size: 输出特征图大小
        """

        super().__init__()

        self.geo_in_ch = geo_in_ch

        # 动态选择编码器的中间层特征图输出索引，不要超过 out_size
        idx_min = int(math.log2(in_size / out_size)) - 1  # 计算最小索引，
        # 索引 0 - 初始卷积层输出的特征图 尺寸1/2  128
        # 索引 1 - 第一个残差块输出特征图 尺寸1/4  64
        # 索引 2 - 第二个残差块输出特征图 尺寸1/8  32
        # 索引 3 - 第三个残差块输出特征图 尺寸1/16 16
        # 索引 4 - 第四个残差块输出特征图 尺寸1/32
        # out_indices = [i for i in range(idx_min, 5)]
        out_indices = [i for i in range(idx_min, 4)]
        # out_indices = [idx_min]

        # 独立编码sar模态
        self.sar_encoder = timm.create_model(
            model_name=model_name,
            features_only=True,
            out_indices=out_indices,  # 使用中间层特征图
            in_chans=sar_in_ch,
        )
        # 独立编码地形模态 (DEM+坡度+坡向+局部入射角)
        self.geo_encoder = timm.create_model(
            model_name=model_name,
            features_only=True,
            out_indices=out_indices,  # 使用中间层特征图
            in_chans=geo_in_ch,
        )

        # print(self.sar_encoder)

        # 获取ResNet中间层特征图的通道数
        with torch.no_grad():
            dummy_sar = torch.zeros(1, sar_in_ch, in_size, in_size)
            sar_feats = self.sar_encoder(dummy_sar)
            sar_channels = [f.shape[1] for f in sar_feats]

            dummy_geo = torch.zeros(1, geo_in_ch, in_size, in_size)
            geo_feats = self.geo_encoder(dummy_geo)
            geo_channels = [f.shape[1] for f in geo_feats]

        # 特征融合层
        self.fusion = MultiModalPyramidAttentionFusion(
            feature_dims_mod1=sar_channels,
            feature_dims_mod2=geo_channels,
            hidden_dim=256,
            num_heads=2,
            dropout=0.1,
            fusion_type=1,
        )

        self.proj_out = nn.Conv2d(sar_channels[0], out_ch, kernel_size=1)

        if ckpt_path is not None:
            cache = torch.load(ckpt_path, map_location="cpu")
            sd = cache["cond_encoder_state_dict"]
            self.load_state_dict(sd, strict=True)
            print(f"Restored from {ckpt_path}")


    def encode(self, cond):
        """条件编码函数，适配LDM模型调用
        Args:
            cond: 包含条件输入的批次数据
        Returns:
            多尺度特征图列表
        """
        if isinstance(cond, dict):
            sar = cond.get("modal_1", None)
            if sar is None:
                raise ValueError("输入字典中必须包含'modal_1'键")

            geo = cond.get("modal_2", None)
            if geo is None:
                raise ValueError("输入字典中必须包含'modal_2'键")

            sar_feats = self.sar_encoder(sar)
            geo_feats = self.geo_encoder(geo)
            enhanced_features = self.fusion(sar_feats, geo_feats)
            cond_feat = self.proj_out(enhanced_features[0])

            return cond_feat
        else:
            raise ValueError("输入必须是字典格式")
        


if __name__ == "__main__":
    model = ConditionEncoders_RHV()
    print(model)
    sar = torch.randn(1, 4, 256, 256)
    geo = torch.randn(1, 4, 256, 256)
    sar_feat, geo_feat = model(sar, geo)
    print(sar_feat[0].shape, geo_feat[0].shape)
