import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialAttention(nn.Module):
    """空间注意力模块：用于单一特征图内的自注意力计算"""
    def __init__(self, in_channels, hidden_dim=256, num_heads=8, dropout=0.1):
        super(SpatialAttention, self).__init__()
        
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim必须能被num_heads整除"
        
        # 投影层
        self.q_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x):
        """
        参数:
            x: 输入特征图，形状为 [B, C, H, W]
        返回:
            out: 注意力增强的特征图，形状为 [B, C, H, W]
        """
        batch_size, _, h, w = x.shape
        
        # 投影特征
        q = self.q_proj(x)  # [B, hidden_dim, H, W]
        k = self.k_proj(x)  # [B, hidden_dim, H, W]
        v = self.v_proj(x)  # [B, hidden_dim, H, W]
        
        # 重塑为多头形式
        q = q.reshape(batch_size, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2) # [B, num_heads, H*W, head_dim]
        k = k.reshape(batch_size, self.num_heads, self.head_dim, -1).permute(0, 1, 2, 3) # [B, num_heads, head_dim, H*W]
        v = v.reshape(batch_size, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2) # [B, num_heads, H*W, head_dim]
        
        # 计算注意力分数
        attn = torch.matmul(q, k)  # [B, num_heads, H*W, H*W]
        attn = attn * self.scale
        
        # 注意力权重归一化
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力权重
        out = torch.matmul(attn, v)  # [B, num_heads, H*W, head_dim]
        
        # 重塑回原始形状
        out = out.permute(0, 1, 3, 2).reshape(batch_size, self.hidden_dim, h, w)  # [B, num_heads, head_dim, H*W]
        
        # 最终投影
        out = self.out_proj(out)
        
        return out


class CrossScaleAttention(nn.Module):
    """跨尺度注意力模块：处理不同尺度特征之间的交互"""
    def __init__(self, in_channels_h, in_channels_l, hidden_dim=256, num_heads=8, dropout=0.1):
        """
        参数:
            in_channels_h: 高分辨率特征的通道数
            in_channels_l: 低分辨率特征的通道数
            hidden_dim: 注意力机制的隐藏维度
            num_heads: 多头注意力的头数
            dropout: Dropout概率
        """
        super(CrossScaleAttention, self).__init__()
        
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim必须能被num_heads整除"
        
        # 高分辨率特征投影
        self.q_high_proj = nn.Conv2d(in_channels_h, hidden_dim, kernel_size=1)
        self.k_high_proj = nn.Conv2d(in_channels_h, hidden_dim, kernel_size=1)
        self.v_high_proj = nn.Conv2d(in_channels_h, hidden_dim, kernel_size=1)
        
        # 低分辨率特征投影
        self.q_low_proj = nn.Conv2d(in_channels_l, hidden_dim, kernel_size=1)
        self.k_low_proj = nn.Conv2d(in_channels_l, hidden_dim, kernel_size=1)
        self.v_low_proj = nn.Conv2d(in_channels_l, hidden_dim, kernel_size=1)
        
        # 输出投影
        self.out_high_proj = nn.Conv2d(hidden_dim, in_channels_h, kernel_size=1)
        self.out_low_proj = nn.Conv2d(hidden_dim, in_channels_l, kernel_size=1)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, high_feat, low_feat):
        """
        参数:
            high_feat: 高分辨率特征，形状为 [B, C_h, H_h, W_h]
            low_feat: 低分辨率特征，形状为 [B, C_l, H_l, W_l]
        返回:
            high_out: 增强的高分辨率特征
            low_out: 增强的低分辨率特征
        """
        batch_size = high_feat.shape[0]
        
        # 统一空间尺寸（这里选择高分辨率）
        h, w = high_feat.shape[2:]
        low_feat_upsampled = F.interpolate(low_feat, size=(h, w), mode='bilinear', align_corners=False)
        
        # === 高分辨率特征关注低分辨率特征 ===
        
        # 特征投影
        q_high = self.q_high_proj(high_feat)
        k_low = self.k_low_proj(low_feat_upsampled)
        v_low = self.v_low_proj(low_feat_upsampled)
        
        # 重塑为多头形式
        q_high = q_high.view(batch_size, self.num_heads, self.head_dim, h * w)
        q_high = q_high.permute(0, 1, 3, 2)  # [B, num_heads, H*W, head_dim]
        
        k_low = k_low.view(batch_size, self.num_heads, self.head_dim, h * w)
        k_low = k_low.permute(0, 1, 2, 3)  # [B, num_heads, head_dim, H*W]
        
        v_low = v_low.view(batch_size, self.num_heads, self.head_dim, h * w)
        v_low = v_low.permute(0, 1, 3, 2)  # [B, num_heads, H*W, head_dim]
        
        # 计算注意力
        attn_high_low = torch.matmul(q_high, k_low)
        attn_high_low = attn_high_low * self.scale
        attn_high_low = F.softmax(attn_high_low, dim=-1)
        attn_high_low = self.dropout(attn_high_low)
        
        # 应用注意力权重
        high_out = torch.matmul(attn_high_low, v_low)
        high_out = high_out.permute(0, 1, 3, 2).contiguous()
        high_out = high_out.view(batch_size, self.hidden_dim, h, w)
        high_out = self.out_high_proj(high_out)
        
        # === 低分辨率特征关注高分辨率特征 ===
        
        # 将注意力结果下采样回低分辨率
        h_low, w_low = low_feat.shape[2:]
        
        # 特征投影
        q_low = self.q_low_proj(low_feat)
        k_high = self.k_high_proj(high_feat)
        v_high = self.v_high_proj(high_feat)
        
        # 下采样高分辨率特征的KV投影
        k_high_downsampled = F.interpolate(k_high, size=(h_low, w_low), mode='bilinear', align_corners=False)
        v_high_downsampled = F.interpolate(v_high, size=(h_low, w_low), mode='bilinear', align_corners=False)
        
        # 重塑为多头形式
        q_low = q_low.view(batch_size, self.num_heads, self.head_dim, h_low * w_low)
        q_low = q_low.permute(0, 1, 3, 2)  # [B, num_heads, H_l*W_l, head_dim]
        
        k_high_downsampled = k_high_downsampled.view(batch_size, self.num_heads, self.head_dim, h_low * w_low)
        k_high_downsampled = k_high_downsampled.permute(0, 1, 2, 3)  # [B, num_heads, head_dim, H_l*W_l]
        
        v_high_downsampled = v_high_downsampled.view(batch_size, self.num_heads, self.head_dim, h_low * w_low)
        v_high_downsampled = v_high_downsampled.permute(0, 1, 3, 2)  # [B, num_heads, H_l*W_l, head_dim]
        
        # 计算注意力
        attn_low_high = torch.matmul(q_low, k_high_downsampled)
        attn_low_high = attn_low_high * self.scale
        attn_low_high = F.softmax(attn_low_high, dim=-1)
        attn_low_high = self.dropout(attn_low_high)
        
        # 应用注意力权重
        low_out = torch.matmul(attn_low_high, v_high_downsampled)
        low_out = low_out.permute(0, 1, 3, 2).contiguous()
        low_out = low_out.view(batch_size, self.hidden_dim, h_low, w_low)
        low_out = self.out_low_proj(low_out)
        
        return high_out, low_out
    

class EfficientCrossScaleAttention(nn.Module):
    """内存优化的跨尺度注意力模块"""
    def __init__(self, in_channels_h, in_channels_l, hidden_dim=256, num_heads=8, dropout=0.1):
        super(EfficientCrossScaleAttention, self).__init__()
        
        # 减小隐藏维度以降低内存占用
        self.hidden_dim = hidden_dim // 2  # 减少一半的隐藏维度
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // num_heads
        assert self.head_dim * num_heads == self.hidden_dim, "hidden_dim/2必须能被num_heads整除"
        
        # 高分辨率特征投影（只保留必要的投影层）
        self.q_high_proj = nn.Conv2d(in_channels_h, self.hidden_dim, kernel_size=1)
        self.v_high_proj = nn.Conv2d(in_channels_h, self.hidden_dim, kernel_size=1)
        
        # 低分辨率特征投影
        self.k_low_proj = nn.Conv2d(in_channels_l, self.hidden_dim, kernel_size=1)
        self.v_low_proj = nn.Conv2d(in_channels_l, self.hidden_dim, kernel_size=1)
        
        # 输出投影
        self.out_high_proj = nn.Conv2d(self.hidden_dim, in_channels_h, kernel_size=1)
        self.out_low_proj = nn.Conv2d(self.hidden_dim, in_channels_l, kernel_size=1)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
        # 使用线性注意力降低计算复杂度
        self.use_linear_attention = False
    
    def linear_attention(self, q, k, v):
        """线性注意力机制，复杂度从O(N²)降至O(N)"""
        # 确保输入形状：q, k, v 都应当是 [B, num_heads, seq_len, head_dim]
        # k 需要转置为 [B, num_heads, head_dim, seq_len]
        
        q = q * self.scale
        
        # 应用softmax
        q = torch.softmax(q, dim=-1)
        k = torch.softmax(k, dim=-2)  # 在序列长度维度上进行softmax
        
        # 应用dropout
        q = self.dropout(q)
        k = self.dropout(k)
        
        # 正确的转置和矩阵乘法
        # k的形状是 [B, num_heads, head_dim, seq_len]
        # v的形状是 [B, num_heads, seq_len, head_dim]
        context = torch.matmul(k.transpose(-2, -1), v)  # 应得到 [B, num_heads, seq_len, head_dim]
        out = torch.matmul(q, context)
        
        return out
        
    def forward(self, high_feat, low_feat):
        batch_size = high_feat.shape[0]
        h_high, w_high = high_feat.shape[2:]
        h_low, w_low = low_feat.shape[2:]
        
        # 仅处理高分辨率特征关注低分辨率特征（单向注意力）
        # 这大幅减少计算量和内存占用
        
        # 特征投影
        q_high = self.q_high_proj(high_feat)
        
        # 直接在低分辨率下计算KV，避免上采样
        k_low = self.k_low_proj(low_feat)
        v_low = self.v_low_proj(low_feat)
        
        # 将q_high下采样到低分辨率空间
        q_high = F.interpolate(q_high, size=(h_low, w_low), mode='bilinear', align_corners=False)
        
        # 重塑为多头形式
        q_high = q_high.view(batch_size, self.num_heads, self.head_dim, h_low * w_low)
        q_high = q_high.permute(0, 1, 3, 2)  # [B, num_heads, H_l*W_l, head_dim]
        
        k_low = k_low.view(batch_size, self.num_heads, self.head_dim, h_low * w_low)
        k_low = k_low.permute(0, 1, 2, 3)  # [B, num_heads, head_dim, H_l*W_l]
        
        v_low = v_low.view(batch_size, self.num_heads, self.head_dim, h_low * w_low)
        v_low = v_low.permute(0, 1, 3, 2)  # [B, num_heads, H_l*W_l, head_dim]
        
        # 使用标准注意力或线性注意力
        if self.use_linear_attention:
            high_out = self.linear_attention(q_high, k_low, v_low)
        else:
            # 计算注意力（标准方法）
            attn = torch.matmul(q_high, k_low)
            attn = attn * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            high_out = torch.matmul(attn, v_low)
        
        # 重塑并上采样回高分辨率
        high_out = high_out.permute(0, 1, 3, 2).contiguous()
        high_out = high_out.view(batch_size, self.hidden_dim, h_low, w_low)
        high_out = F.interpolate(high_out, size=(h_high, w_high), mode='bilinear', align_corners=False)
        high_out = self.out_high_proj(high_out)
        
        # 简化的低分辨率特征更新（不使用注意力，仅使用卷积投影）
        # 这进一步降低了内存使用
        low_out = self.out_low_proj(v_low.permute(0, 1, 3, 2).contiguous().view(batch_size, self.hidden_dim, h_low, w_low))
        
        return high_out, low_out


class ModalCrossAttention(nn.Module):
    """模态间交叉注意力模块：处理不同模态特征的交互"""
    def __init__(self, in_channels_1, in_channels_2, hidden_dim=256, num_heads=8, dropout=0.1):
        super(ModalCrossAttention, self).__init__()
        
        # 减小隐藏维度以降低内存占用
        self.hidden_dim = hidden_dim // 2
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // num_heads
        assert self.head_dim * num_heads == self.hidden_dim, "hidden_dim/2必须能被num_heads整除"
        
        # 模态1特征投影
        self.q_mod1_proj = nn.Conv2d(in_channels_1, self.hidden_dim, kernel_size=1)
        self.k_mod1_proj = nn.Conv2d(in_channels_1, self.hidden_dim, kernel_size=1)
        self.v_mod1_proj = nn.Conv2d(in_channels_1, self.hidden_dim, kernel_size=1)
        
        # 模态2特征投影
        self.q_mod2_proj = nn.Conv2d(in_channels_2, self.hidden_dim, kernel_size=1)
        self.k_mod2_proj = nn.Conv2d(in_channels_2, self.hidden_dim, kernel_size=1)
        self.v_mod2_proj = nn.Conv2d(in_channels_2, self.hidden_dim, kernel_size=1)
        
        # 输出投影
        self.out_mod1_proj = nn.Conv2d(self.hidden_dim, in_channels_1, kernel_size=1)
        self.out_mod2_proj = nn.Conv2d(self.hidden_dim, in_channels_2, kernel_size=1)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, mod1_feat, mod2_feat):
        """
        参数:
            mod1_feat: 模态1特征，形状为 [B, C_1, H, W]
            mod2_feat: 模态2特征，形状为 [B, C_2, H, W]
        返回:
            mod1_out: 增强的模态1特征
            mod2_out: 增强的模态2特征
        """
        batch_size, _, h, w = mod1_feat.shape
        
        # 特征投影
        q_mod1 = self.q_mod1_proj(mod1_feat)
        k_mod1 = self.k_mod1_proj(mod1_feat)
        v_mod1 = self.v_mod1_proj(mod1_feat)
        
        q_mod2 = self.q_mod2_proj(mod2_feat)
        k_mod2 = self.k_mod2_proj(mod2_feat)
        v_mod2 = self.v_mod2_proj(mod2_feat)
        
        # 重塑为多头形式
        q_mod1 = q_mod1.reshape(batch_size, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        k_mod2 = k_mod2.reshape(batch_size, self.num_heads, self.head_dim, h * w).permute(0, 1, 2, 3)
        v_mod2 = v_mod2.reshape(batch_size, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        
        q_mod2 = q_mod2.reshape(batch_size, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        k_mod1 = k_mod1.reshape(batch_size, self.num_heads, self.head_dim, h * w).permute(0, 1, 2, 3)
        v_mod1 = v_mod1.reshape(batch_size, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        
        # 模态1关注模态2 (mod1 -> mod2)
        attn_1to2 = torch.matmul(q_mod1, k_mod2)
        attn_1to2 = attn_1to2 * self.scale
        attn_1to2 = F.softmax(attn_1to2, dim=-1)
        attn_1to2 = self.dropout(attn_1to2)
        
        mod1_out = torch.matmul(attn_1to2, v_mod2)
        mod1_out = mod1_out.permute(0, 1, 3, 2).contiguous().reshape(batch_size, self.hidden_dim, h, w)
        mod1_out = self.out_mod1_proj(mod1_out)
        
        # 模态2关注模态1 (mod2 -> mod1)
        attn_2to1 = torch.matmul(q_mod2, k_mod1)
        attn_2to1 = attn_2to1 * self.scale
        attn_2to1 = F.softmax(attn_2to1, dim=-1)
        attn_2to1 = self.dropout(attn_2to1)
        
        mod2_out = torch.matmul(attn_2to1, v_mod1)
        mod2_out = mod2_out.permute(0, 1, 3, 2).contiguous().reshape(batch_size, self.hidden_dim, h, w)
        mod2_out = self.out_mod2_proj(mod2_out)
        
        return mod1_out, mod2_out


class DyadicCrossAttention(nn.Module):
    """
    对偶交叉注意力：处理两个模态特征的交互融合
    增强版本支持多头注意力和dropout
    """
    def __init__(self, dim1, dim2, hidden_dim=256, num_heads=8, dropout=0.1, output_dim=None):
        """
        参数:
            dim1: 第一个模态特征的通道维度
            dim2: 第二个模态特征的通道维度
            hidden_dim: 注意力机制的隐藏维度
            num_heads: 多头注意力的头数
            dropout: Dropout概率
            output_dim: 输出特征的通道维度，默认与dim1相同
        """
        super(DyadicCrossAttention, self).__init__()
        
        if output_dim is None:
            output_dim = dim1
            
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim必须能被num_heads整除"
        
        # 为第一个模态特征创建Query、Key和Value投影
        self.modal1_query_proj = nn.Conv2d(dim1, hidden_dim, kernel_size=1)
        self.modal1_key_proj = nn.Conv2d(dim1, hidden_dim, kernel_size=1)
        self.modal1_value_proj = nn.Conv2d(dim1, hidden_dim, kernel_size=1)
        
        # 为第二个模态特征创建Query、Key和Value投影
        self.modal2_query_proj = nn.Conv2d(dim2, hidden_dim, kernel_size=1)
        self.modal2_key_proj = nn.Conv2d(dim2, hidden_dim, kernel_size=1)
        self.modal2_value_proj = nn.Conv2d(dim2, hidden_dim, kernel_size=1)
        
        # 输出投影层
        self.modal1_output_proj = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        self.modal2_output_proj = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        self.output_proj = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = self.head_dim ** -0.5
    
    def forward(self, modal1_feat, modal2_feat):
        """
        参数:
            modal1_feat: 第一个模态特征，形状为 [B, C1, H1, W1]
            modal2_feat: 第二个模态特征，形状为 [B, C2, H2, W2]
        返回:
            output1: 增强的第一个模态特征，形状为 [B, output_dim, H1, W1]
            output2: 增强的第二个模态特征，形状为 [B, output_dim, H2, W2]
        """
        # 提取尺寸信息
        B, C1, H1, W1 = modal1_feat.shape
        B, C2, H2, W2 = modal2_feat.shape
        
        # 创建模态1的query, key和value
        q1 = self.modal1_query_proj(modal1_feat)  # [B, hidden_dim, H1, W1]
        k1 = self.modal1_key_proj(modal1_feat)    # [B, hidden_dim, H1, W1]
        v1 = self.modal1_value_proj(modal1_feat)  # [B, hidden_dim, H1, W1]
        
        # 创建模态2的query, key和value
        q2 = self.modal2_query_proj(modal2_feat)  # [B, hidden_dim, H2, W2]
        k2 = self.modal2_key_proj(modal2_feat)    # [B, hidden_dim, H2, W2]
        v2 = self.modal2_value_proj(modal2_feat)  # [B, hidden_dim, H2, W2]
        
        # 重塑为多头形式
        # 模态1的张量重塑
        q1 = q1.reshape(B, self.num_heads, self.head_dim, H1 * W1).permute(0, 1, 3, 2)  # [B, num_heads, H1*W1, head_dim]
        k1 = k1.reshape(B, self.num_heads, self.head_dim, H1 * W1).permute(0, 1, 2, 3)  # [B, num_heads, head_dim, H1*W1]
        v1 = v1.reshape(B, self.num_heads, self.head_dim, H1 * W1).permute(0, 1, 3, 2)  # [B, num_heads, H1*W1, head_dim]
        
        # 模态2的张量重塑
        q2 = q2.reshape(B, self.num_heads, self.head_dim, H2 * W2).permute(0, 1, 3, 2)  # [B, num_heads, H2*W2, head_dim]
        k2 = k2.reshape(B, self.num_heads, self.head_dim, H2 * W2).permute(0, 1, 2, 3)  # [B, num_heads, head_dim, H2*W2]
        v2 = v2.reshape(B, self.num_heads, self.head_dim, H2 * W2).permute(0, 1, 3, 2)  # [B, num_heads, H2*W2, head_dim]
        
        # 计算互注意力矩阵
        # 模态1关注模态2
        attn_1to2 = torch.matmul(q1, k2) * self.scale  # [B, num_heads, H1*W1, H2*W2]
        attn_1to2 = F.softmax(attn_1to2, dim=-1)       # 在模态2的空间维度上进行softmax
        attn_1to2 = self.dropout(attn_1to2)            # 应用dropout
        
        # 模态2关注模态1
        attn_2to1 = torch.matmul(q2, k1) * self.scale  # [B, num_heads, H2*W2, H1*W1]
        attn_2to1 = F.softmax(attn_2to1, dim=-1)       # 在模态1的空间维度上进行softmax
        attn_2to1 = self.dropout(attn_2to1)            # 应用dropout
        
        # 应用注意力权重
        # 模态1关注模态2后的输出
        out1 = torch.matmul(attn_1to2, v2)  # [B, num_heads, H1*W1, head_dim]
        out1 = out1.permute(0, 1, 3, 2).reshape(B, self.hidden_dim, H1, W1)  # [B, hidden_dim, H1, W1]
        
        # 模态2关注模态1后的输出
        out2 = torch.matmul(attn_2to1, v1)  # [B, num_heads, H2*W2, head_dim]
        out2 = out2.permute(0, 1, 3, 2).reshape(B, self.hidden_dim, H2, W2)  # [B, hidden_dim, H2, W2]
        
        # 输出投影
        output1 = self.modal1_output_proj(out1)  # [B, output_dim, H1, W1]
        output2 = self.modal2_output_proj(out2)  # [B, output_dim, H2, W2]
        
        return output1, output2


class ScaleFusionBlock(nn.Module):
    """特征融合块：整合自注意力和跨尺度注意力的计算"""
    def __init__(self, in_channels_h, in_channels_l, hidden_dim=256, num_heads=8, dropout=0.1):
        super(ScaleFusionBlock, self).__init__()
        
        # 跨尺度注意力模块
        self.cross_attn = EfficientCrossScaleAttention(
            in_channels_h=in_channels_h,
            in_channels_l=in_channels_l,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 规范化层
        self.norm_h1 = nn.LayerNorm([in_channels_h])
        self.norm_l1 = nn.LayerNorm([in_channels_l])
    
    def _apply_norm(self, x, norm_layer):
        """应用LayerNorm到特征图"""
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x = norm_layer(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        return x
    
    def forward(self, high_feat, low_feat):
        """
        参数:
            high_feat: 高分辨率特征，形状为 [B, C_h, H_h, W_h]
            low_feat: 低分辨率特征，形状为 [B, C_l, H_l, W_l]
        返回:
            high_out: 融合后的高分辨率特征
            low_out: 融合后的低分辨率特征
        """
        
        # 跨尺度注意力 + 残差连接 + 规范化
        high_cross, low_cross = self.cross_attn(high_feat, low_feat)
        high_feat = high_feat + high_cross
        high_feat = self._apply_norm(high_feat, self.norm_h1)
        
        low_feat = low_feat + low_cross
        low_feat = self._apply_norm(low_feat, self.norm_l1)
        
        return high_feat, low_feat
    

class DyadicScaleFusionBlock(nn.Module):
    """特征融合块：整合自注意力和跨尺度注意力的计算"""
    def __init__(self, in_channels_h, in_channels_l, hidden_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        
        # 跨尺度注意力模块
        self.cross_attn = DyadicCrossAttention(
            dim1=in_channels_h,
            dim2=in_channels_l,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 规范化层
        self.norm_h1 = nn.LayerNorm([in_channels_h])
        self.norm_l1 = nn.LayerNorm([in_channels_l])
    
    def _apply_norm(self, x, norm_layer):
        """应用LayerNorm到特征图"""
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x = norm_layer(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        return x
    
    def forward(self, high_feat, low_feat):
        """
        参数:
            high_feat: 高分辨率特征，形状为 [B, C_h, H_h, W_h]
            low_feat: 低分辨率特征，形状为 [B, C_l, H_l, W_l]
        返回:
            high_out: 融合后的高分辨率特征
            low_out: 融合后的低分辨率特征
        """
        
        # 跨尺度注意力 + 残差连接 + 规范化
        high_cross, low_cross = self.cross_attn(high_feat, low_feat)
        high_feat = high_feat + high_cross
        high_feat = self._apply_norm(high_feat, self.norm_h1)
        
        low_feat = low_feat + low_cross
        low_feat = self._apply_norm(low_feat, self.norm_l1)
        
        return high_feat, low_feat


class ModalFusionBlock(nn.Module):
    """模态间特征融合块：用模态交叉注意力整合不同模态的特征"""
    def __init__(self, in_channels_1, in_channels_2, hidden_dim=256, num_heads=8, dropout=0.1):
        super(ModalFusionBlock, self).__init__()
        
        # 模态间交叉注意力
        self.modal_cross_attn = ModalCrossAttention(
            in_channels_1=in_channels_1,
            in_channels_2=in_channels_2,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 模态融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels_1 + in_channels_2, max(in_channels_1, in_channels_2), kernel_size=1),
            nn.BatchNorm2d(max(in_channels_1, in_channels_2)),
            nn.ReLU(inplace=True)
        )
        
        # 规范化层
        self.norm_1 = nn.LayerNorm([in_channels_1])
        self.norm_2 = nn.LayerNorm([in_channels_2])
    
    def _apply_norm(self, x, norm_layer):
        """应用LayerNorm到特征图"""
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x = norm_layer(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        return x
    
    def forward(self, mod1_feat, mod2_feat):
        """
        参数:
            mod1_feat: 模态1特征，形状为 [B, C_1, H, W]
            mod2_feat: 模态2特征，形状为 [B, C_2, H, W]
        返回:
            fused_feat: 融合后的特征
        """
        # 模态间交叉注意力 + 残差连接 + 规范化
        mod1_cross, mod2_cross = self.modal_cross_attn(mod1_feat, mod2_feat)
        
        mod1_enhanced = mod1_feat + mod1_cross
        mod1_enhanced = self._apply_norm(mod1_enhanced, self.norm_1)
        
        mod2_enhanced = mod2_feat + mod2_cross
        mod2_enhanced = self._apply_norm(mod2_enhanced, self.norm_2)
        
        # 特征拼接和融合
        fused_feat = torch.cat([mod1_enhanced, mod2_enhanced], dim=1)
        fused_feat = self.fusion(fused_feat)
        
        return fused_feat
    

class DyadicModalFusionBlock(nn.Module):
    """模态间特征融合块：用模态交叉注意力整合不同模态的特征"""
    def __init__(self, in_channels_1, in_channels_2, hidden_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        
        # 模态间交叉注意力
        self.dyadic_cross_attn = DyadicCrossAttention(
            dim1=in_channels_1,
            dim2=in_channels_2,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 模态融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels_1 + in_channels_2, max(in_channels_1, in_channels_2), kernel_size=1),
            nn.BatchNorm2d(max(in_channels_1, in_channels_2)),
            nn.ReLU(inplace=True)
        )
        
        # 规范化层
        self.norm_1 = nn.LayerNorm([in_channels_1])
        self.norm_2 = nn.LayerNorm([in_channels_2])
    
    def _apply_norm(self, x, norm_layer):
        """应用LayerNorm到特征图"""
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x = norm_layer(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        return x
    
    def forward(self, mod1_feat, mod2_feat):
        """
        参数:
            mod1_feat: 模态1特征，形状为 [B, C_1, H, W]
            mod2_feat: 模态2特征，形状为 [B, C_2, H, W]
        返回:
            fused_feat: 融合后的特征
        """
        # 模态间交叉注意力 + 残差连接 + 规范化
        mod1_cross, mod2_cross = self.dyadic_cross_attn(mod1_feat, mod2_feat)
        
        mod1_enhanced = mod1_feat + mod1_cross
        mod1_enhanced = self._apply_norm(mod1_enhanced, self.norm_1)
        
        mod2_enhanced = mod2_feat + mod2_cross
        mod2_enhanced = self._apply_norm(mod2_enhanced, self.norm_2)
        
        # 特征拼接和融合
        fused_feat = torch.cat([mod1_enhanced, mod2_enhanced], dim=1)
        fused_feat = self.fusion(fused_feat)
        
        return fused_feat


class PyramidAttentionFusion(nn.Module):
    """金字塔注意力融合模块：处理多尺度特征的信息交互"""
    def __init__(self, feature_dims, hidden_dim=256, num_heads=8, dropout=0.1, num_layers=2):
        """
        参数:
            feature_dims: 列表，包含各层特征图的通道数（从高分辨率到低分辨率）
            hidden_dim: 注意力机制的隐藏维度
            num_heads: 多头注意力的头数
            dropout: Dropout概率
            num_layers: 每个融合阶段的层数
        """
        super(PyramidAttentionFusion, self).__init__()
        
        self.num_levels = len(feature_dims)
        
        # 自顶向下和自底向上的路径
        self.fusion_blocks_bottom_up = nn.ModuleList()
        
        # 创建融合块：自底向上路径
        for i in range(self.num_levels - 1, 0, -1):
            # 当前层和上一层的特征维度
            curr_dim = feature_dims[i]
            prev_dim = feature_dims[i - 1]
            
            # 对每个相邻层对，创建多层融合块
            self.fusion_blocks_bottom_up.append(
                ScaleFusionBlock(
                        in_channels_h=prev_dim,
                        in_channels_l=curr_dim,
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=dropout
                )
            )
    
    def forward(self, features):
        """
        参数:
            features: 特征图列表，从高分辨率到低分辨率排列
                     每个元素形状为 [B, C_i, H_i, W_i]
        返回:
            enhanced_features: 增强后的特征图列表，排列顺序与输入相同
        """
        # 创建特征图副本
        enhanced_features = list(features)
        
        # === 自底向上路径：低分辨率到高分辨率 ===
        for i in range(self.num_levels - 1):
            # 反向索引
            idx = self.num_levels - 2 - i
            
            high_feat = enhanced_features[idx]
            low_feat = enhanced_features[idx + 1]
            
            high_feat, low_feat = self.fusion_blocks_bottom_up[i](high_feat, low_feat)
            
            # 更新特征
            enhanced_features[idx] = high_feat
            enhanced_features[idx + 1] = low_feat
        
        return enhanced_features


class MultiModalPyramidAttentionFusion(nn.Module):
    """多模态金字塔注意力融合模块：先融合模态，再进行尺度间交互"""
    def __init__(self, feature_dims_mod1, feature_dims_mod2, hidden_dim=256, num_heads=8, dropout=0.1, fusion_type=0):
        """
        参数:
            feature_dims_mod1: 列表，包含模态1各层特征图的通道数（从高分辨率到低分辨率）
            feature_dims_mod2: 列表，包含模态2各层特征图的通道数（从高分辨率到低分辨率）
            hidden_dim: 注意力机制的隐藏维度
            num_heads: 多头注意力的头数
            dropout: Dropout概率
        """
        super(MultiModalPyramidAttentionFusion, self).__init__()
        
        assert len(feature_dims_mod1) == len(feature_dims_mod2), "两个模态的特征层数必须相同"
        self.num_levels = len(feature_dims_mod1)
        
        # 模态融合块：在每个尺度上融合两个模态
        if fusion_type == 0:
            self.modal_fusion_blocks = nn.ModuleList([
                ModalFusionBlock(
                    in_channels_1=feature_dims_mod1[i], 
                    in_channels_2=feature_dims_mod2[i],
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout
                ) for i in range(self.num_levels)
            ])
        elif fusion_type == 1:
            self.modal_fusion_blocks = nn.ModuleList([
                DyadicModalFusionBlock(
                    in_channels_1=feature_dims_mod1[i], 
                    in_channels_2=feature_dims_mod2[i],
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout
                ) for i in range(self.num_levels)
            ])
        elif fusion_type == 2:
            self.modal_fusion_blocks = nn.ModuleList([
                DyadicModalFusionBlock(
                    in_channels_1=feature_dims_mod1[i], 
                    in_channels_2=feature_dims_mod2[i],
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout
                ) for i in range(self.num_levels)
            ])
        
        # 计算融合后的特征维度（取两个模态中的较大值）
        fused_dims = [max(feature_dims_mod1[i], feature_dims_mod2[i]) for i in range(self.num_levels)]
        
        # 自底向上的路径：尺度间融合
        self.fusion_blocks_bottom_up = nn.ModuleList()
        
        # 创建尺度融合块：自底向上路径
        for i in range(self.num_levels - 1, 0, -1):
            # 当前层和上一层的特征维度
            curr_dim = fused_dims[i]
            prev_dim = fused_dims[i - 1]
            
            # 创建融合块
            if fusion_type == 0:
                self.fusion_blocks_bottom_up.append(
                    ScaleFusionBlock(
                        in_channels_h=prev_dim,
                        in_channels_l=curr_dim,
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=dropout
                    )
                )
            elif fusion_type == 1:
                self.fusion_blocks_bottom_up.append(
                    ScaleFusionBlock(
                        in_channels_h=prev_dim,
                        in_channels_l=curr_dim,
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=dropout
                    )
                )
            elif fusion_type == 2:
                self.fusion_blocks_bottom_up.append(
                    DyadicScaleFusionBlock(
                        in_channels_h=prev_dim,
                        in_channels_l=curr_dim,
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=dropout
                    )
                )
    
    def forward(self, features_mod1, features_mod2):
        """
        参数:
            features_mod1: 模态1特征图列表，从高分辨率到低分辨率排列
                          每个元素形状为 [B, C_i, H_i, W_i]
            features_mod2: 模态2特征图列表，从高分辨率到低分辨率排列
                          每个元素形状为 [B, C_i, H_i, W_i]
        返回:
            enhanced_features: 增强后的特征图列表，排列顺序与输入相同
        """
        assert len(features_mod1) == len(features_mod2) == self.num_levels, "输入特征图的数量与初始化时不匹配"
        
        # 第一阶段：模态间融合
        fused_features = []
        for i in range(self.num_levels):
            # 确保两个模态的特征在同一空间尺寸上
            if features_mod1[i].shape[2:] != features_mod2[i].shape[2:]:
                target_size = features_mod1[i].shape[2:]
                features_mod2[i] = F.interpolate(
                    features_mod2[i], 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # 应用模态融合
            fused_feat = self.modal_fusion_blocks[i](features_mod1[i], features_mod2[i])
            fused_features.append(fused_feat)
        
        # 第二阶段：尺度间融合（自底向上）
        enhanced_features = list(fused_features)  # 创建副本
        
        for i in range(self.num_levels - 1):
            # 反向索引，从低分辨率到高分辨率
            idx = self.num_levels - 2 - i
            
            high_feat = enhanced_features[idx]
            low_feat = enhanced_features[idx + 1]
            
            # 应用尺度融合
            high_feat, low_feat = self.fusion_blocks_bottom_up[i](high_feat, low_feat)
            
            # 更新特征
            enhanced_features[idx] = high_feat
            enhanced_features[idx + 1] = low_feat
        
        return enhanced_features
    

class PyramidAttentionNet(nn.Module):
    """金字塔注意力网络：演示金字塔注意力融合在CNN特征金字塔上的应用"""
    def __init__(self, input_channels=3, num_classes=1000):
        super(PyramidAttentionNet, self).__init__()
        
        # 定义骨干网络（简化版ResNet）
        self.backbone = nn.ModuleList([
            # 第一阶段: [B, 3, 224, 224] -> [B, 64, 56, 56]
            nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=7, stride=4, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            # 第二阶段: [B, 64, 56, 56] -> [B, 128, 28, 28]
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            # 第三阶段: [B, 128, 28, 28] -> [B, 256, 14, 14]
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            # 第四阶段: [B, 256, 14, 14] -> [B, 512, 7, 7]
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
        ])
        
        # 金字塔注意力融合
        self.pyramid_fusion = PyramidAttentionFusion(
            feature_dims=[64, 128, 256, 512],
            hidden_dim=256,
            num_heads=8,
            dropout=0.1,
            num_layers=2
        )
        
        # 定义分类头（用于演示）
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """
        参数:
            x: 输入图像，形状为 [B, 3, H, W]
        返回:
            out: 类别预测
            features: 增强后的特征图列表
        """
        # 提取多级特征
        features = []
        for stage in self.backbone:
            x = stage(x)
            features.append(x)
        
        # 应用金字塔注意力融合
        enhanced_features = self.pyramid_fusion(features)
        
        # 分类（使用最低分辨率特征）
        out = self.classifier(enhanced_features[-1])
        
        return out, enhanced_features


# 示例用法
def example_usage():
    # 创建不同尺度的特征图
    batch_size = 2
    
    # 假设有四个不同尺度的特征图
    feat1 = torch.randn(batch_size, 64, 56, 56)    # P2
    feat2 = torch.randn(batch_size, 128, 28, 28)   # P3
    feat3 = torch.randn(batch_size, 256, 14, 14)   # P4
    feat4 = torch.randn(batch_size, 512, 7, 7)     # P5
    
    features = [feat1, feat2, feat3, feat4]
    
    print("输入特征形状:")
    for i, feat in enumerate(features):
        print(f"P{i+2}: {feat.shape}")
    
    # 初始化金字塔注意力融合模块
    paf = PyramidAttentionFusion(
        feature_dims=[64, 128, 256, 512],
        hidden_dim=256,
        num_heads=8,
        dropout=0.1,
        num_layers=2
    )
    
    # 应用金字塔注意力融合
    enhanced_features = paf(features)
    
    print("\n增强后的特征形状:")
    for i, feat in enumerate(enhanced_features):
        print(f"P{i+2}: {feat.shape}")
    
    # 完整网络示例
    model = PyramidAttentionNet(input_channels=3, num_classes=1000)
    
    # 模拟输入
    input_image = torch.randn(batch_size, 3, 224, 224)
    
    # 前向传播
    output, enhanced_feats = model(input_image)
    
    print("\n网络输出形状:")
    print(f"分类输出: {output.shape}")
    for i, feat in enumerate(enhanced_feats):
        print(f"P{i+2}: {feat.shape}")

if __name__ == "__main__":
    example_usage()