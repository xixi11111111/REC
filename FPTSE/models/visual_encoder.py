# coding=utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.network_blocks import add_conv, DropBlock, FeatureAdaption, resblock, SPPLayer, upsample


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim):
        """
        改进的位置编码模块
        :param hidden_dim: 隐藏维度（特征维度）
        """
        super(PositionalEncoding, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x):
        """
        将位置编码添加到输入特征上
        :param x: 输入特征，形状为 (batch_size, hidden_dim, height, width)
        :return: 添加位置编码后的特征
        """
        batch_size, c, h, w = x.shape

        # 创建网格坐标
        y_pos = torch.arange(h).float().to(x.device)
        x_pos = torch.arange(w).float().to(x.device)

        # 归一化位置到 [-1, 1]
        y_pos = 2 * (y_pos / (h - 1)) - 1
        x_pos = 2 * (x_pos / (w - 1)) - 1

        # 创建位置网格
        grid_y, grid_x = torch.meshgrid(y_pos, x_pos, indexing='ij')

        # 计算位置编码
        pos_enc = torch.zeros(batch_size, self.hidden_dim, h, w).to(x.device)

        # 使用sin和cos函数生成位置编码
        div_term = torch.exp(torch.arange(0, self.hidden_dim, 2).float() * (-np.log(10000.0) / self.hidden_dim)).to(
            x.device)

        pos_enc[:, 0::4, :, :] = torch.sin(grid_x.unsqueeze(0).unsqueeze(0) * div_term[None, :, None, None])[:,
                                 :pos_enc.shape[1] // 4, :, :]
        pos_enc[:, 1::4, :, :] = torch.cos(grid_x.unsqueeze(0).unsqueeze(0) * div_term[None, :, None, None])[:,
                                 :pos_enc.shape[1] // 4, :, :]
        pos_enc[:, 2::4, :, :] = torch.sin(grid_y.unsqueeze(0).unsqueeze(0) * div_term[None, :, None, None])[:,
                                 :pos_enc.shape[1] // 4, :, :]
        pos_enc[:, 3::4, :, :] = torch.cos(grid_y.unsqueeze(0).unsqueeze(0) * div_term[None, :, None, None])[:,
                                 :pos_enc.shape[1] // 4, :, :]

        return x + pos_enc


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear transformations
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        # Split into heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final linear layer
        output = self.W_o(context)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)

        # Feed Forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        # Multi-head self attention
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, src2, mask)
        src = src + self.dropout1(src2)

        # Feed forward network
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)

        return src


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        return self.norm(src)


class FeaturePyramidTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=256):
        super().__init__()
        self.in_channels = in_channels  # [512, 256, 128]
        self.out_channels = out_channels  # [1024, 512, 256]

        # 位置编码模块
        self.position_encoding = PositionalEncoding(hidden_dim)

        # Transformer 编码器
        self.transformer_encoder = nn.ModuleList([
            TransformerEncoder(
                d_model=hidden_dim,
                nhead=8,
                num_encoder_layers=3,
                dim_feedforward=1024,
                dropout=0.1
            ) for _ in range(3)
        ])

        # Level embeddings
        self.level_embed = nn.Parameter(torch.randn(3, 1, hidden_dim))

        # Input projections
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, hidden_dim, kernel_size=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ) for in_ch in in_channels
        ])

        # Output projections
        self.output_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, out_ch, kernel_size=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ) for out_ch in out_channels
        ])

        # Channel adjustment for residual connections
        self.channel_adjust = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ) for in_ch, out_ch in zip(in_channels, out_channels)
        ])

    def forward(self, features):
        transformed_features = []
        for i, feat in enumerate(features):
            # 调整输入特征的通道数以匹配输出
            adjusted_feat = self.channel_adjust[i](feat)

            # 投影到hidden_dim
            proj_feat = self.input_proj[i](feat)

            # 添加位置编码
            proj_feat = self.position_encoding(proj_feat)

            # Transformer处理
            b, c, h, w = proj_feat.shape
            # 保持空间位置信息的展平方式
            feat_flat = proj_feat.view(b, c, -1).permute(2, 0, 1)

            # 添加level embedding
            level_embed = self.level_embed[i].expand(feat_flat.size(0), -1, -1)
            feat_flat = feat_flat + level_embed

            # Transformer处理
            transformed_feat = self.transformer_encoder[i](feat_flat)

            # 恢复空间维度
            transformed_feat = transformed_feat.permute(1, 2, 0).view(b, c, h, w)

            # 投影到YOLO head需要的通道数
            transformed_feat = self.output_proj[i](transformed_feat)

            # 残差连接
            transformed_feat = transformed_feat + adjusted_feat
            transformed_features.append(transformed_feat)

        return transformed_features


class YOLOv3Head(nn.Module):
    def __init__(self, anch_mask, n_classes, stride, in_ch=1024, ignore_thre=0.7, label_smooth=False, rfb=False,
                 sep=False):
        super(YOLOv3Head, self).__init__()
        # print(f"YOLOv3Head init - Expected input channels: {in_ch}")
        self.anchors = [
            (10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (42, 119),
            (116, 90), (156, 198), (121, 240)
        ]
        if sep:
            self.anchors = [
                (10, 13), (16, 30), (33, 23),
                (30, 61), (62, 45), (42, 119),
                (116, 90), (156, 198), (373, 326)
            ]

        self.anch_mask = anch_mask
        self.n_anchors = 4
        self.n_classes = n_classes

        # 使用正确的输入通道数初始化guide_wh
        self.guide_wh = nn.Conv2d(
            in_channels=in_ch,  # 使用传入的in_ch
            out_channels=2 * self.n_anchors,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.Feature_adaption = FeatureAdaption(in_ch, in_ch, self.n_anchors, rfb, sep)
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=self.n_anchors * (self.n_classes + 5),
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.stride = stride
        self._label_smooth = label_smooth

        self.all_anchors_grid = self.anchors
        self.masked_anchors = [self.all_anchors_grid[i] for i in self.anch_mask]

    def forward(self, xin, labels=None):
        # print(f"YOLOv3Head forward - Input shape: {xin.shape}")
        # print(f"guide_wh conv expects {self.guide_wh.in_channels} channels, got {xin.shape[1]}")
        wh_pred = self.guide_wh(xin)  # Anchor guiding

        if xin.type() == 'torch.cuda.HalfTensor':  # As DCN only support FP32 now, change the feature to float.
            wh_pred = wh_pred.float()
            if labels is not None:
                labels = labels.float()
            self.Feature_adaption = self.Feature_adaption.float()
            self.conv = self.conv.float()
            xin = xin.float()

        feature_adapted = self.Feature_adaption(xin, wh_pred)

        output = self.conv(feature_adapted)
        wh_pred = torch.exp(wh_pred)

        batchsize = output.shape[0]
        fsize = output.shape[2]
        image_size = fsize * self.stride
        n_ch = 5 + self.n_classes
        dtype = torch.cuda.FloatTensor if xin.is_cuda else torch.FloatTensor

        wh_pred = wh_pred.view(batchsize, self.n_anchors, 2, fsize, fsize)
        wh_pred = wh_pred.permute(0, 1, 3, 4, 2).contiguous()

        output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
        output = output.permute(0, 1, 3, 4, 2).contiguous()

        x_shift = dtype(np.broadcast_to(
            np.arange(fsize, dtype=np.float32), output.shape[:4])).to(xin.device)
        y_shift = dtype(np.broadcast_to(
            np.arange(fsize, dtype=np.float32).reshape(fsize, 1), output.shape[:4])).to(xin.device)

        masked_anchors = np.array(self.masked_anchors)

        w_anchors = dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 0], (1, self.n_anchors - 1, 1, 1)), [batchsize, self.n_anchors - 1, fsize, fsize])).to(
            xin.device)
        h_anchors = dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 1], (1, self.n_anchors - 1, 1, 1)), [batchsize, self.n_anchors - 1, fsize, fsize])).to(
            xin.device)

        default_center = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).type(dtype).to(xin.device)

        pred_anchors = torch.cat((default_center, wh_pred), dim=-1).contiguous()

        anchors_based = pred_anchors[:, :self.n_anchors - 1, :, :, :]
        anchors_free = pred_anchors[:, self.n_anchors - 1, :, :, :]
        anchors_based[..., 2] *= w_anchors
        anchors_based[..., 3] *= h_anchors
        anchors_free[..., 2] *= self.stride * 4
        anchors_free[..., 3] *= self.stride * 4
        pred_anchors[..., :2] = pred_anchors[..., :2].detach()
        pred = output.clone()
        pred[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(
            pred[..., np.r_[:2, 4:n_ch]])
        pred[..., 0] += x_shift
        pred[..., 1] += y_shift
        pred[..., :2] *= self.stride
        pred[..., 2] = torch.exp(pred[..., 2]) * (pred_anchors[..., 2])
        pred[..., 3] = torch.exp(pred[..., 3]) * (pred_anchors[..., 3])
        pred_new = pred.view(batchsize, -1, pred.size()[2] * pred.size()[3], n_ch).permute(0, 2, 1, 3)
        refined_pred = pred.view(batchsize, -1, n_ch)
        return refined_pred.data, pred_new.data


class YOLOv3(nn.Module):
    def __init__(self, __C):
        super(YOLOv3, self).__init__()
        self.module_list = create_yolov3_modules(__C.CLASS_NUM, ignore_thre=0.7, label_smooth=False, rfb=False)

        self.fpt = FeaturePyramidTransformer(
            in_channels=[512, 256, 128],  # [layer18, layer27, layer36]的输入通道数
            out_channels=[1024, 512, 256],  # 对应YOLO head需要的通道数
            hidden_dim=256
        )

    def forward(self, x, targets=None, epoch=0):
        output = []
        feature_output = []
        boxes_output = []
        route_layers = []
        fpt_features = []

        for i, module in enumerate(self.module_list):
            # 在YOLO head之前收集特征
            if i in [18, 27, 36]:  # 在YOLO head之前的卷积层
                # print(f"\nBefore YOLO head {i}:")
                # print(f"Feature shape: {x.shape}")
                fpt_features.append(x)

            if i in [19, 28, 37]:  # YOLO head layers
                # print(f"\nAt YOLO head {i}:")
                # print(f"Input shape: {x.shape}")

                if len(fpt_features) == 3:
                    #                     print("Processing FPT features:")
                    #                     print([f.shape for f in fpt_features])

                    transformed_features = self.fpt(fpt_features)
                    # 修改这里：根据当前YOLO head的索引选择对应的特征
                    curr_feature_idx = {19: 0, 28: 1, 37: 2}[i]
                    x = transformed_features[curr_feature_idx]
                    # print(f"Using transformed feature {curr_feature_idx}, shape:", x.shape)

                feature_output.append(x)
                x, box_output = module(x)
                boxes_output.append(box_output)
                output.append(x)
            else:
                x = module(x)

            # 维护路由连接
            if i in [6, 8, 17, 26]:
                route_layers.append(x)
            if i == 19:
                x = route_layers[2]
            if i == 28:
                x = route_layers[3]
            if i == 21:
                x = torch.cat((x, route_layers[1]), 1)
            if i == 30:
                x = torch.cat((x, route_layers[0]), 1)

        return torch.cat(output, 1), feature_output, boxes_output


def create_yolov3_modules(num_classes, ignore_thre, label_smooth, rfb):
    """
    Build yolov3 layer modules.
    """
    # DarkNet53
    mlist = nn.ModuleList()
    mlist.append(add_conv(in_ch=3, out_ch=32, ksize=3, stride=1))  # 0
    mlist.append(add_conv(in_ch=32, out_ch=64, ksize=3, stride=2))  # 1
    mlist.append(resblock(ch=64))  # 2
    mlist.append(add_conv(in_ch=64, out_ch=128, ksize=3, stride=2))  # 3
    mlist.append(resblock(ch=128, nblocks=2))  # 4
    mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=2))  # 5
    mlist.append(resblock(ch=256, nblocks=8))  # shortcut 1 from here     #6
    mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=2))  # 7
    mlist.append(resblock(ch=512, nblocks=8))  # shortcut 2 from here     #8
    mlist.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=2))  # 9
    mlist.append(resblock(ch=1024, nblocks=4))  # 10

    # YOLOv3
    mlist.append(resblock(ch=1024, nblocks=1, shortcut=False))  # 11
    mlist.append(add_conv(in_ch=1024, out_ch=512, ksize=1, stride=1))  # 12
    # SPP Layer
    mlist.append(SPPLayer())  # 13

    mlist.append(add_conv(in_ch=2048, out_ch=512, ksize=1, stride=1))  # 14
    mlist.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=1))  # 15
    mlist.append(DropBlock(block_size=1, keep_prob=1.0))  # 16
    mlist.append(add_conv(in_ch=1024, out_ch=512, ksize=1, stride=1))  # 17
    # 1st yolo branch
    mlist.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=1))  # 18
    mlist.append(
        YOLOv3Head(anch_mask=[6, 7, 8], n_classes=num_classes, stride=32, in_ch=1024,
                   ignore_thre=ignore_thre, label_smooth=label_smooth, rfb=rfb))  # 19

    mlist.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))  # 20
    mlist.append(upsample(scale_factor=2, mode='nearest'))  # 21
    mlist.append(add_conv(in_ch=768, out_ch=256, ksize=1, stride=1))  # 22
    mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))  # 23
    mlist.append(DropBlock(block_size=1, keep_prob=1.0))  # 24
    mlist.append(resblock(ch=512, nblocks=1, shortcut=False))  # 25
    mlist.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))  # 26
    # 2nd yolo branch
    mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))  # 27
    mlist.append(
        YOLOv3Head(anch_mask=[3, 4, 5], n_classes=num_classes, stride=16, in_ch=512,
                   ignore_thre=ignore_thre, label_smooth=label_smooth, rfb=rfb))  # 28

    mlist.append(add_conv(in_ch=256, out_ch=128, ksize=1, stride=1))  # 29
    mlist.append(upsample(scale_factor=2, mode='nearest'))  # 30
    mlist.append(add_conv(in_ch=384, out_ch=128, ksize=1, stride=1))  # 31
    mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=1))  # 32
    mlist.append(DropBlock(block_size=1, keep_prob=1.0))  # 33
    mlist.append(resblock(ch=256, nblocks=1, shortcut=False))  # 34
    mlist.append(add_conv(in_ch=256, out_ch=128, ksize=1, stride=1))  # 35
    mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=1))  # 36
    mlist.append(
        YOLOv3Head(anch_mask=[0, 1, 2], n_classes=num_classes, stride=8, in_ch=256,
                   ignore_thre=ignore_thre, label_smooth=label_smooth, rfb=rfb))  # 37

    return mlist


backbone_dict = {
    'yolov3': YOLOv3,
}


def visual_encoder(__C):
    vis_enc = backbone_dict[__C.VIS_ENC](__C)
    return vis_enc

