# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.language_encoder import language_encoder
from models.visual_encoder import visual_encoder
from models.RefCLIP.head import WeakREChead
from models.network_blocks import MultiScaleFusion
from models.co_attention import CoAttention, CrossModalEncoder, getContrast


class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size):
        super(Net, self).__init__()
        self.select_num = __C.SELECT_NUM
        self.visual_encoder = visual_encoder(__C).eval()
        self.lang_encoder = language_encoder(__C, pretrained_emb, token_size)

        self.linear_vs = nn.Linear(1024, __C.HIDDEN_SIZE)
        self.linear_ts = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        
        # 修改通道调整层，保持特征从小到大的顺序
        self.channel_adjusters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1),  # 1024 -> 256
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Identity(),  # 保持512不变
            nn.Sequential(
                nn.Conv2d(256, 1024, kernel_size=1),  # 256 -> 1024
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True)
            )
        ])
        
        self.cross_modal_encoder = CrossModalEncoder(__C)
        self.head = WeakREChead(__C)
        # MultiScaleFusion期望的输入通道数顺序是从小到大
        self.multi_scale_manner = MultiScaleFusion(v_planes=(256, 512, 1024), hiden_planes=1024, scaled=True)
        self.class_num = __C.CLASS_NUM
        
        if __C.VIS_FREEZE:
            self._frozen(self.visual_encoder)

    def _frozen(self, module):
        if getattr(module, 'module', False):
            for child in module.module():
                for param in child.parameters():
                    param.requires_grad = False
        else:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, x, y):
        # Vision and Language Encoding
        with torch.no_grad():
            boxes_all, x_, boxes_sml = self.visual_encoder(x)
        y_ = self.lang_encoder(y)

        # Vision Multi Scale Fusion
        s, m, l = x_  # s:1024, m:512, l:256
        print("原始特征通道数:")
        print(f"small scale: {s.shape[1]}")  # 1024
        print(f"medium scale: {m.shape[1]}")  # 512
        print(f"large scale: {l.shape[1]}")  # 256
        
        # 调整通道数
        s_new = self.channel_adjusters[0](s)  # 1024 -> 256
        m_new = self.channel_adjusters[1](m)  # 保持512不变
        l_new = self.channel_adjusters[2](l)  # 256 -> 1024
        
        print("\n调整后的特征通道数:")
        print(f"small scale (adjusted): {s_new.shape[1]}")  # 256
        print(f"medium scale (adjusted): {m_new.shape[1]}")  # 512
        print(f"large scale (adjusted): {l_new.shape[1]}")  # 1024
        
        # 按照从小到大的顺序输入到MultiScaleFusion
        x_input = [s_new, m_new, l_new]  # [256, 512, 1024]
        l_fused, m_fused, s_fused = self.multi_scale_manner(x_input)
        
        print("\n多尺度融合后的特征通道数:")
        print(f"small scale (fused): {s_fused.shape[1]}")
        print(f"medium scale (fused): {m_fused.shape[1]}")
        print(f"large scale (fused): {l_fused.shape[1]}")
        
        x_ = [s_fused, m_fused, l_fused]

        # Cross Modal Interaction
        vis_feat, lang_feat = self.cross_modal_encoder(x_, y_['lang_feat'])
        
        print("\n跨模态特征维度:")
        print(f"Visual feature: {vis_feat.shape}")
        print(f"Language feature: {lang_feat.shape}")

        if self.training:
            loss = getContrast(vis_feat, lang_feat)
            return loss
        else:
            # Anchor Selection for test mode
            boxes_sml_new = []
            mean_i = torch.mean(boxes_sml[0], dim=2, keepdim=True)
            mean_i = mean_i.squeeze(2)[:, :, 4]
            vals, indices = mean_i.topk(k=int(self.select_num), dim=1, largest=True, sorted=True)
            bs, gridnum, anncornum, ch = boxes_sml[0].shape
            bs_, selnum = indices.shape
            box_sml_new = boxes_sml[0].masked_select(
                torch.zeros(bs, gridnum).to(boxes_sml[0].device).scatter(1, indices, 1).bool().unsqueeze(2).unsqueeze(
                    3).expand(bs, gridnum, anncornum, ch)).contiguous().view(bs, selnum, anncornum, ch)
            boxes_sml_new.append(box_sml_new)

            similarity = F.cosine_similarity(vis_feat.unsqueeze(1), lang_feat.unsqueeze(0), dim=-1)
            predictions = torch.sigmoid(similarity * 10)
            predictions = predictions.unsqueeze(-1)
            predictions_list = [predictions]

            box_pred = get_boxes(boxes_sml_new, predictions_list, self.class_num)
            return box_pred
def get_boxes(boxes_sml, predictionslist, class_num):
    batchsize = predictionslist[0].size()[0]
    pred = []
    for i in range(len(predictionslist)):
        # 获取预测分数
        pred_scores = predictionslist[i].squeeze(-1)  # [B, B]
        
        # 对每个样本选择最高分数的预测框
        max_scores, max_indices = pred_scores.max(dim=1)  # [B]
        
        # 组织预测框
        boxes = boxes_sml[i].contiguous()  # 确保内存连续
        bs, n_anchors, n_dims = boxes.shape[0], boxes.shape[1], boxes.shape[3]
        boxes = boxes.reshape(bs, -1, n_dims)  # [B, H*W, class_num+5]
        
        # 为每个batch选择对应的预测框
        selected_boxes = []
        for b in range(batchsize):
            selected_box = boxes[b]  # [H*W, class_num+5]
            refined_box = selected_box.reshape(-1, class_num+5)
            
            # 处理坐标
            refined_box[:, 0] = refined_box[:, 0] - refined_box[:, 2] / 2
            refined_box[:, 1] = refined_box[:, 1] - refined_box[:, 3] / 2
            refined_box[:, 2] = refined_box[:, 0] + refined_box[:, 2]
            refined_box[:, 3] = refined_box[:, 1] + refined_box[:, 3]
            
            selected_boxes.append(refined_box)
        
        batch_boxes = torch.stack(selected_boxes)  # [B, H*W, class_num+5]
        pred.append(batch_boxes)
    
    boxes = torch.cat(pred, 1)
    score = boxes[:, :, 4]
    max_score, ind = torch.max(score, -1)
    ind_new = ind.unsqueeze(1).unsqueeze(1).repeat(1, 1, 5)
    box_new = torch.gather(boxes, 1, ind_new)
    return box_new