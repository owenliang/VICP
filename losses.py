"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class ArcFace(nn.Module):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta+m)
        """

    def __init__(self, in_features, out_features, s=64.0, m=0.50, easy_margin=False):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).to(cosine)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size()).to(input)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

class CosFace(nn.Module):
    r"""Implement of CosFace (https://arxiv.org/pdf/1801.09414.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta)-m
    """

    def __init__(self, in_features, out_features, s=64.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size()).to(input)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = self.s * (cosine - self.m * one_hot)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', s = ' + str(self.s) \
               + ', m = ' + str(self.m) + ')'


class AdaFace(nn.Module):
    def __init__(self,
                 in_features,
                 out_features=70722,
                 m=0.4,
                 h=0.333,
                 s=64.,
                 t_alpha=0.01,
                 ):
        super(AdaFace, self).__init__()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))

        # initial kernel
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m 
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)

        print('\n\AdaFace with the following property')
        print('self.m', self.m)
        print('self.h', self.h)
        print('self.s', self.s)
        print('self.t_alpha', self.t_alpha)

    def forward(self, input, label):

        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # kernel_norm = l2_norm(self.kernel,axis=0)
        # cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps) # for stability
        norms = torch.norm(input, 2, 1, True)

        safe_norms = torch.clip(norms, min=0.001, max=100) # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps) # 66% between -1, 1
        margin_scaler = margin_scaler * self.h # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)

        # g_angular
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                #  base_temperature=0.07
                 ):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        # self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class HardTripletLoss(nn.Module):
    """Hard/Hardest Triplet Loss - 用于物体重识别的度量学习
    
    核心目标:
        让同类样本靠近，异类样本远离
        Loss = max(d(A,P) - d(A,N) + margin, 0)
        目标: d(A,N) >= d(A,P) + margin
    
    关键策略:
        - Hardest Positive: 同类中距离最远的样本（最难匹配的正样本）
        - Hardest Negative: 异类中距离最近的样本（最容易混淆的负样本）
    
    示例:
        labels = [0, 1, 1, 0]  # 4个样本，2个类别
        Anchor=0 (类别0):
            - Hardest Positive: 样本3 (同类，距离最大)
            - Hardest Negative: 样本1或2 (异类，距离最小)
    
    参考: https://omoindrot.github.io/triplet-loss
    """
    def __init__(self, margin=0.1, hardest=False, squared=False):
        """初始化 Hard Triplet Loss
        
        Args:
            margin (float): 正负样本之间的安全间隔
                - 创建"安全区域": d(A,N) 必须 >= d(A,P) + margin
                - VICP 推荐值: 0.3 (平衡性能和泛化能力)
                - 较小 margin (0.1-0.2): 易于训练，可能过拟合
                - 较大 margin (0.5-1.0): 约束更强，收敛困难
            
            hardest (bool): 挖掘策略
                - True: 每个anchor只用1个最难正样本 + 1个最难负样本
                  * 更快 (O(batch_size²))，但不够稳定
                  * 适合大批次
                - False: 使用所有有效的困难三元组 (loss > 0)
                  * 较慢 (O(batch_size³))，但更稳健
                  * VICP 推荐
            
            squared (bool): 距离度量
                - True: 平方欧式距离 ||a-b||²
                - False: 欧式距离 ||a-b|| (推荐)
        """
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        self.hardest = hardest
        self.squared = squared

    def forward(self, embeddings, labels):
        """计算 Hard Triplet Loss
        
        Args:
            embeddings: 特征向量, shape [batch_size, embed_dim]
                示例: [4, 128] - 4个样本，128维特征
            labels: 类别标签, shape [batch_size]
                示例: [0, 1, 1, 0] - 4个样本，2个类别
        
        Returns:
            triplet_loss: 标量损失值
        
        处理流程:
            1. 计算成对距离矩阵 [N, N]
            2. 找最难正样本 (同类中距离最大)
            3. 找最难负样本 (异类中距离最小)
            4. 计算 loss = max(d(A,P) - d(A,N) + margin, 0)
        """
        # 步骤1: 计算所有成对距离
        # 输入: embeddings [batch_size, embed_dim]
        # 输出: pairwise_dist [batch_size, batch_size]
        # pairwise_dist[i][j] = 样本i和样本j之间的距离
        pairwise_dist = _pairwise_distance(embeddings, squared=self.squared)

        if self.hardest:
            # ==================== Hardest 模式: 只使用极端样本 ====================
            
            # 步骤2a: 找最难正样本（同类中距离最远的样本）
            # mask_anchor_positive[i][j] = 1 当且仅当 i≠j 且 label[i]==label[j]
            # 示例 labels=[0,1,1,0]:
            #     [[0,0,0,1],  <- 样本0: 只有样本3是同类
            #      [0,0,1,0],  <- 样本1: 只有样本2是同类
            #      [0,1,0,0],  <- 样本2: 只有样本1是同类
            #      [1,0,0,0]]  <- 样本3: 只有样本0是同类
            mask_anchor_positive = _get_anchor_positive_triplet_mask(labels).float()
            
            # 只保留同类距离，其他置0
            # valid_positive_dist[i][j] = 同类则为距离，否则为0
            valid_positive_dist = pairwise_dist * mask_anchor_positive
            
            # 找每行的最大距离（最难正样本）
            # hardest_positive_dist[i] = 到同类样本的最大距离
            # 形状: [batch_size, 1]
            hardest_positive_dist, _ = torch.max(valid_positive_dist, dim=1, keepdim=True)

            # 步骤2b: 找最难负样本（异类中距离最近的样本）
            # mask_anchor_negative[i][j] = 1 当且仅当 label[i]≠label[j]
            # 示例 labels=[0,1,1,0]:
            #     [[0,1,1,0],  <- 样本0: 样本1,2是异类
            #      [1,0,0,1],  <- 样本1: 样本0,3是异类
            #      [1,0,0,1],  <- 样本2: 样本0,3是异类
            #      [0,1,1,0]]  <- 样本3: 样本1,2是异类
            mask_anchor_negative = _get_anchor_negative_triplet_mask(labels).float()
            
            # 获取每行最大距离（用作惩罚值）
            # 形状: [batch_size, 1]
            max_anchor_negative_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True)
            
            # 核心技巧: 给非负样本加惩罚，防止它们被min选中
            # 对于负样本 (mask=1): 距离保持不变
            # 对于正样本或自己 (mask=0): 距离 += max_dist (变得很大)
            # 示例:
            #   原始: [0.0, 3.2, 2.8, 0.5]
            #   掩码: [0,   1,   1,   0  ]
            #   结果: [∞,   3.2, 2.8, ∞  ]  <- min = 2.8 ✓
            anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
                    1.0 - mask_anchor_negative)
            
            # 找每行的最小距离（最难负样本）
            # 只有负样本有真实距离，其他都很大
            # hardest_negative_dist[i] = 到异类样本的最小距离
            # 形状: [batch_size, 1]
            hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)

            # 步骤3: 计算三元组损失
            # loss = max(d(A,P) - d(A,N) + margin, 0)
            # ReLU移除负值（已满足条件的三元组）
            # 示例 margin=0.3:
            #   d(A,P)=2.0, d(A,N)=3.5 → loss=2.0-3.5+0.3=-1.2 → relu=0 (已满足)
            #   d(A,P)=2.0, d(A,N)=2.1 → loss=2.0-2.1+0.3=0.2 → relu=0.2 (违规)
            triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
            triplet_loss = torch.mean(triplet_loss)
        else:
            # ==================== Hard 模式: 使用所有有效的困难三元组 ====================
            
            # 步骤2: 使用广播枚举所有可能的三元组
            # 将距离矩阵扩展到3D以枚举三元组
            anc_pos_dist = pairwise_dist.unsqueeze(dim=2)  # [batch, batch, 1]
            anc_neg_dist = pairwise_dist.unsqueeze(dim=1)  # [batch, 1, batch]
            
            # 广播机制一次性计算所有 batch³ 个三元组损失！
            # loss[i,j,k] = d(i,j) - d(i,k) + margin
            #             = d(锚点=i, 正样本=j) - d(锚点=i, 负样本=k) + margin
            # 
            # 形状演变 batch_size=4:
            #   anc_pos_dist: [4,4,1] → 广播到 [4,4,4]
            #   anc_neg_dist: [4,1,4] → 广播到 [4,4,4]
            #   loss:         [4,4,4] → 64个可能的三元组
            # 
            # 示例: loss[0,3,1] = d(样本0, 样本3) - d(样本0, 样本1) + margin
            loss = anc_pos_dist - anc_neg_dist + self.margin

            # 步骤3: 过滤有效三元组
            # mask[i,j,k] = 1 当且仅当 triplet(i,j,k) 有效:
            #   - i, j, k 互不相同
            #   - label[i] == label[j] (正样本)
            #   - label[i] != label[k] (负样本)
            mask = _get_triplet_mask(labels).float()
            triplet_loss = loss * mask
            
            # 步骤4: 移除简单三元组 (loss < 0, 已满足条件)
            # 如果 d(A,P) - d(A,N) + margin < 0, 三元组已满足约束
            # 这些对梯度贡献为0，浪费计算资源
            triplet_loss = F.relu(triplet_loss)
            
            # 步骤5: 只对困难三元组求平均
            # 统计困难三元组 (loss > 0)
            hard_triplets = torch.gt(triplet_loss, 1e-16).float()
            num_hard_triplets = torch.sum(hard_triplets)
            
            # 对困难三元组求平均损失
            # 为什么只用困难三元组？因为简单的(loss=0)对训练无帮助
            # 示例: batch_size=32 → 32³=32768 个可能的三元组
            #       但只有约100-500个是困难的(loss>0)
            triplet_loss = torch.sum(triplet_loss) / (num_hard_triplets + 1e-16)

        return triplet_loss


def _pairwise_distance(x, squared=False, eps=1e-16):
    """使用矩阵运算计算成对距离矩阵（比循环快50-100倍）
    
    公式: ||a-b||² = ||a||² - 2<a,b> + ||b||²
    
    Args:
        x: 特征向量, shape [N, D] (N个样本，D维)
           示例: [4, 128] - 4个样本，128维特征
        squared: True返回平方距离；False返回欧式距离
        eps: 避免sqrt(0)梯度问题的小值
    
    Returns:
        distances: 成对距离矩阵, shape [N, N]
                   distances[i][j] = 样本i和样本j之间的距离
    
    形状演变示例 (N=4, D=128):
        x:            [4, 128]
        x.t():        [128, 4]
        cor_mat:      [4, 4]     (内积矩阵)
        norm_mat:     [4]        (每个样本的||x[i]||²)
        norm_mat.unsqueeze(1): [4, 1]  (列向量，用于广播)
        norm_mat.unsqueeze(0): [1, 4]  (行向量，用于广播)
        distances:    [4, 4]     (最终距离矩阵)
    """
    # 步骤1: 计算内积矩阵
    # cor_mat[i][j] = x[i] · x[j] (点积)
    # 形状: [N, D] @ [D, N] → [N, N]
    cor_mat = torch.matmul(x, x.t())
    
    # 步骤2: 提取对角线（平方模）
    # norm_mat[i] = cor_mat[i][i] = x[i] · x[i] = ||x[i]||²
    # 形状: [N]
    norm_mat = cor_mat.diag()
    
    # 步骤3: 使用广播计算平方距离
    # distances[i][j] = ||x[i]||² - 2*x[i]·x[j] + ||x[j]||²
    #                 = ||x[i] - x[j]||²
    # 
    # 广播机制:
    #   norm_mat.unsqueeze(1): [N,1] → 广播到 [N,N] (重复列)
    #   norm_mat.unsqueeze(0): [1,N] → 广播到 [N,N] (重复行)
    # 
    # 示例 N=4:
    #   norm_mat = [a², b², c², d²]
    #   unsqueeze(1) = [[a²], [b²], [c²], [d²]] → [[a²,a²,a²,a²], [b²,b²,b²,b²], ...]
    #   unsqueeze(0) = [[a², b², c², d²]]       → [[a²,b²,c²,d²], [a²,b²,c²,d²], ...]
    distances = norm_mat.unsqueeze(1) - 2 * cor_mat + norm_mat.unsqueeze(0)
    
    # 避免数值误差导致的负值
    distances = F.relu(distances)
    
    if not squared:
        # 转换为欧式距离: ||a-b|| = sqrt(||a-b||²)
        # 
        # 小心处理sqrt(0)以避免梯度问题:
        # - 在sqrt前对distance=0的位置加eps
        # - sqrt后恢复为0（最终结果中不要eps）
        mask = torch.eq(distances, 0.0).float()
        distances = distances + mask * eps  # 0 → eps, others unchanged
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)  # eps → 0, others unchanged
    
    return distances


def _get_anchor_positive_triplet_mask(labels):
    """生成有效(anchor, positive)对的掩码
    
    有效对: i ≠ j 且 label[i] == label[j]
    
    Args:
        labels: 类别标签, shape [N]
                示例: [0, 1, 1, 0]
    
    Returns:
        mask: 二值掩码, shape [N, N]
              mask[i][j] = 1 当i≠j且同类，否则为0
    
    示例 labels = [0, 1, 1, 0]:
        indices_not_equal (i≠j):
            [[0,1,1,1],  ← 对角线为0（排除自己）
             [1,0,1,1],
             [1,1,0,1],
             [1,1,1,0]]
        
        labels_equal (label[i]==label[j]):
            [[1,0,0,1],  ← 样本0,3是类别0
             [0,1,1,0],  ← 样本1,2是类别1
             [0,1,1,0],
             [1,0,0,1]]
        
        mask (两者AND):
            [[0,0,0,1],  ← 样本0: 只有样本3是有效正样本
             [0,0,1,0],  ← 样本1: 只有样本2是有效正样本
             [0,1,0,0],  ← 样本2: 只有样本1是有效正样本
             [1,0,0,0]]  ← 样本3: 只有样本0是有效正样本
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 条件1: i ≠ j (排除自己配对)
    # 单位矩阵: 对角线=1, 其他=0
    # 异或1: 对角线=0, 其他=1
    indices_not_equal = torch.eye(labels.shape[0]).to(device).byte() ^ 1
    
    # 条件2: label[i] == label[j] (同类)
    # 广播: [1,N] == [N,1] → [N,N]
    # labels.unsqueeze(0): [[label0, label1, label2, label3]]
    # labels.unsqueeze(1): [[label0], [label1], [label2], [label3]]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    
    # 合并两个条件（逐元素AND）
    mask = indices_not_equal * labels_equal
    
    return mask


def _get_anchor_negative_triplet_mask(labels):
    """生成有效(anchor, negative)对的掩码
    
    有效对: label[i] ≠ label[j] (异类)
    注意: 不需要检查i≠j，因为异类就意味着不同索引
    
    Args:
        labels: 类别标签, shape [N]
                示例: [0, 1, 1, 0]
    
    Returns:
        mask: 二值掩码, shape [N, N]
              mask[i][j] = 1 当异类，否则为0
    
    示例 labels = [0, 1, 1, 0]:
        labels_equal (label[i]==label[j]):
            [[1,0,0,1],  ← 样本0,3是类别0
             [0,1,1,0],  ← 样本1,2是类别1
             [0,1,1,0],
             [1,0,0,1]]
        
        mask (取反, label[i]≠label[j]):
            [[0,1,1,0],  ← 样本0: 样本1,2是异类（负样本）
             [1,0,0,1],  ← 样本1: 样本0,3是异类（负样本）
             [1,0,0,1],  ← 样本2: 样本0,3是异类（负样本）
             [0,1,1,0]]  ← 样本3: 样本1,2是异类（负样本）
    """
    # 检查标签是否相等
    # 广播: [1,N] == [N,1] → [N,N]
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    
    # 取反得到不同标签（异或1）
    mask = labels_equal ^ 1
    
    return mask


def _get_triplet_mask(labels):
    """生成有效三元组(anchor, positive, negative)的3D掩码
    
    有效三元组(i, j, k)必须满足:
        1. i, j, k 互不相同
        2. label[i] == label[j] (正样本对)
        3. label[i] != label[k] (负样本)
    
    Args:
        labels: 类别标签, shape [N]
                示例: [0, 1, 1, 0]
    
    Returns:
        mask: 二值掩码, shape [N, N, N]
              mask[i][j][k] = 1 当triplet(i,j,k)有效，否则为0
    
    形状演变 (N=4):
        indices_not_same: [4, 4]     (i≠j 矩阵)
        i_not_equal_j:    [4, 4, 1]  (扩展到三元组)
        i_not_equal_k:    [4, 1, 4]  (扩展到三元组)
        j_not_equal_k:    [1, 4, 4]  (扩展到三元组)
        distinct_indices: [4, 4, 4]  (i≠j 且 i≠k 且 j≠k)
        
        label_equal:      [4, 4]     (label[i]==label[j] 矩阵)
        i_equal_j:        [4, 4, 1]  (扩展到三元组)
        i_equal_k:        [4, 1, 4]  (扩展到三元组)
        valid_labels:     [4, 4, 4]  (label[i]==label[j] 且 label[i]≠label[k])
        
        mask:             [4, 4, 4]  (合并所有条件)
    
    示例 labels = [0, 1, 1, 0]:
        mask[0][3][1] = 1  ✓ 有效: 锚点=0(类0), 正=3(类0), 负=1(类1)
        mask[0][0][1] = 0  ✗ 无效: 锚点=正样本 (i==j)
        mask[0][1][3] = 0  ✗ 无效: j和k是异类但配对错误
    """
    # 条件1: i, j, k 必须互不相同
    # 从矩阵开始: 对角线=0, 其他=1
    indices_not_same = torch.eye(labels.shape[0]).to(labels.device).byte() ^ 1
    
    # 扩展到3D用于三元组组合
    # i_not_equal_j[i,j,k]: 检查i≠j（与k无关）
    # i_not_equal_k[i,j,k]: 检查i≠k（与j无关）
    # j_not_equal_k[i,j,k]: 检查j≠k（与i无关）
    i_not_equal_j = torch.unsqueeze(indices_not_same, 2)  # [N, N, 1] broadcasts along 3rd dim
    i_not_equal_k = torch.unsqueeze(indices_not_same, 1)  # [N, 1, N] broadcasts along 2nd dim
    j_not_equal_k = torch.unsqueeze(indices_not_same, 0)  # [1, N, N] broadcasts along 1st dim
    
    # 合并: 三个条件都必须为真
    distinct_indices = i_not_equal_j * i_not_equal_k * j_not_equal_k
    
    # 条件2: label[i] == label[j] 且 label[i] != label[k]
    label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    
    # 扩展到3D
    i_equal_j = torch.unsqueeze(label_equal, 2)  # [N, N, 1] label[i]==label[j]
    i_equal_k = torch.unsqueeze(label_equal, 1)  # [N, 1, N] label[i]==label[k]
    
    # 合并: i和j同类，i和k异类
    valid_labels = i_equal_j * (i_equal_k ^ 1)
    
    # 最终掩码: 合并索引和标签条件
    mask = distinct_indices * valid_labels
    
    return mask