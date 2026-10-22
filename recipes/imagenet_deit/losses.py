# DTS research implementation; release maintenance by DTS contributors (2026).
# See docs/provenance.md for source and changes.
# # Copyright (c) 2015-present, Facebook, Inc.
# # All rights reserved.
# """
# Implements the knowledge distillation loss
# """
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# import torch.distributed as dist
# import math

# # --- 辅助函数：时间调度曲线 ---
# def get_interpolation_alpha(t, schedule_type='linear'):
#     t = max(0.0, min(1.0, t))
#     if schedule_type == 'linear': return t
#     elif schedule_type == 'cosine': return math.sin(t * math.pi / 2) ** 2
#     elif schedule_type == 'convex': return t ** 2.0
#     elif schedule_type == 'concave': return math.sqrt(t)
#     elif schedule_type == 'sigmoid_cifar10':
#         k, x0 = 6.11, 0.57
#         s = lambda x: 1 / (1 + math.exp(-k * (x - x0)))
#         return (s(t) - s(0)) / (s(1) - s(0))
#     return t

# ==============================================================================
# LFA 核心 Loss: 基于 EMA 统计的自适应 Logit 流
# ==============================================================================
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
import math

# --- 辅助函数：时间调度曲线 ---
def get_interpolation_alpha(t, schedule_type='linear', b=1):
    t = max(0.0, min(1.0, t))
    if schedule_type == 'linear': return t
    elif schedule_type == 'constant': return 1.0
    elif schedule_type == 'deep_supervision': return 1.0
    elif schedule_type == 'cosine': return math.sin(t * math.pi / 2) ** 2
    elif schedule_type == 'tanh':
        temp = max(abs(float(b)), 1e-3)
        start = math.tanh(-1.0 / temp)
        end = math.tanh(1.0 / temp)
        value = math.tanh((2.0 * t - 1.0) / temp)
        return (value - start) / (end - start)
    elif schedule_type == 'convex': return t ** 2.0
    elif schedule_type == 'concave': return math.sqrt(t)
    elif schedule_type == 'sigmoid_cifar10':
        k, x0 = 6.11, 0.57
        s = lambda x: 1 / (1 + math.exp(-k * (x - x0)))
        return (s(t) - s(0)) / (s(1) - s(0))
    elif schedule_type == 'exponential':
        if b is None:
            raise ValueError('An exponential schedule requires a curvature parameter.')
        return t if abs(b) < 1e-12 else math.expm1(b * t) / math.expm1(b)
    return t


class AdaptiveLogitFlowLoss(nn.Module):
    def __init__(self, num_classes, num_layers, momentum=0.99, fixed_scale=None, fixed_bias=None, schedule_type='exponential', init_lfa_scale=1.0, decay_lfa_scale=False, decay_last_layer=True, final_loss="bce", **kwargs):
        """
        [Scheme B: Constant Energy Implementation]
        num_layers: 模型总层数 (blocks 数)
        fixed_scale: 如果不为 None，则关闭 EMA，使用固定 k = fixed_scale
        fixed_bias: 如果不为 None，则关闭 EMA，使用固定 b = fixed_bias
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.momentum = momentum
        self.fixed_scale = fixed_scale
        self.fixed_bias = fixed_bias
        self.schedule_type = schedule_type

        # EMA 缓冲区：k (置信度增益), b (初始置信度)
        self.register_buffer('k_ema', torch.tensor(10.0))
        self.register_buffer('b_ema', torch.tensor(0.0))
        self.init_lfa_scale = init_lfa_scale
        self.lfa_scale = init_lfa_scale
        self.decay_lfa_scale = decay_lfa_scale
        self.decay_last_layer = decay_last_layer
        self.final_loss = final_loss

    def set_progress(self, progress_ratio):
        if self.decay_lfa_scale:
            self.lfa_scale = self.init_lfa_scale * (1.0 - progress_ratio)

    def update_stats(self, logits_tensor, targets):
        """
        从 3D Logits Tensor 中提取统计量并更新 EMA。
        (保持原版逻辑不变)
        """
        if not self.training:
            return

        with torch.no_grad():
            logits_start = logits_tensor[0].float()
            logits_end = logits_tensor[-1].float()

            def get_gap(logits, y):
                B = logits.shape[0]
                if y.dim() > 1: y_idx = y.argmax(dim=-1)
                else: y_idx = y
                gt_logits = logits[torch.arange(B, device=logits.device), y_idx]
                sum_logits = logits.sum(dim=1)
                other_mean = (sum_logits - gt_logits) / (self.num_classes - 1)
                return (gt_logits - other_mean).mean()

            curr_b = get_gap(logits_start, targets)
            curr_end = get_gap(logits_end, targets)
            curr_k = curr_end - curr_b

            if dist.is_initialized():
                stats = torch.stack([curr_k, curr_b])
                dist.all_reduce(stats, op=dist.ReduceOp.AVG)
                curr_k, curr_b = stats[0], stats[1]

            curr_k = max(5.0, curr_k.item())
            curr_b = curr_b.item()

            self.k_ema.copy_(self.momentum * self.k_ema + (1 - self.momentum) * curr_k)
            self.b_ema.copy_(self.momentum * self.b_ema + (1 - self.momentum) * curr_b)

    def forward(self, logits_tensor, targets):
        # 1. 动态更新统计锚点
        if self.fixed_scale is None:
            self.update_stats(logits_tensor, targets)
            k, b = self.k_ema.item(), self.b_ema.item()
        else:
            k, b = self.fixed_scale, 0.0

        # 2. 准备目标分布
        if targets.dim() == 1:
            target_probs = F.one_hot(targets, self.num_classes).float()
        else:
            target_probs = targets # Mixup Soft Labels

        if self.final_loss == "ce":
            final_layer_loss = F.cross_entropy(logits_tensor[-1], target_probs)
        else:
            final_layer_loss = F.binary_cross_entropy_with_logits(logits_tensor[-1], target_probs.gt(0).float())

        # ======================================================================
        # 方案 B: 恒定能量流 (Constant Energy Flow)
        # ----------------------------------------------------------------------
        # 物理假设：模型的“总功率”（总 Logit 能量 Sum(e^z)）是有限且恒定的。
        # 当遇到 Mixup (如 0.5/0.5) 时，能量分散到两个类，每个类的 Logit 自然降低。
        #
        # 数学推导：
        # 单标签目标 (y=1): z = G.  Energy = e^G
        # 双标签目标 (y=0.5): z = G + ln(0.5) = G - 0.69.
        #    Energy = e^{G-0.69} + e^{G-0.69} = 0.5*e^G + 0.5*e^G = e^G.
        #
        # 结论：不需要熵补偿 H(y)。直接使用 ln(y) 进行分配即可实现能量守恒。
        # ======================================================================

        # 计算 ln(y) [B, C], 加 eps 防止 log(0)
        y_safe = target_probs + 1e-8
        log_y = torch.log(y_safe)

        # 正例掩码
        pos_mask = (target_probs > 0).float()

        losses = []
        L_total = logits_tensor.shape[0]


        # ======================================================================
        #  Fisher-Rao 路径的起点和终点 (如果使用)
        # ======================================================================
        if self.schedule_type == 'fisher_rao':
            if self.fixed_scale is None:
                raise ValueError("Fisher-Rao schedule requires a 'fixed_scale' (target logit gap) to be set.")

            with torch.no_grad():
                # --- 定义路径起点：均匀分布 ---
                y_start = torch.ones_like(target_probs) / self.num_classes
                q_start = torch.sqrt(y_start)

                # --- 定义路径终点：由 fixed_scale (target_gap) 决定 ---
                # 1. 构造终点的 Logits
                z_end_raw = self.fixed_scale + log_y
                z_end = z_end_raw * pos_mask
                # 2. 得到终点的概率分布
                p_end = F.softmax(z_end, dim=-1)
                q_end = torch.sqrt(p_end)

                # --- 计算球面夹角 Omega ---
                cos_omega = (q_start * q_end).sum(dim=-1, keepdim=True)
                cos_omega = torch.clamp(cos_omega, -1.0 + 1e-6, 1.0 - 1e-6)
                omega = torch.acos(cos_omega)
                sin_omega = torch.sin(omega)


        for i in range(0, L_total):
            logits = logits_tensor[i].float()

            t = (i + 1) / L_total

            # --- 构造目标概率 p_target ---
            if self.schedule_type == 'deep_supervision':
                p_target = target_probs
            elif self.schedule_type == 'fisher_rao':
                with torch.no_grad():
                    # 球面线性插值 (Slerp)
                    eps = 1e-7
                    sin_omega_safe = torch.where(sin_omega < eps, torch.ones_like(sin_omega), sin_omega)

                    scale0 = torch.sin((1 - t) * omega) / sin_omega_safe
                    scale1 = torch.sin(t * omega) / sin_omega_safe

                    q_t = scale0 * q_start + scale1 * q_end

                    # 对于起点终点重合的情况，直接取 q_end
                    q_t = torch.where(sin_omega < eps, q_end, q_t)

                    # 映射回概率并归一化
                    p_target = q_t.pow(2)
                    p_target = p_target / p_target.sum(dim=-1, keepdim=True)
            else:
                # 原有逻辑: 基于 Logits Gap 的调度
                alpha = get_interpolation_alpha(t, self.schedule_type, self.fixed_bias)

                # --- 构造 Logit 目标 (无熵补偿) ---
                # 基础 Gap (理想的单标签强度)
                target_gap_base = k * alpha + b


                with torch.no_grad():
                    # 1. 基础高度 = 目标 Gap (不加熵！)
                    # 这意味着对于 Mixup 样本，Logit 值会比单标签样本低
                    base_magnitude = target_gap_base

                    # 2. 比例分配：加上 ln(y_i)
                    # 例子：G=10, y=[0.5, 0.5]
                    # z_target = 10 + ln(0.5) ≈ 9.31
                    # Gap = 9.31 (比单标签的 10 小了 0.69)
                    z_target_raw = base_magnitude + log_y

                    # 3. 掩码：只保留正例，负例设为 0
                    z_target = z_target_raw * pos_mask

                    # 4. 转为概率分布
                    # 虽然 Logit 值降低了，但 Softmax 后的比例依然完美符合 y
                    p_target = F.softmax(z_target, dim=-1)

            # print(i, target_gap_base, p_target[pos_mask.bool()], target_probs[pos_mask.bool()])

            # 计算 Loss
            l_loss = torch.sum(-p_target * F.log_softmax(logits, dim=-1), dim=-1).mean()
            l_loss = l_loss + 0.0 * logits_tensor[i].sum()
            losses.append(l_loss)

        if self.decay_last_layer:
            return self.lfa_scale * (sum(losses) / len(losses)) + (1.0 - self.lfa_scale) * final_layer_loss
        else:
            return self.lfa_scale * (sum(losses) / len(losses)) + final_layer_loss


# ==============================================================================
# LFA 核心 Loss: 基于 EMA 统计的自适应 Logit 流 (适配 3D Tensor)
# ==============================================================================
class AdaptiveLogitFlowLoss_old(nn.Module):
    def __init__(self, num_classes, num_layers, momentum=0.99, fixed_scale=None, fixed_bias=None, schedule_type='linear', init_lfa_scale=1.0, decay_lfa_scale=False, decay_last_layer=True):
        """
        [Scheme B: Constant Energy Implementation]
        num_layers: 模型总层数 (blocks 数)
        fixed_scale: 如果不为 None，则关闭 EMA，使用固定 k = fixed_scale
        fixed_bias: 如果不为 None，则关闭 EMA，使用固定 b = fixed_bias
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.momentum = momentum
        self.fixed_scale = fixed_scale
        self.fixed_bias = fixed_bias
        self.schedule_type = schedule_type

        # EMA 缓冲区：k (置信度增益), b (初始置信度)
        self.register_buffer('k_ema', torch.tensor(10.0))
        self.register_buffer('b_ema', torch.tensor(0.0))
        self.init_lfa_scale = init_lfa_scale
        self.lfa_scale = init_lfa_scale
        self.decay_lfa_scale = decay_lfa_scale
        self.decay_last_layer = decay_last_layer

    def set_progress(self, progress_ratio):
        if self.decay_lfa_scale:
            self.lfa_scale = self.init_lfa_scale * (1.0 - progress_ratio)

    def update_stats(self, logits_tensor, targets):
        """
        从 3D Logits Tensor 中提取统计量并更新 EMA。
        (保持原版逻辑不变)
        """
        if not self.training:
            return

        with torch.no_grad():
            logits_start = logits_tensor[0].float()
            logits_end = logits_tensor[-1].float()

            def get_gap(logits, y):
                B = logits.shape[0]
                if y.dim() > 1: y_idx = y.argmax(dim=-1)
                else: y_idx = y
                gt_logits = logits[torch.arange(B, device=logits.device), y_idx]
                sum_logits = logits.sum(dim=1)
                other_mean = (sum_logits - gt_logits) / (self.num_classes - 1)
                return (gt_logits - other_mean).mean()

            curr_b = get_gap(logits_start, targets)
            curr_end = get_gap(logits_end, targets)
            curr_k = curr_end - curr_b

            if dist.is_initialized():
                stats = torch.stack([curr_k, curr_b])
                dist.all_reduce(stats, op=dist.ReduceOp.AVG)
                curr_k, curr_b = stats[0], stats[1]

            curr_k = max(5.0, curr_k.item())
            curr_b = curr_b.item()

            self.k_ema.copy_(self.momentum * self.k_ema + (1 - self.momentum) * curr_k)
            self.b_ema.copy_(self.momentum * self.b_ema + (1 - self.momentum) * curr_b)

    def forward(self, logits_tensor, targets):
            # 1. 动态更新统计锚点
            if self.fixed_scale is None:
                self.update_stats(logits_tensor, targets)
                k, b = self.k_ema.item(), self.b_ema.item()
            else:
                k, b = self.fixed_scale, 0.0

            # 2. 准备目标分布和索引 (修正点：定义 y_idx)
            if targets.dim() == 1:
                target_probs = F.one_hot(targets, self.num_classes).float()
                y_idx = targets
            else:
                target_probs = targets # Mixup Soft Labels
                y_idx = targets.argmax(dim=-1) # 取 Mixup 权重最大的类作为 Gap 基准

            final_ce_loss = F.cross_entropy(logits_tensor[-1], target_probs)
            final_bce_loss = F.binary_cross_entropy_with_logits(logits_tensor[-1], target_probs.gt(0).float())

            # 计算 ln(y) [B, C], 加 eps 防止 log(0)
            y_safe = target_probs + 1e-8
            log_y = torch.log(y_safe)
            pos_mask = (target_probs > 0).float()

            losses = []
            L_total = logits_tensor.shape[0]

            for i in range(0, L_total):
                logits = logits_tensor[i].float()

                t = (i + 1) / L_total
                alpha = get_interpolation_alpha(t, self.schedule_type, self.fixed_bias)

                target_gap_base = k * alpha + b

                with torch.no_grad():
                    # --- 修正点：计算 actual_gap ---
                    # gt_logits shape: [B]
                    gt_logits = logits.gather(1, y_idx.unsqueeze(1)).squeeze(1)
                    sum_logits = logits.sum(dim=1)
                    other_mean = (sum_logits - gt_logits) / (self.num_classes - 1)
                    actual_gap = gt_logits - other_mean

                    # 构造目标概率分布
                    z_target = (target_gap_base + log_y) * pos_mask
                    p_target = F.softmax(z_target, dim=-1)

                # --- 修正点：per_sample_loss 不要在这里加 .mean() ---
                # shape: [B]
                per_sample_loss = torch.sum(-p_target * F.log_softmax(logits, dim=-1), dim=-1)

                # 如果 实际 Gap >= 目标 Gap，Mask 为 0
                mask = (actual_gap < target_gap_base).float()

                # --- 修正点：使用掩码计算平均 Loss ---
                if mask.sum() > 0:
                    # 仅对未达标的样本求平均
                    l_loss = (per_sample_loss * mask).sum() / (mask.sum() + 1e-8)
                else:
                    # 如果全员达标，Loss 归零，但乘一个 scale 保持计算图连通性（可选）
                    l_loss = per_sample_loss.mean() * 0.0

                losses.append(l_loss)

            # 3. 最终 Loss 合成
            intermediate_loss = sum(losses) / len(losses)
            if self.decay_last_layer:
                loss = self.lfa_scale * intermediate_loss + (1.0 - self.lfa_scale) * final_bce_loss
            else:
                loss = self.lfa_scale * intermediate_loss + final_bce_loss

            return loss

class AdaptiveBCELogitFlowLoss(nn.Module):
    def __init__(self, num_classes, num_layers, momentum=0.99, fixed_scale=10.0, fixed_bias=None,
                 schedule_type='exponential', schedule_k=0.5,
                 init_lfa_scale=1.0, decay_lfa_scale=False, decay_last_layer=True,
                 negative_scale=1.0):
        """
        Args:
            negative_scale: 负例下降的比例因子。
                            1.0 表示负例下降幅度与正例上升幅度相同 (对称分离)。
                            0.5 表示负例只下降 k/2。
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.momentum = momentum
        self.fixed_scale = fixed_scale
        self.fixed_bias = fixed_bias
        self.schedule_type = schedule_type
        self.schedule_k = schedule_k
        self.negative_scale = negative_scale # 新增：控制负例下降幅度

        # EMA 缓冲区
        # k: 正例的增长幅度 (Positive Gain)
        # b: 初始 Bias (p ~ 1/1000)
        self.register_buffer('k_ema', torch.tensor(14.0))
        self.register_buffer('b_ema', torch.tensor(-7.0))

        self.init_lfa_scale = init_lfa_scale
        self.lfa_scale = init_lfa_scale
        self.decay_lfa_scale = decay_lfa_scale
        self.decay_last_layer = decay_last_layer

    def set_progress(self, progress_ratio):
        if self.decay_lfa_scale:
            self.lfa_scale = self.init_lfa_scale * (1.0 - progress_ratio)

    def update_stats(self, logits_tensor, targets):
        """
        统计逻辑保持不变，k 依然代表 '正例相对于起点的增益'。
        负例的下降由 negative_scale * k 推导，不需要额外统计。
        """
        if not self.training:
            return

        with torch.no_grad():
            logits_start = logits_tensor[0].detach().float()
            logits_end = logits_tensor[-1].detach().float()

            # 1. b = 起始层的平均 Logit (背景噪声水平)
            curr_b = logits_start.mean()

            # 2. k = 最后一层正例 Logit - 起始层 b
            # (只统计正例的增长能力)
            if targets.dim() > 1:
                pos_mask = (targets > 0.5).float()
            else:
                pos_mask = F.one_hot(targets, self.num_classes).float()

            num_pos = pos_mask.sum()
            if num_pos > 0:
                pos_logits_end = (logits_end * pos_mask).sum() / num_pos
                curr_end = pos_logits_end
            else:
                curr_end = logits_end.max()

            curr_k = curr_end - curr_b

            if dist.is_initialized():
                stats = torch.stack([curr_k, curr_b])
                dist.all_reduce(stats, op=dist.ReduceOp.AVG)
                curr_k, curr_b = stats[0], stats[1]

            curr_k = max(1.0, curr_k.item())
            curr_b = curr_b.item()

            self.k_ema.copy_(self.momentum * self.k_ema + (1 - self.momentum) * curr_k)
            self.b_ema.copy_(self.momentum * self.b_ema + (1 - self.momentum) * curr_b)

    def forward(self, logits_tensor, targets):
        # 1. 获取统计参数
        if self.fixed_scale is None:
            self.update_stats(logits_tensor, targets)
            k, b = self.k_ema.item(), self.b_ema.item()
        else:
            k = self.fixed_scale
            # 默认 b 大约在 -k/2 或者用户指定
            b = self.fixed_bias if self.fixed_bias is not None else math.log(1 / (self.num_classes - 1))

        if self.fixed_bias is not None:
            b = self.fixed_bias

        # 2. 准备目标分布
        if targets.dim() == 1:
            target_probs = F.one_hot(targets, self.num_classes).float()
        else:
            target_probs = targets

        final_loss = F.binary_cross_entropy_with_logits(logits_tensor[-1], target_probs)

        if self.lfa_scale < 1e-6:
            return final_loss

        losses = []
        L_total = len(logits_tensor)

        for i in range(L_total):
            logits = logits_tensor[i]

            if self.schedule_type == 'deep_supervision':
                target_prob_sigmoid = target_probs
            else:
                # 计算进度 alpha (0 -> 1)
                t = (i + 1) / L_total
                alpha = get_interpolation_alpha(t, self.schedule_type, self.schedule_k)

                # ======================================================================
                # 双向 Flow 逻辑 (Bidirectional Flow)
                # ----------------------------------------------------------------------
                # 公式: Target = b + alpha * (Pos_Force - Neg_Force)
                # Pos_Force = y * k
                # Neg_Force = (1-y) * k * neg_scale
                #
                # Case y=1 (正例): Target -> b + k
                # Case y=0 (负例): Target -> b - k * neg_scale
                # Case y=0.5 (Mixup): Target -> b + 0.5*k - 0.5*k*scale
                # ======================================================================

                with torch.no_grad():
                    # 计算正向推力和负向推力
                    pos_push = target_probs * k
                    neg_push = (1.0 - target_probs) * (k * self.negative_scale)

                    # 净推力
                    net_push = pos_push - neg_push

                    # 目标 Logit
                    target_logit = b + alpha * net_push

                    # 转为 Sigmoid 概率供 BCE 使用
                    target_prob_sigmoid = torch.sigmoid(target_logit)

            # 计算 BCE Loss
            l_loss = F.binary_cross_entropy_with_logits(logits, target_prob_sigmoid)
            l_loss = l_loss + 0.0 * logits.sum()
            losses.append(l_loss)

        intermediate_loss = sum(losses) / len(losses)
        if self.decay_last_layer:
            loss = self.lfa_scale * intermediate_loss + (1.0 - self.lfa_scale) * final_loss
        else:
            loss = self.lfa_scale * intermediate_loss + final_loss
        return loss


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                #We provide the teacher's targets in log probability because we use log_target=True
                #(as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                #but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
            #We divide by outputs_kd.numel() to have the legacy PyTorch behavior.
            #But we also experiments output_kd.size(0)
            #see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss


class LFALoss(nn.Module):
    def __init__(self, num_layers, num_classes, initial_k=10.0, reg_weight=1.0,
                 learn_k=False, loss_type='incremental', use_bce=False):
        super().__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.reg_weight = reg_weight
        self.learn_k = learn_k
        self.loss_type = loss_type
        self.use_bce = use_bce

        if self.learn_k:
            self.k = nn.Parameter(torch.tensor(initial_k))
        else:
            self.register_buffer('k', torch.tensor(initial_k))

    def forward(self, logits_tensor, targets):
        # logits_tensor: [L, B, C]
        L = logits_tensor.shape[0]
        B = logits_tensor.shape[1]
        C = logits_tensor.shape[2]

        # [核心修复]：确保 targets 是 [B, C] 的浮点张量
        if targets.dim() == 1:
            # 如果是类别索引，转为 One-hot
            targets = F.one_hot(targets, num_classes=self.num_classes).float()
        else:
            # 如果已经是 2D，确保是 float 类型
            targets = targets.float()

        # 针对 cosub 等特殊情况，如果 B 不匹配，进行对齐（防止 targets 被 cat 翻倍后不匹配）
        if targets.shape[0] != B:
            targets = targets[:B]

        k_pos = F.softplus(self.k) if self.learn_k else self.k
        step_k = k_pos / L

        # 1. 最终层损失
        final_logits = logits_tensor[-1]
        if self.use_bce:
            final_loss = F.binary_cross_entropy_with_logits(final_logits, targets)
        else:
            # 这里的 targets 必须是 [B, C], log_softmax 是 [B, C]
            final_loss = torch.sum(-targets * F.log_softmax(final_logits, dim=-1), dim=-1).mean()

        final_loss = 0

        flow_losses = []
        if self.loss_type == 'incremental':
            step_val = step_k
            for i in range(L - 1):
                z_prev = logits_tensor[i].detach().float()
                z_curr = logits_tensor[i+1].float()
                with torch.no_grad():
                    if self.use_bce:
                        p_target = torch.sigmoid(z_prev + targets * step_val)
                    else:
                        p_target = F.softmax(z_prev + targets * step_val, dim=-1)

                if self.use_bce:
                    l_loss = F.binary_cross_entropy_with_logits(z_curr, p_target)
                else:
                    l_loss = torch.sum(-p_target * F.log_softmax(z_curr, dim=-1), dim=-1).mean()

                # DDP 梯度补偿补丁
                l_loss = l_loss + 0.0 * logits_tensor[i].sum()
                flow_losses.append(l_loss)

        elif self.loss_type == 'linear':
            for i in range(L - 1):
                z_curr = logits_tensor[i].float()
                t = (i + 1) / L
                with torch.no_grad():
                    if self.use_bce:
                        p_target = torch.sigmoid(t * k_pos * targets)
                    else:
                        p_target = F.softmax(t * k_pos * targets, dim=-1)

                if self.use_bce:
                    l_loss = F.binary_cross_entropy_with_logits(z_curr, p_target)
                else:
                    l_loss = torch.sum(-p_target * F.log_softmax(z_curr, dim=-1), dim=-1).mean()
                flow_losses.append(l_loss)

        total_flow_loss = sum(flow_losses) / len(flow_losses)
        if self.learn_k:
            reg_loss = self.reg_weight * (1.0 / (k_pos + 1e-6)) + 0.01 * (k_pos ** 2)
            return final_loss + total_flow_loss + reg_loss
        return final_loss + total_flow_loss
