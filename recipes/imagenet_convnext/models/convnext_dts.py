# DTS research implementation; release maintenance by DTS contributors (2026).
# See docs/provenance.md for source and changes.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model

from models.convnext import ConvNeXt


class ConvNeXtTrajectory(nn.Module):
    def __init__(
        self,
        depths,
        dims,
        num_classes=1000,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
        traj_strategy="none",
        traj_scale=0.001,
        traj_beta=0.5,
        traj_granularity="stage",
        **kwargs,
    ):
        super().__init__()
        self.traj_strategy = traj_strategy
        self.traj_scale = traj_scale
        self.traj_beta = traj_beta
        self.traj_granularity = traj_granularity
        self.num_classes = num_classes
        self.depths = list(depths)
        self.dims = list(dims)

        self.backbone = ConvNeXt(
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            head_init_scale=head_init_scale,
        )

        if self.traj_granularity == "stage":
            # Stem feature plus the four stage endpoints.
            self.aux_dims = [dims[0], dims[0], dims[1], dims[2], dims[3]]
        elif self.traj_granularity == "block":
            # Stem feature plus every ConvNeXt block output. The final block slot
            # is replaced with the normalized classifier input feature.
            self.aux_dims = [dims[0]]
            for dim, depth in zip(dims, depths):
                self.aux_dims.extend([dim] * depth)
        else:
            raise ValueError(f"Unsupported trajectory granularity: {self.traj_granularity}")
        self.aux_norms = nn.ModuleList([nn.LayerNorm(dim, eps=1e-6) for dim in self.aux_dims])
        self.aux_heads = nn.ModuleList([nn.Linear(dim, num_classes) for dim in self.aux_dims])

    def _pool(self, x):
        return x.mean([-2, -1])

    def _forward_with_aux(self, x):
        aux_states = []

        x = self.backbone.downsample_layers[0](x)
        aux_states.append(self._pool(x))

        if self.traj_granularity == "stage":
            x = self.backbone.stages[0](x)
            aux_states.append(self._pool(x))

            x = self.backbone.downsample_layers[1](x)
            x = self.backbone.stages[1](x)
            aux_states.append(self._pool(x))

            x = self.backbone.downsample_layers[2](x)
            x = self.backbone.stages[2](x)
            aux_states.append(self._pool(x))

            x = self.backbone.downsample_layers[3](x)
            x = self.backbone.stages[3](x)
            final_feature = self.backbone.norm(self._pool(x))
            aux_states.append(final_feature)
        else:
            stage0 = x
            for block in self.backbone.stages[0]:
                stage0 = block(stage0)
                aux_states.append(self._pool(stage0))

            stage1 = self.backbone.downsample_layers[1](stage0)
            for block in self.backbone.stages[1]:
                stage1 = block(stage1)
                aux_states.append(self._pool(stage1))

            stage2 = self.backbone.downsample_layers[2](stage1)
            for block in self.backbone.stages[2]:
                stage2 = block(stage2)
                aux_states.append(self._pool(stage2))

            stage3 = self.backbone.downsample_layers[3](stage2)
            for block in self.backbone.stages[3]:
                stage3 = block(stage3)
                aux_states.append(self._pool(stage3))

            final_feature = self.backbone.norm(self._pool(stage3))
            aux_states[-1] = final_feature

        aux_logits = []
        for feat, norm, head in zip(aux_states, self.aux_norms, self.aux_heads):
            aux_logits.append(head(norm(feat)))

        logits = self.backbone.head(final_feature)
        return {
            "logits": logits,
            "aux_logits": torch.stack(aux_logits, dim=0),
        }

    def forward(self, x):
        if self.traj_strategy == "none":
            return self.backbone(x)
        return self._forward_with_aux(x)


@register_model
def convnext_tiny_dts(pretrained=False, in_22k=False, **kwargs):
    if pretrained or in_22k:
        raise ValueError("convnext_tiny_dts only supports training from scratch in this repo.")
    return ConvNeXtTrajectory(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        **kwargs,
    )


@register_model
def convnext_base_dts(pretrained=False, in_22k=False, **kwargs):
    if pretrained or in_22k:
        raise ValueError("convnext_base_dts only supports training from scratch in this repo.")
    return ConvNeXtTrajectory(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        **kwargs,
    )


class TrajectoryLoss(nn.Module):
    def __init__(self, num_classes, strategy="none", aux_scale=0.001, beta=0.5, smoothing=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.strategy = strategy
        self.aux_scale = aux_scale
        self.beta = beta
        self.smoothing = smoothing

    def _ensure_prob_targets(self, targets):
        if targets.ndim == 2:
            return targets.float()
        target_probs = F.one_hot(targets, self.num_classes).float()
        if self.smoothing > 0:
            target_probs = target_probs * (1.0 - self.smoothing) + self.smoothing / self.num_classes
        return target_probs

    def _soft_ce(self, logits, target_probs):
        return -(target_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

    def _depth_weight(self, depth_idx, total_depths):
        if total_depths <= 1:
            return 1.0
        t = float(depth_idx + 1) / float(total_depths)
        if abs(self.beta) < 1e-12:
            return t
        return (math.exp(self.beta * t) - 1.0) / (math.exp(self.beta) - 1.0)

    def forward(self, outputs, targets):
        if not isinstance(outputs, dict):
            target_probs = self._ensure_prob_targets(targets)
            return self._soft_ce(outputs, target_probs)

        logits = outputs["logits"]
        aux_logits = outputs["aux_logits"]
        target_probs = self._ensure_prob_targets(targets)

        final_loss = self._soft_ce(logits, target_probs)
        if self.strategy == "none":
            return final_loss

        aux_losses = []
        uniform = torch.full_like(target_probs, 1.0 / self.num_classes)
        total_aux = aux_logits.shape[0]
        for idx in range(total_aux):
            if self.strategy == "local_deep_supervision":
                aux_target = target_probs
            elif self.strategy == "dts_exponential":
                alpha = self._depth_weight(idx, total_aux)
                aux_target = alpha * target_probs + (1.0 - alpha) * uniform
            else:
                raise ValueError(f"Unsupported trajectory strategy: {self.strategy}")
            aux_losses.append(self._soft_ce(aux_logits[idx], aux_target))

        return final_loss + self.aux_scale * torch.stack(aux_losses).mean()
