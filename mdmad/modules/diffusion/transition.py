# transition.py — DPM transitions for positions, rotations (angular kernel), and categorical sequences.
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mdmad.modules.common.layers import clampped_one_hot
from mdmad.modules.common.so3 import (
    ApproxAngularDistribution,
    random_normal_so3,
    so3vec_to_rotation,
    rotation_to_so3vec,
)


class VarianceSchedule(nn.Module):
    """
    Cosine schedule (DDPM-like). Exposes betas, alphas, alpha_bars, sigmas.
    """

    def __init__(self, num_steps=100, s=0.01):
        super().__init__()
        T = int(num_steps)
        t = torch.arange(0, T + 1, dtype=torch.float32)  # 0..T
        f_t = torch.cos((math.pi / 2) * ((t / T) + s) / (1 + s)) ** 2
        alpha_bars = f_t / f_t[0]

        betas = 1.0 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        betas = betas.clamp_max(0.999)

        sigmas = torch.zeros_like(betas)
        for i in range(1, betas.numel()):
            sigmas[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas = torch.sqrt(sigmas)

        self.register_buffer("betas", betas)  # (T+1,)
        self.register_buffer("alphas", 1.0 - betas)  # (T+1,)
        self.register_buffer("alpha_bars", alpha_bars)  # (T+1,)
        self.register_buffer("sigmas", sigmas)  # (T+1,)


class PositionTransition(nn.Module):
    """
    Standard Gaussian forward for positions in NORMALIZED coords.
    Training target: eps_p (global); compat uses LOCAL target by R_t^T in the loss, not here.
    """

    def __init__(self, num_steps, var_sched_opt={}):
        super().__init__()
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

    def add_noise(self, p_0, mask_generate, t):
        """
        p_0: (N,L,3) normalized positions
        mask_generate: (N,L) bool
        t: (N,) long in [1..T]
        Returns:
            p_t: noisy positions (N,L,3)
            eps_p: the additive noise used (N,L,3) (global)
        """
        alpha_bar = self.var_sched.alpha_bars[t]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)

        eps = torch.randn_like(p_0)
        p_t = c0 * p_0 + c1 * eps
        p_t = torch.where(mask_generate[..., None], p_t, p_0)

        return p_t, eps

    def denoise(self, p_t, eps_hat, mask_generate, t):
        """
        One reverse DDPM step with predicted epsilon (global coord).
        Uses improved DDPM style with sigma[t].
        """
        # guard: clamp alpha to avoid instability at t=T
        alpha = self.var_sched.alphas[t].clamp_min(self.var_sched.alphas[-2])
        alpha_bar = self.var_sched.alpha_bars[t]
        sigma = self.var_sched.sigmas[t].view(-1, 1, 1)

        c0 = (1.0 / torch.sqrt(alpha + 1e-8)).view(-1, 1, 1)
        c1 = ((1 - alpha) / torch.sqrt(1 - alpha_bar + 1e-8)).view(-1, 1, 1)

        z = torch.where(
            (t > 1)[:, None, None].expand_as(p_t),
            torch.randn_like(p_t),
            torch.zeros_like(p_t),
        )

        p_prev = c0 * (p_t - c1 * eps_hat) + sigma * z
        p_prev = torch.where(mask_generate[..., None], p_prev, p_t)
        return p_prev


class RotationTransition(nn.Module):
    """
    Angular-kernel forward/inverse (ApproxAngularDistribution)
    Forward: noisy R_t = E_scaled @ so3(c0*v_0)
    Inverse: sample small angular noise and left-multiply predicted next rotation.
    """

    def __init__(
        self,
        num_steps,
        var_sched_opt={},
        angular_distrib_fwd_opt={},
        angular_distrib_inv_opt={},
    ):
        super().__init__()
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

        # Forward (perturb): scale parameter uses sqrt(1 - alpha_bar)
        c1 = torch.sqrt(1 - self.var_sched.alpha_bars)  # (T+1,)
        self.angular_distrib_fwd = ApproxAngularDistribution(
            c1.tolist(), **angular_distrib_fwd_opt
        )

        # Inverse (generate): scale parameter uses sigma[t]
        sigma = self.var_sched.sigmas  # (T+1,)
        self.angular_distrib_inv = ApproxAngularDistribution(
            sigma.tolist(), **angular_distrib_inv_opt
        )

        self.register_buffer(
            "_dummy",
            torch.empty(
                [
                    0,
                ]
            ),
        )

    def add_noise(self, v_0, mask_generate, t):
        """
        Args:
            v_0:    (N, L, 3).
            mask_generate:    (N, L).
            t:  (N,).
        Returns:
            v_noisy: (N,L,3)
            e_scaled: (N,L,3) angular noise (scaled) used for forward (returned only for logging/debug)
        """
        N, L = mask_generate.size()
        alpha_bar = self.var_sched.alpha_bars[t]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)

        # Angular forward noise (scaled)
        e_scaled = random_normal_so3(
            t[:, None].expand(N, L), self.angular_distrib_fwd, device=self._dummy.device
        )  # (N,L,3)
        E_scaled = so3vec_to_rotation(e_scaled)  # (N,L,3,3)

        # Scaled true rotation
        R0_scaled = so3vec_to_rotation(c0 * v_0)  # (N,L,3,3)

        R_noisy = E_scaled @ R0_scaled
        v_noisy = rotation_to_so3vec(R_noisy)
        v_noisy = torch.where(mask_generate[..., None].expand_as(v_0), v_noisy, v_0)

        return v_noisy, e_scaled

    def denoise(self, v_t, v_next, mask_generate, t):
        """
        Args:
            v_t:    (N,L,3) current (noisy) so3vec
            v_next: (N,L,3) predicted next so3vec (from model update R_t @ U)
        Returns:
            v_prev: (N,L,3) sampled previous so3vec using angular inverse kernel
        """
        N, L = mask_generate.size()
        # Angular inverse noise
        e = random_normal_so3(
            t[:, None].expand(N, L), self.angular_distrib_inv, device=self._dummy.device
        )  # (N,L,3)
        e = torch.where(
            (t > 1)[:, None, None].expand(N, L, 3),
            e,
            torch.zeros_like(e),  # no noise at the last step
        )
        E = so3vec_to_rotation(e)
        R_prev = E @ so3vec_to_rotation(v_next)
        v_prev = rotation_to_so3vec(R_prev)
        v_prev = torch.where(mask_generate[..., None].expand_as(v_prev), v_prev, v_t)
        return v_prev


class AminoacidCategoricalTransition(nn.Module):
    """
    Discrete forward with uniform mixing; posterior is closed-form (used for KL and reverse).
    """

    def __init__(self, num_steps, num_classes=20, var_sched_opt={}):
        super().__init__()
        self.num_classes = num_classes
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)
        self.num_steps = int(num_steps)

    @staticmethod
    def _sample(c):
        """
        Args:
            c:    (N, L, K).
        Returns:
            x:    (N, L) Long
        """
        N, L, K = c.size()
        c = c.view(N * L, K) + 1e-8
        x = torch.multinomial(c, 1).view(N, L)
        return x

    def add_noise(self, x_0, mask_generate, t):
        """
        Args:
            x_0:    (N, L) Long true labels in [0..K-1]
            mask_generate: (N,L) bool
            t:      (N,) long in [1..T]
        Returns:
            c_t:    (N,L,K) probabilities after corruption
            x_t:    (N,L) sampled labels
        """
        N, L = x_0.size()
        K = self.num_classes
        c_0 = clampped_one_hot(x_0, num_classes=K).float()  # (N,L,K)

        a = self.var_sched.alpha_bars[t][:, None, None]  # (N,1,1)
        c_noisy = (a * c_0) + ((1 - a) / K)
        c_t = torch.where(mask_generate[..., None].expand(N, L, K), c_noisy, c_0)
        x_t = self._sample(c_t)
        return c_t, x_t

    def posterior(self, x_t, x0_or_probs, t):
        """
        q(x0 | x_t) for the same uniform-mixing corruption.
        Args:
            x_t:            (N,L) Long or (N,L,K) probs
            x0_or_probs:    (N,L) Long or (N,L,K) probs (true one-hot or predicted c0)
            t:              (N,) long
        Returns:
            theta: (N,L,K)
        """
        K = self.num_classes

        # x_t -> onehot
        if x_t.dim() == 3:
            c_t = x_t
        else:
            c_t = F.one_hot(x_t.clamp(min=0), num_classes=K).float()

        # x0_or_probs -> probs
        if x0_or_probs.dim() == 3:
            c0 = x0_or_probs
            c0 = c0 / (c0.sum(dim=-1, keepdim=True) + 1e-12)
        else:
            c0 = clampped_one_hot(x0_or_probs, num_classes=K).float()

        a = self.var_sched.alpha_bars[t][:, None, None]
        dot = (c0 * c_t).sum(dim=-1, keepdim=True)  # c0(j)
        theta = ((1 - a) / K) * c0 + a * dot * c_t
        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-12)
        return theta

    def denoise(self, x_t, c0_pred, mask_generate, t):
        """
        Reverse categorical step using posterior with predicted c0.
        Args:
            x_t:        (N,L) Long labels at time t
            c0_pred:    (N,L,K) probs for x0
            mask_generate: (N,L)
            t:          (N,) long time index in [1..T]
        Returns:
            post:   (N,L,K) posterior
            x_prev: (N,L) sample
        """
        K = self.num_classes
        c0_pred = (c0_pred + 1e-12) / (c0_pred.sum(dim=-1, keepdim=True) + 1e-12)
        c_t = F.one_hot(x_t.clamp(min=0), num_classes=K).float()

        a = self.var_sched.alpha_bars[t][:, None, None]
        dot = (c0_pred * c_t).sum(dim=-1, keepdim=True)
        theta = ((1 - a) / K) * c0_pred + a * dot * c_t

        theta = torch.where(mask_generate[..., None], theta, c_t)
        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-12)

        x_prev = torch.multinomial(theta.view(-1, K), 1).view(
            x_t.shape[0], x_t.shape[1]
        )
        return theta, x_prev
