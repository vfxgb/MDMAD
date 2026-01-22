# mdmad.py — DPM with MDN heads
# - Sequence: categorical DPM with posterior-KL training and discrete reverse.
# - Position: MDN with NLL loss in local frame.
# - Rotation: MDN with NLL loss using quaternion parametrization.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from mdmad.modules.common.geometry import (
    apply_rotation_to_vector,
    quaternion_1ijk_to_rotation_matrix,
)
from mdmad.modules.common.so3 import (
    so3vec_to_rotation,
    rotation_to_so3vec,
    random_uniform_so3,
)
from mdmad.modules.encoders.ga import GAEncoder
from mdmad.modules.common.layers import clampped_one_hot
from .transition import (
    AminoacidCategoricalTransition,
    PositionTransition,
    RotationTransition,
)


# numeric guards / helpers
EPS = 1e-6
SIGMA_FLOOR = 1e-4
SIGMA_CAP = 10.0
LOG_STD_MAX = 6.9
MDN_LOGITS_CLAMP = 50.0
MEAN_POS_CLAMP = 5.0
MEAN_ROT_CLAMP = 3.0
SAFE_BIG = 1e4


def sanitize_(x: torch.Tensor, clamp_abs: float = None) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if clamp_abs is not None:
        x = x.clamp(min=-clamp_abs, max=clamp_abs)
    return x


def rotation_matrix_cosine_loss(R_pred, R_true):
    """
    Args:
        R_pred: (*, 3, 3); R_true: (*, 3, 3)
    Returns:
        (*,)
    """
    size = list(R_pred.shape[:-2])
    ncol = R_pred.numel() // 3
    RT_pred = R_pred.transpose(-2, -1).reshape(ncol, 3)
    RT_true = R_true.transpose(-2, -1).reshape(ncol, 3)
    ones = torch.ones(
        [
            ncol,
        ],
        dtype=torch.long,
        device=R_pred.device,
    )
    loss = F.cosine_embedding_loss(RT_pred, RT_true, ones, reduction="none")
    loss = loss.reshape(size + [3]).sum(dim=-1)
    return loss


def mdn_nll_iso_loss(
    pred_means: torch.Tensor,  # (N,L,K,D)
    target: torch.Tensor,  # (N,L,D)
    pred_logstds: torch.Tensor,  # (N,L,K,1) isotropic per component
    pred_logits: torch.Tensor,  # (N,L,K)
) -> torch.Tensor:
    """
    Isotropic-covariance MDN NLL (one std per component, shared across D).
    Returns: (N,L)
    """
    D = pred_means.size(-1)

    mu = sanitize_(pred_means)
    log_s = sanitize_(pred_logstds).clamp(
        min=math.log(EPS), max=LOG_STD_MAX
    )  # (N,L,K,1)
    logits = sanitize_(pred_logits).clamp(min=-MDN_LOGITS_CLAMP, max=MDN_LOGITS_CLAMP)

    inv_s = torch.exp(-log_s).clamp(max=1.0 / EPS)  # (N,L,K,1)
    diff = (mu - target.unsqueeze(-2)) * inv_s  # (N,L,K,D)
    quad = diff.square().sum(dim=-1).clamp(max=SAFE_BIG)  # (N,L,K)

    # log |Σ| = D * log(s^2) = 2D * log s
    log_det = (2.0 * D) * log_s.squeeze(-1)  # (N,L,K)
    gaussian_ll = -0.5 * (quad + log_det + D * math.log(2 * math.pi))  # (N,L,K)

    logweights = F.log_softmax(logits, dim=-1)  # (N,L,K)
    ll = torch.logsumexp(gaussian_ll + logweights, dim=-1)  # (N,L)
    loss = -ll
    return torch.nan_to_num(loss, nan=SAFE_BIG, posinf=SAFE_BIG, neginf=SAFE_BIG).clamp(
        min=0.0, max=SAFE_BIG
    )


def rotation_mdn_nll_loss(
    pred_means: torch.Tensor,  # (N,L,K,3) quaternion imaginary parts
    R_target: torch.Tensor,  # (N,L,3,3)
    R_current: torch.Tensor,  # (N,L,3,3)
    pred_logstds: torch.Tensor,  # (N,L,K,1)   # isotropic σ per site
    pred_logits: torch.Tensor,  # (N,L,K)
) -> torch.Tensor:
    N, L, K, _ = pred_means.shape
    D = 3  # rotation space parameterized by a 3D tangent vector

    mu = sanitize_(pred_means, clamp_abs=MEAN_ROT_CLAMP)  # (N,L,K,3)
    log_s = sanitize_(pred_logstds).clamp(
        min=math.log(EPS), max=LOG_STD_MAX
    )  # (N,L,K,1)
    logits = sanitize_(pred_logits).clamp(min=-MDN_LOGITS_CLAMP, max=MDN_LOGITS_CLAMP)

    # Component rotations
    U = quaternion_1ijk_to_rotation_matrix(mu.reshape(N * L * K, 3)).reshape(
        N, L, K, 3, 3
    )  # (N,L,K,3,3)
    R_pred = R_current.unsqueeze(2) @ U  # (N,L,K,3,3)
    R_tgt_exp = R_target.unsqueeze(2).expand_as(R_pred)

    # Cosine distance proxy (sum over 3 rows)
    nrows = R_pred.numel() // 3
    RT_pred = R_pred.transpose(-2, -1).reshape(nrows, 3)
    RT_true = R_tgt_exp.transpose(-2, -1).reshape(nrows, 3)
    ones = torch.ones(nrows, device=R_pred.device)

    cosine_dist = (
        F.cosine_embedding_loss(RT_pred, RT_true, ones, reduction="none")
        .reshape(N, L, K, 3)
        .sum(dim=-1)
    )  # (N,L,K)

    # Isotropic Gaussian proxy NLL per component:
    # 0.5 * dist/σ^2  + D * log σ   (+ const, optional)
    sigma_sq = torch.exp(2 * log_s).squeeze(-1)  # (N,L,K)
    scaled_dist = cosine_dist / (2 * sigma_sq.clamp(min=EPS))
    log_norm = D * log_s.squeeze(-1)  # (N,L,K)

    # (no need constant) const = 0.5 * D * log(2π); it doesn't affect grads w.r.t.  a, μ, σ
    nll_per_comp = scaled_dist + log_norm  # + 0.5 * D * math.log(2 * math.pi)

    # Mixture combine
    logweights = F.log_softmax(logits, dim=-1)  # (N,L,K)
    ll = torch.logsumexp(-nll_per_comp + logweights, dim=-1)  # (N,L)
    loss = -ll
    return torch.nan_to_num(loss, nan=SAFE_BIG, posinf=SAFE_BIG, neginf=SAFE_BIG).clamp(
        min=0.0, max=SAFE_BIG
    )


# EpsilonNet (MDN heads) + c0 logits for sequence
class EpsilonNet(nn.Module):
    """
    MDN heads for position & rotation; c0 logits for sequence (categorical DPM).
    - Position MDN is in LOCAL frame (3D).
    - Rotation MDN output is a 3-vector interpreted by quaternion_1ijk_to_rotation_matrix.
    - IMPORTANT: we feed integer sequence indices through nn.Embedding (compat with original).
    """

    def __init__(
        self,
        res_feat_dim,
        pair_feat_dim,
        num_layers,
        K_pos: int = 8,
        K_rot: int = 8,
        encoder_opt={},
    ):
        super().__init__()
        self.K_pos = int(K_pos)
        self.K_rot = int(K_rot)

        # Sequence embedding with integer indices
        self.current_sequence_embedding = nn.Embedding(25, res_feat_dim)

        # Pre-encoder fusion of current sequence into residue features
        self.res_feat_mixer = nn.Sequential(
            nn.Linear(res_feat_dim * 2, res_feat_dim),
            nn.ReLU(),
            nn.Linear(res_feat_dim, res_feat_dim),
        )
        self.encoder = GAEncoder(res_feat_dim, pair_feat_dim, num_layers, **encoder_opt)

        # After encoder, we will concatenate a 3-dim time feature [β, sinβ, cosβ]
        in_dim = res_feat_dim + 3

        # Sequence head: logits for p(x0) (no softmax here)
        self.seq_head = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.ReLU(),
            nn.Linear(in_dim * 2, in_dim * 2),
            nn.ReLU(),
            nn.Linear(in_dim * 2, 20),
        )

        # MDN heads (isotropic per-site shared σ)
        Dp = 3
        Dr = 3
        self.pos_head = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.ReLU(),
            nn.Linear(in_dim * 2, in_dim * 2),
            nn.ReLU(),
            nn.Linear(
                in_dim * 2, self.K_pos * Dp + 1 + self.K_pos
            ),  # K*D + 1(shared logσ) + K(logits)
        )
        self.rot_head = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.ReLU(),
            nn.Linear(in_dim * 2, in_dim * 2),
            nn.ReLU(),
            nn.Linear(
                in_dim * 2, self.K_rot * Dr + 1 + self.K_rot
            ),  # K*D + 1(shared logσ) + K(logits)
        )

    def _unpack_mdn_iso(self, raw: torch.Tensor, K: int, D: int, mean_clamp: float):
        """
        Single shared log-σ per (N,L) site; expanded to (N,L,K,1) for shape convenience.
        Returns:
            mu (N,L,K,D), log_sigma (N,L,K,1), logits (N,L,K), log_sigma_shared (N,L,1,1)
        """
        N, L = raw.shape[:2]
        sz_mu = K * D
        sz_lsig = 1
        mu = raw[..., :sz_mu].view(N, L, K, D)
        raw_lsig = raw[..., sz_mu : sz_mu + sz_lsig].view(N, L, 1, 1)
        logits = raw[..., sz_mu + sz_lsig :]
        mu = torch.tanh(mu) * mean_clamp
        sigma = F.softplus(raw_lsig) + max(SIGMA_FLOOR, EPS)
        if SIGMA_CAP is not None and SIGMA_CAP > 0:
            sigma = sigma.clamp(max=SIGMA_CAP)
        log_sigma_shared = torch.log(sigma.clamp(min=EPS)).clamp(
            min=math.log(EPS), max=LOG_STD_MAX
        )
        log_sigma = log_sigma_shared.expand(N, L, K, 1)
        logits = (logits - logits.max(dim=-1, keepdim=True).values).clamp(
            min=-MDN_LOGITS_CLAMP, max=MDN_LOGITS_CLAMP
        )
        return (
            sanitize_(mu, clamp_abs=mean_clamp),
            sanitize_(log_sigma),
            sanitize_(logits),
            sanitize_(log_sigma_shared),
        )

    def forward(
        self, R, p, s_idx, beta_scalar, res_feat, pair_feat, mask_generate, mask_res
    ):
        """
        Inputs:
            R:          (N,L,3,3) rotation matrices at time t
            p:          (N,L,3) normalized positions at time t
            s_idx:      (N,L) integer sequence indices
            beta_scalar:(N,) float in [0,1] (we form [β, sinβ, cosβ])
        Returns:
            c0_pred_logits: (N,L,20)
            vel_pos_exp:    (N,L,3) expected global eps for positions
            u_exp:          (N,L,3) rotation "imaginary" vector (quaternion parametrization)
            (mu_p, log_sigma_p, logit_pi_p, log_sigma_p_shared)
            (mu_r, log_sigma_r, logit_pi_r, log_sigma_r_shared)
        """
        N, L = mask_res.size()

        # Fuse current sequence indices via embedding (compat)
        s_emb = self.current_sequence_embedding(s_idx)  # (N,L,res_dim)
        res_feat_fused = self.res_feat_mixer(torch.cat([res_feat, s_emb], dim=-1))

        # Graph encoder (unchanged)
        enc = self.encoder(R, p, res_feat_fused, pair_feat, mask_res)  # (N,L,res_dim)

        # 3-dim time feature t_embed = [β, sinβ, cosβ]
        t_embed = torch.stack(
            [beta_scalar, torch.sin(beta_scalar), torch.cos(beta_scalar)], dim=-1
        )[
            :, None, :
        ]  # (N,1,3)
        t_embed = t_embed.expand(N, L, 3)

        in_feat = torch.cat([enc, t_embed], dim=-1)  # (N,L,res_dim+3)

        # Sequence logits (for c0)
        c0_pred_logits = self.seq_head(in_feat)
        c0_pred_logits = torch.where(
            mask_generate[..., None], c0_pred_logits, torch.zeros_like(c0_pred_logits)
        )

        # Position MDN (LOCAL, isotropic)
        pos_raw = self.pos_head(in_feat)
        mu_p, log_sigma_p, logit_pi_p, log_sigma_p_shared = self._unpack_mdn_iso(
            pos_raw, self.K_pos, 3, MEAN_POS_CLAMP
        )
        pi_p = F.softmax(logit_pi_p, dim=-1)[..., None]
        exp_local = (pi_p * mu_p).sum(dim=-2)  # (N,L,3)
        vel_pos_exp = apply_rotation_to_vector(R, exp_local)  # LOCAL -> GLOBAL
        vel_pos_exp = torch.where(
            mask_generate[..., None], vel_pos_exp, torch.zeros_like(vel_pos_exp)
        )

        # Rotation MDN (3-vector to be interpreted by quaternion_1ijk_to_rotation_matrix)
        rot_raw = self.rot_head(in_feat)
        mu_r, log_sigma_r, logit_pi_r, log_sigma_r_shared = self._unpack_mdn_iso(
            rot_raw, self.K_rot, 3, MEAN_ROT_CLAMP
        )
        pi_r = F.softmax(logit_pi_r, dim=-1)[..., None]
        u_exp = (pi_r * mu_r).sum(dim=-2)  # (N,L,3)
        u_exp = torch.where(mask_generate[..., None], u_exp, torch.zeros_like(u_exp))

        return (
            c0_pred_logits,
            vel_pos_exp,
            u_exp,
            (mu_p, log_sigma_p, logit_pi_p, log_sigma_p_shared),
            (mu_r, log_sigma_r, logit_pi_r, log_sigma_r_shared),
        )


# FullDPM (training + sampling)
class FullDPM(nn.Module):

    def __init__(
        self,
        res_feat_dim,
        pair_feat_dim,
        num_steps,
        eps_net_opt={},
        trans_rot_opt={},
        trans_pos_opt={},
        trans_seq_opt={},
        position_mean=[0.0, 0.0, 0.0],
        position_scale=[10.0],
    ):
        super().__init__()
        self.pred_net = EpsilonNet(res_feat_dim, pair_feat_dim, **eps_net_opt)
        self.num_steps = int(num_steps)
        self.trans_seq = AminoacidCategoricalTransition(num_steps, **trans_seq_opt)
        self.trans_pos = PositionTransition(num_steps, **trans_pos_opt)
        self.trans_rot = RotationTransition(num_steps, **trans_rot_opt)

        self.register_buffer(
            "position_mean", torch.FloatTensor(position_mean).view(1, 1, -1)
        )
        self.register_buffer(
            "position_scale", torch.FloatTensor(position_scale).view(1, 1, -1)
        )
        self.register_buffer(
            "_dummy",
            torch.empty(
                [
                    0,
                ],
                dtype=torch.float32,
            ),
        )

    def _normalize_position(self, p):
        return (p - self.position_mean) / self.position_scale

    def _unnormalize_position(self, p_norm):
        return p_norm * self.position_scale + self.position_mean

    def forward(
        self,
        v_0,
        p_0,
        s_0,
        res_feat,
        pair_feat,
        mask_generate,
        mask_res,
        denoise_structure=True,
        denoise_sequence=True,
        t=None,
    ):
        """
        Training with proper MDN NLL losses.
        - Rotation: MDN NLL using quaternion + cosine distance
        - Position: MDN NLL in LOCAL frame
        - Sequence: posterior-KL (unchanged)
        """
        N, L = res_feat.shape[:2]
        device = self._dummy.device

        # discrete t in [1..T], and form β (cosine schedule) → normalized [0,1]
        if t is None:
            t_idx = torch.randint(1, self.num_steps + 1, (N,), device=device)
        else:
            t_idx = t.clamp(min=1, max=self.num_steps).to(
                device=device, dtype=torch.long
            )
        beta_cont = (t_idx.float() / float(self.num_steps)).clamp(0.0, 1.0)

        # normalize positions
        p_0n = self._normalize_position(p_0)

        # Add noise / targets
        if denoise_structure:
            v_noisy, _ = self.trans_rot.add_noise(
                v_0, mask_generate, t_idx
            )  # angular forward under RotationTransition
            p_noisy, eps_p = self.trans_pos.add_noise(p_0n, mask_generate, t_idx)
        else:
            v_noisy, p_noisy = v_0, p_0n
            eps_p = torch.zeros_like(p_noisy)

        if denoise_sequence:
            _, s_noisy = self.trans_seq.add_noise(s_0, mask_generate, t_idx)
        else:
            s_noisy = s_0

        R_t = so3vec_to_rotation(v_noisy)
        R_0 = so3vec_to_rotation(v_0)

        # network forward (sequence indices are integers)
        c0_pred_logits, vel_pos_exp, u_exp, mdn_pos, mdn_rot = self.pred_net(
            R_t,
            p_noisy,
            s_noisy,
            beta_cont,
            res_feat,
            pair_feat,
            mask_generate,
            mask_res,
        )
        mu_p, log_sigma_p, logit_pi_p, _ = mdn_pos  # (N,L,K,3), (N,L,K,1), (N,L,K)
        mu_r, log_sigma_r, logit_pi_r, _ = mdn_rot  # (N,L,K,3), (N,L,K,1), (N,L,K)

        # Position MDN NLL in LOCAL frame
        R_tT = R_t.transpose(-1, -2)
        eps_local = apply_rotation_to_vector(R_tT, eps_p)  # (N,L,3)
        loss_pos_full = mdn_nll_iso_loss(
            mu_p, sanitize_(eps_local, MEAN_POS_CLAMP), log_sigma_p, logit_pi_p
        )
        loss_pos = (loss_pos_full * mask_generate.float()).sum() / (
            mask_generate.sum().float() + 1e-8
        )

        # Rotation MDN NLL using quaternion + cosine distance
        loss_rot_full = rotation_mdn_nll_loss(mu_r, R_0, R_t, log_sigma_r, logit_pi_r)
        loss_rot = (loss_rot_full * mask_generate.float()).sum() / (
            mask_generate.sum().float() + 1e-8
        )

        # Sequence KL (categorical DPM)
        s_noisy_oh = clampped_one_hot(s_noisy, 20).float()
        s0_oh = clampped_one_hot(s_0, 20).float()
        post_true = self.trans_seq.posterior(s_noisy_oh, s0_oh, t_idx)  # q(x0|x_t)
        c0_pred = F.softmax(c0_pred_logits, dim=-1)
        log_post_pred = torch.log(
            self.trans_seq.posterior(s_noisy_oh, c0_pred, t_idx) + 1e-12
        )
        kldiv = F.kl_div(
            input=log_post_pred, target=post_true, reduction="none", log_target=False
        ).sum(dim=-1)
        loss_seq = (kldiv * mask_generate.float()).sum() / (
            mask_generate.sum().float() + 1e-8
        )

        return {"seq": loss_seq, "pos": loss_pos, "rot": loss_rot}

    @torch.no_grad()
    def sample(
        self,
        R,
        p,
        s_labels,
        res_feat,
        pair_feat,
        mask_generate,
        mask_res,
        mask_anchor=None,
        sample_structure: bool = True,
        sample_sequence: bool = True,
        pbar: bool = False,
        mixture_mode: str = "global",  # "expectation", "map", "stochastic", or "global"
        tau: float = 1.0,  # temperature for mixture weights
    ):
        """
        Enhanced sampler with mixture modes:
        - "expectation": weighted average over components (smooth, RMSD-friendly)
        - "map": per-residue argmax component (deterministic, sharp)
        - "stochastic": per-residue component sampling (explores modes; can switch across residues)
        - "global": sample ONE component per structure by aggregating logits across generated residues
                    (reduces per-residue mode switching / patchwork artifacts)
        """
        N, L = p.shape[:2]
        device = self._dummy.device

        def _sample_global_k(
            logit_pi: torch.Tensor, mask: torch.Tensor, tau_: float
        ) -> torch.Tensor:
            """
            Args:
                logit_pi: (N, L, K)
                mask:     (N, L) bool
                tau_:     float
            Returns:
                k_global: (N,) long
            """
            mask_f = mask.float()
            denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)  # (N,1)
            logits_global = (logit_pi * mask_f[..., None]).sum(dim=1) / denom  # (N,K)
            pi_global = F.softmax(logits_global / max(tau_, 1e-6), dim=-1)  # (N,K)
            k_global = torch.multinomial(pi_global, num_samples=1).squeeze(-1)  # (N,)
            return k_global

        # normalize positions, coerce R to rotation matrices
        p_norm = self._normalize_position(p)
        if R.ndim == 3 and R.shape[-1] == 3:
            R = so3vec_to_rotation(R)
        elif not (R.ndim == 4 and R.shape[-2:] == (3, 3)):
            raise ValueError(
                f"R must be (N,L,3)[so3vec] or (N,L,3,3)[R], got {tuple(R.shape)}"
            )

        # init generated sites: random rotations where we generate
        v_rand = random_uniform_so3([N, L], device=R.device)
        R_init = so3vec_to_rotation(v_rand)
        R_t = torch.where(mask_generate[..., None, None], R_init, R)
        v_t = rotation_to_so3vec(R_t)

        # positions init
        aa_mask = p_norm.norm(dim=-1) != 0
        context_mask = torch.logical_and(aa_mask, ~mask_generate)
        denom = context_mask.sum(dim=1).clamp(min=1)[:, None]
        p_avg = (p_norm * context_mask[:, :, None]).sum(dim=1) / denom
        p_t = torch.where(
            mask_generate[..., None],
            torch.randn_like(p_norm) + p_avg[:, None, :],
            p_norm,
        )

        # sequence init: random ints on generated sites
        if sample_sequence:
            s_rand = torch.randint_like(s_labels, low=0, high=20)
            c_t = torch.where(mask_generate, s_rand, s_labels)  # (N,L) int
        else:
            c_t = s_labels

        traj = {self.num_steps: (v_t, self._unnormalize_position(p_t), c_t)}
        iterator = (
            tqdm(range(self.num_steps, 0, -1), total=self.num_steps, desc="Sampling")
            if pbar
            else range(self.num_steps, 0, -1)
        )

        # reverse diffusion
        for t_idx in iterator:
            beta_cont = torch.full(
                (N,), float(t_idx) / float(self.num_steps), device=device
            )
            t_long = torch.full((N,), t_idx, dtype=torch.long, device=device)

            # forward pass
            c0_logits, vel_pos_exp, u_exp, mdn_pos, mdn_rot = self.pred_net(
                R_t, p_t, c_t, beta_cont, res_feat, pair_feat, mask_generate, mask_res
            )
            mu_p, log_sigma_p, logit_pi_p, _ = mdn_pos  # (N,L,K,3), (N,L,K,1), (N,L,K)
            mu_r, log_sigma_r, logit_pi_r, _ = mdn_rot  # (N,L,K,3), (N,L,K,1), (N,L,K)

            # sequence reverse
            if sample_sequence:
                c0_pred = F.softmax(c0_logits, dim=-1)
                _, c_prev = self.trans_seq.denoise(c_t, c0_pred, mask_generate, t_long)
            else:
                c_prev = c_t

            # positions reverse
            if sample_structure:
                if mixture_mode == "expectation":
                    if tau != 1.0:
                        pi_p = F.softmax(logit_pi_p / max(tau, 1e-6), dim=-1)[
                            ..., None
                        ]  # (N,L,K,1)
                        eps_local = (pi_p * mu_p).sum(dim=-2)  # (N,L,3)
                        eps_hat_global = apply_rotation_to_vector(R_t, eps_local)
                    else:
                        eps_hat_global = vel_pos_exp

                elif mixture_mode == "map":
                    k_pos = F.softmax(logit_pi_p, dim=-1).argmax(dim=-1)  # (N,L)
                    k_idx = k_pos.unsqueeze(-1).unsqueeze(-1).expand(N, L, 1, 3)
                    mu_loc = torch.gather(mu_p, 2, k_idx).squeeze(2)  # (N,L,3)
                    eps_hat_global = apply_rotation_to_vector(R_t, mu_loc)

                elif mixture_mode == "global":
                    k_pos_g = _sample_global_k(
                        logit_pi_p, mask_generate, tau_=tau
                    )  # (N,)
                    k_idx = k_pos_g[:, None, None, None].expand(N, L, 1, 3)
                    mu_loc = torch.gather(mu_p, 2, k_idx).squeeze(2)  # (N,L,3)
                    eps_hat_global = apply_rotation_to_vector(R_t, mu_loc)

                else:
                    raise ValueError(
                        f"mixture_mode must be 'expectation', 'map', 'stochastic', or 'global', got {mixture_mode!r}"
                    )

                eps_hat_global = torch.where(
                    mask_generate[..., None],
                    eps_hat_global,
                    torch.zeros_like(eps_hat_global),
                )
                p_prev = self.trans_pos.denoise(
                    p_t, eps_hat_global, mask_generate, t_long
                )
            else:
                p_prev = p_t

            # rotations reverse
            if sample_structure:
                if mixture_mode == "expectation":
                    if tau != 1.0:
                        pi_r = F.softmax(logit_pi_r / max(tau, 1e-6), dim=-1)[
                            ..., None
                        ]  # (N,L,K,1)
                        u_hat = (pi_r * mu_r).sum(dim=-2)  # (N,L,3)
                    else:
                        u_hat = u_exp

                elif mixture_mode == "map":
                    k_rot = F.softmax(logit_pi_r, dim=-1).argmax(dim=-1)  # (N,L)
                    k_idx_r = k_rot.unsqueeze(-1).unsqueeze(-1).expand(N, L, 1, 3)
                    u_hat = torch.gather(mu_r, 2, k_idx_r).squeeze(2)  # (N,L,3)

                elif mixture_mode == "global":
                    k_rot_g = _sample_global_k(
                        logit_pi_r, mask_generate, tau_=tau
                    )  # (N,)
                    k_idx_r = k_rot_g[:, None, None, None].expand(N, L, 1, 3)
                    u_hat = torch.gather(mu_r, 2, k_idx_r).squeeze(2)  # (N,L,3)

                else:
                    raise ValueError(
                        f"mixture_mode must be 'expectation', 'map', 'stochastic', or 'global', got {mixture_mode!r}"
                    )

                u_hat = torch.where(
                    mask_generate[..., None],
                    u_hat,
                    torch.zeros_like(u_hat),
                )

                U = quaternion_1ijk_to_rotation_matrix(u_hat)  # (N,L,3,3)
                R_next = R_t @ U
                v_next = rotation_to_so3vec(R_next)
                v_prev = self.trans_rot.denoise(v_t, v_next, mask_generate, t_long)
                R_prev = so3vec_to_rotation(v_prev)
            else:
                v_prev, R_prev = v_t, R_t

            # advance
            traj[t_idx - 1] = (v_prev, self._unnormalize_position(p_prev), c_prev)
            v_t, R_t, p_t, c_t = v_prev, R_prev, p_prev, c_prev

        return traj
