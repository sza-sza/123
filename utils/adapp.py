
import numpy as np
import torch
import math

class AdaptiveStabilityController:
    def __init__(self, base_sigma=0.5, sigma_min=0.005, sigma_max=1,
                 seed=42, verbose=False, device=None, rounds=100):
        self.base_sigma = float(base_sigma)
        self.prev_sigma = float(base_sigma)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

        self.prev_var = None
        self.prev_loss = None
        self.prev_strength = None

        self.verbose = verbose
        self.seed = int(seed)
        self.rng_global = np.random.RandomState(self.seed)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.total_rounds = int(rounds)

        # smoothing / momentum
        self.ema_alpha = 0.6
        self.sigma_momentum = 0.85
        self.smooth_beta = 0.95

        # dual-factor hyperparams
        self.alpha_var = 0.3
        self.beta_conv = 0.15
        self.k_cos = 0.8
        self.k_var = 0.5
        self.k_conv = 0.01

        # gamma params
        self.gamma_base = 1.0
        self.gamma_k = 0.06
        self.gamma_alpha = 0.25
        self.gamma_beta = 1.0

        # byzantine detection params
        self.suspicious_threshold = 1.8
        self.detect_ratio = 0.2
        self.noise_ratio_max = 0.05
        self.delta_ratio_thresh = 1e-5

        # diagnostics
        self.last_gamma = float("nan")
        self.last_percent_change = 0.0
        self.last_rel_update = 0.0
        self.last_delta_l2 = 0.0
        self.last_prev_l2 = 0.0
        self.last_max_layer_rel = 0.0
        self.last_suspicious_count = 0
        self.prev_updates_mean = None

        self.per_client_prev_var = {}
        self.per_client_prev_vec = {}
        self.per_client_prev_sigma = {}

    # ---------- directional gamma (safe) ----------
    def directional_gamma(self, round_num, strength):
        try:
            rn = float(round_num)
        except Exception:
            rn = 1.0
        if strength is None or np.isnan(strength) or np.isinf(strength):
            strength = float(self.prev_strength if self.prev_strength is not None else 1.0)
        gamma = self.gamma_base * math.exp(-self.gamma_k * rn)
        denom = 1.0
        try:
            denom = 1.0 + float(self.gamma_alpha) * (float(strength) ** float(self.gamma_beta))
        except Exception:
            denom = 1.0
        if denom <= 0 or np.isnan(denom) or np.isinf(denom):
            denom = 1.0
        gamma = gamma / denom
        if not np.isfinite(gamma):
            gamma = float(self.last_gamma if np.isfinite(self.last_gamma) else self.gamma_base)

        self.last_gamma = float(gamma)
        return float(gamma)

    # ---------- adaptive noise by var  ----------
    def adaptive_noise_by_var(self, var_now, var_prev, loss_now, loss_prev):
        eps = 1e-8
        var_factor = (var_now / (var_prev + eps)) - 1.0
        conv_factor = abs(loss_now - loss_prev) / (abs(loss_prev) + eps)
        sigma_dynamic = self.base_sigma * (1 + self.alpha_var * var_factor) * (1 + self.beta_conv * conv_factor)
        sigma_dynamic = float(np.clip(sigma_dynamic, self.sigma_min, self.sigma_max))
        sigma = float(self.sigma_momentum * self.prev_sigma + (1.0 - self.sigma_momentum) * sigma_dynamic)
        return sigma

    # ---------- mean pairwise cos ----------
    def _mean_pairwise_cos(self, client_updates):
        if not client_updates or len(client_updates) < 2:
            return 1.0
        flat = [u.view(-1).cpu().float() for u in client_updates]
        n = len(flat)
        cos_vals = []
        for i in range(n):
            ai = flat[i]
            ai_norm = torch.norm(ai) + 1e-12
            for j in range(i+1, n):
                aj = flat[j]
                aj_norm = torch.norm(aj) + 1e-12
                cos_vals.append(float(torch.dot(ai, aj) / (ai_norm * aj_norm)))
        return float(np.mean(cos_vals)) if cos_vals else 1.0

    # ---------- client sigma interface ----------
    def client_sigma_for_update(self,
                                client_update,  # numpy 1D vector (masked vals)
                                var_now,
                                loss_now,
                                loss_prev,
                                client_weight,
                                global_sigma,
                                client_id=None):

        # safe casts
        try:
            var_now = float(var_now)
        except:
            var_now = 0.0
        prev_var = 0.0
        prev_update_vec = None
        if not hasattr(self, "per_client_prev_var"):
            self.per_client_prev_var = {}
            self.per_client_prev_vec = {}

        if client_id is not None:
            prev_var = float(self.per_client_prev_var.get(client_id, var_now))
            prev_update_vec = self.per_client_prev_vec.get(client_id, None)

        # compute shift (relative L2 change) robustly
        shift = 0.0
        try:
            if prev_update_vec is not None and prev_update_vec.size == client_update.size:
                num = np.linalg.norm(client_update - prev_update_vec)
                den = (np.linalg.norm(prev_update_vec) + 1e-12)
                shift = float(num / den)
        except Exception:
            shift = 0.0

        # ----- DEBUG: print input shapes/norms -----
        try:
            u_vec = np.asarray(client_update, dtype=np.float32).reshape(-1) if client_update is not None else None
            lg = self.last_global_dir if hasattr(self, "last_global_dir") else None
            print(f"[DBG-client_sigma-IN] cid={client_id} u_vec_shape={(None if u_vec is None else u_vec.shape)} "
                  f"u_vec_norm={(None if u_vec is None else np.linalg.norm(u_vec)):.6e} "
                  f"has_last_global_dir={(lg is not None)} "
                  f"last_global_dir_norm={(None if lg is None else np.linalg.norm(lg)):.6e}")
        except Exception as e:
            print("[DBG-client_sigma-IN] error printing inputs:", e)

        # compute mean_cos_local if available (optional)
        mean_cos_local = 1.0
        if hasattr(self, "last_global_dir") and self.last_global_dir is not None and client_update is not None:
            try:
                gdir = np.asarray(self.last_global_dir, dtype=np.float32).reshape(-1)
                u_vec = np.asarray(client_update, dtype=np.float32).reshape(-1)
                if gdir.size == u_vec.size:
                    # --- FIX: define u_norm properly ---
                    g_norm = np.linalg.norm(gdir) + 1e-12
                    u_norm = np.linalg.norm(u_vec) + 1e-12
                    mean_cos_local = float(np.dot(u_vec, gdir) / (u_norm * g_norm))
                    # DEBUG: detailed diagnostics
                    try:
                        dot = float(np.dot(u_vec, gdir))
                        diff_norm = float(np.linalg.norm(u_vec - gdir))
                        max_abs_diff = float(np.max(np.abs(u_vec - gdir)))
                        print(f"[DBG-dircos] cid={client_id} dot={dot:.8e} u_norm={u_norm:.8e} g_norm={g_norm:.8e} "
                              f"dir_cos_raw={mean_cos_local:.8f} diff_norm={diff_norm:.8e} max_abs_diff={max_abs_diff:.8e}")
                    except Exception:
                        pass
                else:
                    # shape mismatch -> log and be conservative
                    print(
                        f"[WARN-client_sigma] cid={client_id} shape mismatch: last_global_dir.size={gdir.size} u_vec.size={u_vec.size}")
                    mean_cos_local = 1.0
            except Exception as e:
                print(f"[WARN-client_sigma] cid={client_id} mean_cos_local compute failed: {e}")
                mean_cos_local = 1.0
        else:
            # no last_global_dir -> will use conservative 1.0
            if not hasattr(self, "last_global_dir") or self.last_global_dir is None:
                print(f"[DBG-client_sigma] cid={client_id} no last_global_dir available -> mean_cos_local=1.0")

        mean_cos_local = float(np.clip(mean_cos_local, -1.0, 1.0))

        # parameters (tuneable)
        sigma_base = float(global_sigma if global_sigma is not None else self.base_sigma)
        sigma_min = getattr(self, "sigma_min", sigma_base * 0.01)
        sigma_max = getattr(self, "sigma_max", sigma_base * 4.0)
        momentum = getattr(self, "sigma_momentum", 0.85)

        k_var = getattr(self, "k_var", 1.0)
        lambda_shift = getattr(self, "lambda_shift", 1.0)
        shift_clip = getattr(self, "shift_clip", 5.0)
        alpha_shift = getattr(self, "alpha_shift", 0.8)
        k_cos_local = getattr(self, "k_cos_local", 0.5)

        # build safe components
        var_ratio = np.clip(var_now / (prev_var + 1e-6), 0.1, 10.0)
        shift_clipped = float(np.clip(shift, 0.0, shift_clip))
        shift_factor = float(np.exp(-lambda_shift * shift_clipped))
        direction_factor = float(np.exp(-k_cos_local * (1.0 - np.clip(mean_cos_local, -1.0, 1.0))))

        sigma_candidate = sigma_base * (var_ratio ** k_var) * (1.0 + alpha_shift * (1.0 - shift_factor)) * (
                1.0 / (direction_factor + 1e-12))
        sigma_candidate = float(np.clip(sigma_candidate, sigma_min, sigma_max))

        prev_sigma = float(self.per_client_prev_sigma.get(client_id, sigma_base)) if hasattr(self,
                                                                                             "per_client_prev_sigma") else sigma_base
        sigma_new = momentum * prev_sigma + (1.0 - momentum) * sigma_candidate
        sigma_new = float(np.clip(sigma_new, sigma_min, sigma_max))

        # store history
        if client_id is not None:
            self.per_client_prev_var[client_id] = var_now
            try:
                self.per_client_prev_vec[client_id] = client_update.copy()
            except Exception:
                self.per_client_prev_vec[client_id] = np.array(client_update, dtype=np.float32)
            if not hasattr(self, "per_client_prev_sigma"):
                self.per_client_prev_sigma = {}
            self.per_client_prev_sigma[client_id] = sigma_new

        # safety
        if not np.isfinite(sigma_new) or sigma_new <= 0:
            sigma_new = sigma_base
        if self.verbose:
            print(f"  raw_var_ratio={var_ratio:.3f} prev_var={prev_var:.6e}")

        if self.verbose:
            print(f"[Controller.client_sigma] cid={client_id} var_ratio={var_ratio:.3f} shift={shift:.3f} "
                  f"dir_cos={mean_cos_local:.3f} sigma_cand={sigma_candidate:.4f} sigma_new={sigma_new:.4f}")

        return float(sigma_new)

    # ---------- multi-factor suspicious detection ----------
    def detect_suspicious_clients_multifactor(self,
                                              client_updates,
                                              client_ids=None,
                                              client_weights=None,
                                              use_history=True,
                                              hard_neg_th=-0.7,
                                              w_cos=0.5,
                                              w_norm=0.3,
                                              w_drift=0.2,
                                              score_q=0.80):

        K = len(client_updates)
        if K == 0:
            return [], np.ones(0), {}, {"suspicious_ratio": 0.0}

        # ----- stream flatten to avoid large memory spike -----
        flats = []
        norms = np.empty(K, dtype=np.float32)
        for i, u in enumerate(client_updates):
            f = u.view(-1).cpu().float()
            flats.append(f)
            norms[i] = float(torch.norm(f).item())

        # mean direction (streaming is possible; keep simple here)
        # caution: avoid torch.stack for very large K*dim if you hit memory
        mean_vec = torch.stack(flats, dim=0).mean(dim=0)
        mean_norm = float(torch.norm(mean_vec).item()) + 1e-12
        mean_dir = (mean_vec / mean_norm).cpu().float()

        # save metadata
        try:
            mg = mean_dir.view(-1).cpu().numpy().astype(np.float32)
            self.last_global_dir = mg
            self.last_global_dir_shape = mg.size
            self.last_global_dir_nnz = int(np.count_nonzero(mg))
            self.last_global_dir_sample_firstk = mg[:8].tolist()
        except Exception:
            self.last_global_dir = None

        # ----- compute cos values -----
        cos_vals = np.zeros(K, dtype=np.float32)
        for i, f in enumerate(flats):
            fn = torch.norm(f).item()
            cos_vals[i] = float(torch.dot(f, mean_dir).item() / (fn + 1e-12)) if fn > 0 else 0.0

        # ----- norm z-score (+ positive part) -----
        mean_norm_v = float(np.mean(norms) + 1e-12)
        std_norm = float(np.std(norms) + 1e-12)
        norm_z = (norms - mean_norm_v) / (std_norm + 1e-12)
        norm_z_pos = np.maximum(0.0, norm_z)

        # ----- drift (w.r.t history) -----
        drift_vals = np.zeros(K, dtype=np.float32)
        try:
            if use_history and self.prev_updates_mean is not None:
                prev_mean = self.prev_updates_mean.view(-1).cpu().float()
                prev_norm = float(torch.norm(prev_mean).item()) + 1e-12
                for i, f in enumerate(flats):
                    drift_vals[i] = float(torch.norm(f - prev_mean).item()) / prev_norm
        except Exception:
            pass

        # ----- normalize each metric via robust min-max (0..1) ----- #
        def robust_minmax(x):
            mn, mx = np.min(x), np.max(x)
            if mx - mn < 1e-12:
                return np.zeros_like(x)
            return (x - mn) / (mx - mn)

        # cos abnormality: lower cos -> more suspicious => use 1 - cos
        cos_score = robust_minmax(1.0 - cos_vals)
        # norm_score: compress positive z (large norms suspicious)
        norm_score_raw = np.tanh(norm_z_pos / 2.0)
        norm_score = robust_minmax(norm_score_raw)
        # drift_score: relative large drift suspicious
        drift_score = robust_minmax(drift_vals)

        # ----- composite score ----- #
        w_total = float(w_cos + w_norm + w_drift)
        comp = (w_cos * cos_score + w_norm * norm_score + w_drift * drift_score) / (w_total + 1e-12)

        # ----- top-quantile thresholding + hard-neg sign flips ----- #
        thr = np.quantile(comp, score_q)
        suspicious_idxs = set(np.where(comp >= thr)[0].tolist())
        # add strong sign-flips regardless
        hard_idxs = set([i for i, c in enumerate(cos_vals) if c < hard_neg_th])
        suspicious_idxs.update(hard_idxs)
        suspicious_idxs = sorted(list(suspicious_idxs))

        # ----- soft_weights: trust score (1 - comp) clipped ----- #
        soft_weights = np.clip(1.0 - comp, 0.0, 1.0)

        # ----- diagnostics ----- #
        diagnostics = {}
        for i in range(K):
            diagnostics[i] = {
                "cos": float(cos_vals[i]),
                "norm": float(norms[i]),
                "norm_z": float(norm_z[i]),
                "norm_score": float(norm_score[i]),
                "drift": float(drift_vals[i]),
                "drift_score": float(drift_score[i]),
                "comp": float(comp[i]),
                "soft_weight": float(soft_weights[i])
            }

        # ----- update history EMA -----
        try:
            current_mean = mean_vec.cpu()
            if not hasattr(self, "prev_updates_mean") or self.prev_updates_mean is None:
                self.prev_updates_mean = current_mean.clone()
            else:
                sm_alpha = 0.85
                self.prev_updates_mean = (sm_alpha * self.prev_updates_mean + (1.0 - sm_alpha) * current_mean).clone()
        except Exception:
            pass

        self.last_suspicious_count = int(len(suspicious_idxs))
        return suspicious_idxs, soft_weights, diagnostics

    def compute_strength(self, client_updates):
        """
        计算客户端更新的一致性强度。用于动态调整 γ 和 σ。
        """
        if client_updates is None or len(client_updates) < 2:
            return 1.0

        try:
            cos_vals = []
            for i in range(len(client_updates)):
                ui = client_updates[i].flatten()
                ni = torch.norm(ui) + 1e-12
                ui = ui / ni

                for j in range(i + 1, len(client_updates)):
                    uj = client_updates[j].flatten()
                    nj = torch.norm(uj) + 1e-12
                    uj = uj / nj

                    cos_vals.append(torch.dot(ui, uj).item())

            if len(cos_vals) == 0:
                return 1.0

            mean_cos = float(np.mean(cos_vals))
            strength = float(1.0 - mean_cos)
            strength = np.clip(strength, 0.0, 2.0)

            return strength

        except Exception:
            return 1.0
    # ---------- update (robust, uses client_updates; returns sigma, gamma) ----------
    def update(self, var_now, loss_now, strength, model, model_prev, rng=None,
               round_num=1, client_fractions=None, client_updates=None,
               suspicious_idxs=None, soft_weights=None,
           median_client_sigma=None):

        # ================================
        # 1. Safe scalar wrapper
        # ================================
        def safe_scalar(x, default=0.0):
            try:
                x = float(x)
                if np.isnan(x) or np.isinf(x):
                    return default
                return x
            except:
                return default

        # sanitize
        var_now = safe_scalar(var_now, 0.0)
        loss_now = safe_scalar(loss_now, 0.0)
        strength = safe_scalar(strength, 1.0)

        prev_var = safe_scalar(self.prev_var, var_now)
        prev_loss = safe_scalar(self.prev_loss, loss_now)
        prev_sigma = safe_scalar(self.prev_sigma, self.base_sigma)

        # ================================
        # 2. Norm change (ΔL2, percent)
        # ================================
        device = next(model.parameters()).device

        try:
            delta_l2_sq = torch.tensor(0.0, device=device)
            prev_l2_sq = torch.tensor(0.0, device=device)

            for p, q in zip(model.parameters(), model_prev.parameters()):
                delta_l2_sq += torch.sum((p - q) ** 2)
                prev_l2_sq += torch.sum(q ** 2)

            delta_l2 = safe_scalar(torch.sqrt(delta_l2_sq).item(), 0.0)
            prev_l2 = safe_scalar(torch.sqrt(prev_l2_sq).item(), 1.0)

        except:
            delta_l2 = 0.0
            prev_l2 = 1.0

        rel_update = delta_l2 / (math.sqrt(sum(p.numel() for p in model.parameters())) + 1e-12)
        percent_change = 100.0 * delta_l2 / (prev_l2 + 1e-12)

        # ================================
        # 3. mean_cos
        # ================================
        mean_cos = 1.0
        if client_updates and len(client_updates) >= 2:
            try:
                mean_cos = safe_scalar(self._mean_pairwise_cos(client_updates), 1.0)
            except:
                mean_cos = 1.0

        # ================================
        # 4. baseline gamma
        # ================================
        gamma_base = self.directional_gamma(round_num, strength)
        gamma_base = safe_scalar(gamma_base, 1.0)

        # ================================
        # 5. integrate detect outputs
        # ================================
        suspicious_idxs = suspicious_idxs or []
        soft_weights = np.array(soft_weights, dtype=np.float32) if soft_weights is not None else None

        K = len(soft_weights) if soft_weights is not None else len(client_updates or [1])

        susp_ratio = len(suspicious_idxs) / max(1, K)
        mean_soft = float(np.mean(soft_weights)) if soft_weights is not None else 1.0

        # anomaly strength
        anom_strength = 0.6 * susp_ratio + 0.4 * (1 - mean_soft)
        anom_strength = float(np.clip(anom_strength, 0.0, 1.0))

        # ================================
        # 6. gamma adjustment
        # ================================
        gamma = gamma_base * (1.0 - 0.7 * anom_strength)
        gamma = float(np.clip(gamma, 0.02, 0.98))

        # ================================
        # 7. sigma_candidate (base formula)
        # ================================
        var_ratio = safe_scalar(var_now / (prev_var + 1e-6), 1.0)
        var_ratio = np.clip(var_ratio, 0.1, 10.0)

        loss_delta = abs(loss_now - prev_loss) / (abs(prev_loss) + 1e-6)
        loss_delta = np.clip(loss_delta, 0.0, 10.0)

        direction_factor = 1.0 + self.k_cos * (1.0 - mean_cos)
        direction_factor = np.clip(direction_factor, 0.1, 3.0)

        variance_factor = 1.0 + self.k_var * abs(var_now - prev_var)
        variance_factor = np.clip(variance_factor, 0.1, 3.0)

        convergence_factor = 1.0 + self.k_conv * percent_change
        convergence_factor = np.clip(convergence_factor, 0.1, 3.0)

        sigma_candidate = (
                self.base_sigma *
                direction_factor *
                variance_factor *
                convergence_factor *
                var_ratio *
                (1.0 + loss_delta)
        )
        sigma_candidate = safe_scalar(sigma_candidate, self.base_sigma)

        # ================================
        # 8. anomaly correction for σ
        #    —— 重点改动：不再被 sigma_max=0.2 卡死
        # ================================
        # 动态上界：越可疑，上界越高
        cap_low = self.base_sigma * 2.0
        cap_mid = self.base_sigma * 4.0
        cap_high = self.base_sigma * 8.0

        if susp_ratio < 0.1:
            sigma_cap = cap_low
        elif susp_ratio < 0.4:
            sigma_cap = cap_mid
        else:
            sigma_cap = cap_high

        # 额外 anomaly multiplier
        sigma_candidate *= (1.0 + 2.5 * anom_strength)

        # 裁剪到动态 cap，而不是固定 sigma_max
        sigma_candidate = float(np.clip(sigma_candidate, self.sigma_min, sigma_cap))

        # ================================
        # 9. 双向动量 smoothing
        # ================================
        if sigma_candidate > prev_sigma:
            mom = 0.45  # 升噪快
        else:
            mom = 0.85  # 降噪慢

        sigma_new = mom * prev_sigma + (1 - mom) * sigma_candidate

        if rng:
            sigma_new *= float(rng.uniform(0.995, 1.005))

        # 最终绝对安全边界（可选）但不要用死的 0.2
        hard_cap = self.base_sigma * 12.0
        sigma_new = float(np.clip(sigma_new, self.sigma_min, hard_cap))

        # ================================
        # 10. save states
        # ================================
        self.prev_var = var_now
        self.prev_loss = loss_now
        self.prev_strength = strength

        self.last_gamma = gamma
        self.last_rel_update = rel_update
        self.last_percent_change = percent_change
        self.last_delta_l2 = delta_l2
        self.last_prev_l2 = prev_l2
        if median_client_sigma is not None:
            # combine server sigma update + client sigma consensus
            self.prev_sigma = float(
                0.7 * sigma_new + 0.3 * median_client_sigma
            )
        else:
            self.prev_sigma = float(sigma_new)

        if self.verbose:
            print(f"[Controller.update] R{round_num} gamma={gamma:.4f}, mean_cos={mean_cos:.4f}, "
                  f"anom={anom_strength:.3f}, susp={susp_ratio:.3f}, mean_soft={mean_soft:.3f}, "
                  f"sigma={sigma_new:.4f}, cand={sigma_candidate:.4f}, base={self.base_sigma:.4f}")

        return float(gamma), float(sigma_new)


