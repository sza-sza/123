

from typing import Callable, Dict, Optional, Tuple, List
import numpy as np
import math

# ---------------- Helper functions ----------------

def _safe_float(x):
    try:
        return float(x)
    except:
        return 0.0

def _gaussian_rdp_no_subsample(alpha: float, sensitivity: float, sigma: float) -> float:
    """
    RDP of Gaussian mechanism with sensitivity `sensitivity` and noise std `sigma`
    for Renyi order alpha, no subsampling.
    Formula: RDP(α) = α * sensitivity^2 / (2 * sigma^2)
    (valid for α >= 1)
    """
    alpha = float(alpha)
    s = float(sensitivity)
    sigma = max(float(sigma), 1e-12)
    return (alpha * (s ** 2)) / (2.0 * (sigma ** 2))

def _gaussian_rdp_subsampled_poisson_approx(alpha: float, q: float, sensitivity: float, sigma: float) -> float:
    """
    Conservative approximation for RDP of subsampled (Poisson) Gaussian mechanism.
    Approx: rdp ≈ q^2 * α * sensitivity^2 / (2 sigma^2)

    Notes:
      - This is a simplification useful for experiments.
      - For precise bounds, use specialized implementations (e.g., tensorflow_privacy.rdp_accountant).
    """
    return _gaussian_rdp_no_subsample(alpha, sensitivity, sigma) * (q ** 2)

# ---------------- Client-side adaptive DP function ----------------
'''
def apply_dp_dual_adaptive_secure(
        vec,
        clip_norm,
        round_num,
        client_weight=None,
        base_sigma=None,
        prev_sigma=None,
        prev_update_vec=None,
        last_global_dir=None,
        suspicious_flag=False,
        rng=None,
        noise_type="gaussian"):

    try:
        # sanitize inputs
        v = _np.asarray(vec, dtype=_np.float32).reshape(-1)
        clip_norm = float(clip_norm) if clip_norm is not None else ( _np.linalg.norm(v) + 1e-12 )
        base_sigma = float(base_sigma) if base_sigma is not None else 0.0
        prev_sigma = float(prev_sigma) if prev_sigma is not None else base_sigma
        if rng is None:
            rng = _np.random.RandomState((int(round_num) if round_num is not None else 0) + 123456)

        # normalize client_weight
        try:
            if hasattr(client_weight, "dataset"):
                client_n = float(len(client_weight.dataset))
            else:
                client_n = float(client_weight) if client_weight is not None else 1.0
        except Exception:
            client_n = float(client_weight) if client_weight is not None else 1.0

        # choose used_sigma (prefer base_sigma passed by caller; fall back to prev)
        used_sigma = float(base_sigma if base_sigma is not None and base_sigma > 0 else prev_sigma)
        if not _np.isfinite(used_sigma) or used_sigma <= 0:
            used_sigma = float(prev_sigma if prev_sigma is not None and prev_sigma > 0 else max(1e-6, base_sigma))

        # L2 clip the whole vector to clip_norm (clip_norm should be computed by caller using sensible coords)
        v_norm = float(_np.linalg.norm(v)) + 1e-12
        if v_norm > clip_norm:
            v_clipped = v * (clip_norm / v_norm)
        else:
            v_clipped = v.copy()

        # noise scale: used_sigma * clip_norm
        noise_scale = float(used_sigma) * float(clip_norm)
        # guard against degenerate scale
        if not _np.isfinite(noise_scale) or noise_scale < 0:
            noise_scale = max(1e-12, float(used_sigma) * (float(clip_norm) + 1e-12))

        if noise_type == "gaussian":
            noise = rng.normal(loc=0.0, scale=noise_scale, size=v_clipped.shape).astype(_np.float32)
        else:
            noise = rng.laplace(loc=0.0, scale=noise_scale, size=v_clipped.shape).astype(_np.float32)

        vals_dp = v_clipped + noise

        # account info (conservative placeholders; integrate accountant later)
        account_info = {
            "used_sigma": float(used_sigma),
            "clip_norm": float(clip_norm),
            "sampling_rate": float(sampling_rate) if sampling_rate is not None else None,
            "client_n": float(client_n),
            "noise_type": str(noise_type)
        }

        vals_dp = _np.nan_to_num(vals_dp, nan=0.0, posinf=1e6, neginf=-1e6)
        return vals_dp, float(used_sigma), account_info

    except Exception as e:
        # fallback: return sanitized original vec with base_sigma
        try:
            fallback = _np.nan_to_num(_np.asarray(vec, dtype=_np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
        except Exception:
            fallback = _np.zeros_like(vec, dtype=_np.float32)
        account_info = {"error": str(e), "used_sigma": float(base_sigma if base_sigma is not None else 0.0)}
        return fallback, float(base_sigma if base_sigma is not None else 0.0), account_info
'''
def apply_dp_dual_adaptive_secure(
    vec,
    clip_norm,
    sigma,
    rng,
    noise_type="gaussian",
    sampling_rate=1.0,
    accountant=None
):
    """
    DP on dense vector, with sqrt(k) normalization to keep noise L2 stable.
    - vec: dense numpy array (float32)
    - clip_norm: L2 clip bound computed from masked elements (top-k)
    - sigma: per-client sigma chosen by controller
    - rng: numpy RandomState
    - noise_type: 'gaussian' or 'laplace'
    - sampling_rate: for privacy accountant (client sampling)
    """

    # 1) sanity convert
    v = np.asarray(vec, dtype=np.float32)
    if not np.all(np.isfinite(v)):
        v = np.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)
    # 2) clipping
    v_L2 = float(np.linalg.norm(v)) + 1e-12
    c = float(clip_norm)

    if v_L2 > c:
        scale = c / v_L2
        v_clipped = v * scale
    else:
        v_clipped = v.copy()


    k = float(v_clipped.size)
    if k < 1:
        k = 1.0

    noise_std = float(sigma) * float(c) / (k ** 0.5)

    if noise_type == "gaussian":
        noise = rng.normal(0.0, noise_std, size=v_clipped.shape).astype(np.float32)
        v_dp = v_clipped + noise
    else:
        noise = rng.laplace(0.0, noise_std, size=v_clipped.shape).astype(np.float32)
        v_dp = v_clipped + noise

    # ------------------------
    # 4) numeric safety
    # ------------------------
    v_dp = np.nan_to_num(v_dp, nan=0.0, posinf=1e6, neginf=-1e6)

    # ------------------------
    # 5) return accountant info
    # ------------------------
    account_info = {
        "sigma": float(sigma),
        "clip_norm": float(c),
        "noise_std": float(noise_std),
        "dimension": int(k)
    }

    return v_dp.astype(np.float32), float(sigma), account_info




# ---------------- Adaptive RDP Accountant ----------------

class AdaptiveRDPAccountant:
    """
    RDP Accountant to aggregate per-client/per-round RDP contributions.

    Usage:
        acct = AdaptiveRDPAccountant(sampling_rate=q, delta=1e-5)
        # for each client in a round, after apply_dp_dual_adaptive_secure:
        acct.accumulate_from_account_info(account_info)
        # after many rounds:
        eps, delta = acct.get_privacy_spent()
    """

    def __init__(self, sampling_rate: float = 1.0, delta: float = 1e-5, orders: Optional[List[float]] = None, mode: str = "poisson_approx"):
        """
        Args:
            sampling_rate: q (for approximate subsampled RDP). Use 1.0 if no subsampling.
            delta: target delta for (ε, δ)
            orders: list of Renyi orders to track
            mode: "no_subsampling" or "poisson_approx"
                  - "no_subsampling": uses exact Gaussian RDP per order
                  - "poisson_approx": uses conservative approx rdp ≈ q^2 * α * s^2 / (2 σ^2)
        """
        if orders is None:
            self.orders = [1.25,2,3,4,8,16,32,64,128,256,512]
        else:
            self.orders = orders
        self.sampling_rate = float(sampling_rate)
        self.delta = float(delta)
        self.mode = mode
        self.rdp_cumulative = {a: 0.0 for a in self.orders}

    def accumulate_from_account_info(self, account_info: Dict):
        """
        account_info: dict returned by apply_dp_dual_adaptive_secure
        It should contain "rdp_grad" and "rdp_report" which are dicts order->rdp
        We simply add them into cumulative RDP.
        """
        rdp_grad = account_info.get("rdp_grad", {})
        rdp_report = account_info.get("rdp_report", {})

        for a in self.orders:
            self.rdp_cumulative[a] += float(rdp_grad.get(a, 0.0)) + float(rdp_report.get(a, 0.0))

    def accumulate_manual(self, sigma: float, sensitivity: float, steps: int = 1):
        """
        Convenience: add RDP for a Gaussian mechanism with noise sigma and sensitivity.
        This uses the accountant's mode to select formula.
        steps multiplies the rdp (e.g., repeated applications).
        """
        for a in self.orders:
            if self.mode == "no_subsampling":
                r = _gaussian_rdp_no_subsample(a, sensitivity, sigma)
            else:
                r = _gaussian_rdp_subsampled_poisson_approx(a, self.sampling_rate, sensitivity, sigma)
            self.rdp_cumulative[a] += steps * float(r)

    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Convert accumulated RDP to (eps, delta).
        Returns (epsilon, delta) where delta is self.delta.

        epsilon = min_a (RDP(a) - log(delta) / (a - 1))
        """
        epsilons = []
        for a in self.orders:
            rdp = float(self.rdp_cumulative.get(a, 0.0))
            eps_a = rdp - math.log(self.delta) / (a - 1.0)
            epsilons.append(eps_a)
        eps = float(min(epsilons))
        return eps, float(self.delta)

    def reset(self):
        for a in self.orders:
            self.rdp_cumulative[a] = 0.0


