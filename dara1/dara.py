import copy
import torch
import numpy as np
import random
import os, math, csv, time
import logging
import json

# local utils (assume these exist in your repo)
from utils.adapp import AdaptiveStabilityController
from utils.training_monitor import log_training_metrics, visualize_training_metrics
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import InterpolationMode
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from models.resnet_gn import CIFARResNet18_GN as CIFARResNet18
from utils.adaptive_dp import apply_dp_dual_adaptive_secure, AdaptiveRDPAccountant
# ---------------- Config ----------------
NUM_CLIENTS = 20
BYZANTINE = 4
ROUNDS = 200
LOCAL_EPOCHS = 3
BATCH_SIZE = 128
LR = 0.02
MOMENTUM = 0.8
WEIGHT_DECAY = 5e-4
Q_LEVELS = 1024
TOPK = 1.0
SIGN_FLIP_ALPHA = 3
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_SIGMA = 0.05
CLIENT_FRACTION = 1
CLIENTS_PER_ROUND = max(1, int(NUM_CLIENTS * CLIENT_FRACTION))
DP_CLIP_NORM = 5.0
OUTDIR = "./fltrustmn"
os.makedirs(OUTDIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
FIXED_DP = False
FIXED_SIGMA = 0.05   # 论文中使用的 σ0

class GaussianPrivacyAccountant:
    def __init__(self, sampling_rate=1.0, delta=1e-5):
        self.sampling_rate = sampling_rate
        self.delta = delta
        self.orders = [1.25,2,3,4,8,16,32,64,128,256,512]
        self.rdp_cumulative = np.zeros(len(self.orders), dtype=np.float64)
    def update(self, sigma, steps=1):
        sigma = max(sigma, 1e-12)
        for i, alpha in enumerate(self.orders):
            rdp = steps * (alpha * self.sampling_rate ** 2)/(2*sigma**2)
            self.rdp_cumulative[i] += rdp
    def get_privacy_spent(self):
        epsilons = [rdp - math.log(self.delta)/(alpha-1) for rdp, alpha in zip(self.rdp_cumulative,self.orders)]
        return min(epsilons), self.delta
    def reset(self):
        self.rdp_cumulative[:] = 0.0

def simple_adaptive_attack(local_update, global_update, alpha=0.8, beta=-0.2):
    """
    一个简单的自适应攻击：
    - 保持与 global 同方向（骗过 cosine）
    - 混入少量反向 local（破坏收敛）
    """

    g_local = local_update
    g_global = global_update

    # 归一化（避免尺度问题）
    g_local_norm = np.linalg.norm(g_local) + 1e-12
    g_global_norm = np.linalg.norm(g_global) + 1e-12

    g_global_unit = g_global / g_global_norm

    # 构造攻击
    adv = alpha * g_global_unit * g_local_norm + beta * g_local

    return adv


# ---------------- Logging ----------------
def save_csv_row(filepath, row_dict):
    file_exists = os.path.isfile(filepath)
    with open(filepath,"a",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

# ---------------- Data partition ----------------
def partition_mnist_dirichlet(num_clients, alpha=0.5, seed=42):
    """
    将 MNIST 通过 Dirichlet 分配给 num_clients 个客户端。
    为了兼容 CIFARResNet18（3ch, 32x32），把 MNIST 转为 RGB 并 resize 到 32x32。
    """
    print("\n======= Loading MNIST (RGB 32x32) =======")

    tfm_train = transforms.Compose([
        transforms.Resize(32, interpolation=InterpolationMode.BILINEAR),
        transforms.Lambda(lambda img: img.convert("RGB")),  # MNIST 转 3 通道
        transforms.RandomHorizontalFlip(),                  # 和 CIFAR 一样，保持一致
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307),
                             (0.3081, 0.3081, 0.3081))
    ])

    tfm_test = transforms.Compose([
        transforms.Resize(32, interpolation=InterpolationMode.BILINEAR),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307),
                             (0.3081, 0.3081, 0.3081))
    ])

    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=tfm_train)
    mnist_test  = datasets.MNIST('./data', train=False, download=True, transform=tfm_test)

    labels = np.array(mnist_train.targets)
    n_classes = 10

    client_indices = [[] for _ in range(num_clients)]
    rng = np.random.RandomState(seed)

    # ---------- Dirichlet 分配 ----------
    for c in range(n_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)

        proportions = rng.dirichlet([alpha] * num_clients)
        proportions = (proportions / proportions.sum() * len(idx_c)).astype(int)

        # 调整长度
        while proportions.sum() < len(idx_c):
            proportions[rng.randint(num_clients)] += 1
        while proportions.sum() > len(idx_c):
            proportions[rng.randint(num_clients)] -= 1

        start = 0
        for cid, count in enumerate(proportions):
            if count > 0:
                client_indices[cid].extend(idx_c[start:start+count])
                start += count

    # ---------- 构造 client loaders ----------
    client_loaders = {}
    print("\n=== MNIST Client Label Histograms (Dirichlet) ===")
    for cid in range(num_clients):
        ds = Subset(mnist_train, client_indices[cid])
        loader = DataLoader(
            ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=2, pin_memory=True
        )
        client_loaders[cid] = loader

        labels_c = [labels[i] for i in client_indices[cid]]
        hist = np.bincount(labels_c, minlength=n_classes)
        print(f" Client {cid}: total={len(ds)}, hist={hist.tolist()}")

    test_loader = DataLoader(
        mnist_test, batch_size=256, shuffle=False,
        num_workers=2, pin_memory=True
    )
    print("================================\n")
    return client_loaders, test_loader

def partition_cifar10_dirichlet(num_clients, alpha=0.5, seed=42):
    tfm_train = transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
    ])
    tfm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
    ])
    cifar_train = datasets.CIFAR10('./data',train=True,download=True,transform=tfm_train)
    cifar_test  = datasets.CIFAR10('./data',train=False,download=True,transform=tfm_test)
    labels = np.array(cifar_train.targets)
    n_classes = 10
    client_indices = [[] for _ in range(num_clients)]
    rng = np.random.RandomState(seed)
    for c in range(n_classes):
        idx_c = np.where(labels==c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(np.repeat(alpha,num_clients))
        proportions = (proportions/proportions.sum()*len(idx_c)).astype(int)
        while proportions.sum()<len(idx_c):
            proportions[rng.randint(num_clients)]+=1
        while proportions.sum()>len(idx_c):
            proportions[rng.randint(num_clients)]-=1
        start=0
        for cid,count in enumerate(proportions):
            client_indices[cid].extend(idx_c[start:start+count])
            start+=count
    client_loaders={}
    print("\n=== Client Label Histograms (Dirichlet) ===")
    for cid in range(num_clients):
        ds=Subset(cifar_train,client_indices[cid])
        loader=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=2,pin_memory=True)
        client_loaders[cid]=loader
        labels_c = [labels[i] for i in client_indices[cid]]
        hist = np.bincount(labels_c, minlength=n_classes)
        print(f" Client {cid}: total={len(client_indices[cid])}, hist={hist.tolist()}")
    test_loader=DataLoader(cifar_test,batch_size=256,shuffle=False,num_workers=2,pin_memory=True)
    print("================================\n")
    return client_loaders,test_loader

def partition_cifar100_dirichlet(num_clients, alpha=0.5, seed=42):
    tfm_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4865, 0.4409),
            (0.2673, 0.2564, 0.2762)
        )
    ])
    tfm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4865, 0.4409),
            (0.2673, 0.2564, 0.2762)
        )
    ])

    cifar_train = datasets.CIFAR100('./data', train=True, download=True, transform=tfm_train)
    cifar_test  = datasets.CIFAR100('./data', train=False, download=True, transform=tfm_test)

    labels = np.array(cifar_train.targets)
    n_classes = 100

    client_indices = [[] for _ in range(num_clients)]
    rng = np.random.RandomState(seed)

    for c in range(n_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)

        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions / proportions.sum() * len(idx_c)).astype(int)

        while proportions.sum() < len(idx_c):
            proportions[rng.randint(num_clients)] += 1
        while proportions.sum() > len(idx_c):
            proportions[rng.randint(num_clients)] -= 1

        start = 0
        for cid, count in enumerate(proportions):
            client_indices[cid].extend(idx_c[start:start + count])
            start += count

    client_loaders = {}
    print("\n=== CIFAR-100 Client Label Histograms ===")
    for cid in range(num_clients):
        ds = Subset(cifar_train, client_indices[cid])
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=2, pin_memory=True)
        client_loaders[cid] = loader
        print(f" Client {cid}: total={len(client_indices[cid])}")

    test_loader = DataLoader(cifar_test, batch_size=256,
                             shuffle=False, num_workers=2, pin_memory=True)

    print("=======================================\n")
    return client_loaders, test_loader

# ---------------- Model utils ----------------
def get_model_vector(model):
    return torch.cat([p.detach().flatten().cpu() for p in model.parameters()]).numpy().astype(np.float32)

def set_model_vector(model, vec):
    idx=0
    for p in model.parameters():
        num=p.numel()
        chunk = vec[idx:idx+num].reshape(p.shape)
        p.data.copy_(torch.from_numpy(chunk).to(p.data.device,dtype=p.data.dtype))
        idx+=num

def add_model_delta(model, delta):
    vec=get_model_vector(model)
    set_model_vector(model, vec+delta.astype(np.float32))

def bulyan_aggregate(updates, byz_frac=0.2, clip_norm=None):
    n = len(updates)
    if n == 0:
        return np.array([], dtype=np.float32)

    f = int(n * byz_frac)
    m = n - 2 * f
    m = max(1, m)

    # ---------- Step 1: Multi-Krum 选择 ----------
    flattened = [u.flatten().astype(np.float32) for u in updates]
    distances = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(flattened[i] - flattened[j])
            distances[i, j] = distances[j, i] = dist

    scores = []
    for i in range(n):
        sorted_dists = np.sort(distances[i])
        score = np.sum(sorted_dists[1: n - f - 1])
        scores.append(score)

    selected_idx = np.argsort(scores)[:m]
    selected_updates = [flattened[i] for i in selected_idx]
    selected = np.stack(selected_updates, axis=0)  # shape=(m, d)

    # ---------- Step 2: Trimmed Mean ----------
    # 使用矢量化操作取代逐维循环
    low = f
    high = m - f
    if high <= low:  # 避免越界
        low, high = 0, m

    sorted_vals = np.sort(selected, axis=0)
    trimmed = sorted_vals[low:high, :]
    agg = np.mean(trimmed, axis=0).astype(np.float32)

    # ---------- Step 3: 可选 Clip ----------
    if clip_norm is not None:
        norm = np.linalg.norm(agg)
        if norm > clip_norm:
            agg = agg * (clip_norm / (norm + 1e-12))

    print(f"[Bulyan] n={n}, f={f}, selected={m}, trim=[{low},{high}], norm={np.linalg.norm(agg):.4f}")
    return agg



def multi_krum_aggregate(updates, byz_frac=0.2, byz_count=None):
    """
    Multi-Krum 聚合实现（兼容显式 byz_count 与比例 byz_frac）
    """
    n = len(updates)
    if n == 0:
        return np.array([], dtype=np.float32)

    # 确定恶意客户端数 f
    f = byz_count if byz_count is not None else int(n * byz_frac)
    f = min(f, n - 2)  # 保证合法

    num_selected = max(1, n - f - 2)

    # 展平更新向量
    flattened = [u.flatten().astype(np.float32) for u in updates]
    distances = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(flattened[i] - flattened[j])
            distances[i, j] = distances[j, i] = dist

    # 计算每个客户端得分
    scores = []
    for i in range(n):
        sorted_dists = np.sort(distances[i])
        score = np.sum(sorted_dists[1: n - f - 1])  # 理论：最近 n-f-2 个
        scores.append(score)

    selected_idx = np.argsort(scores)[:num_selected]
    selected_updates = [updates[i] for i in selected_idx]

    agg = np.mean(np.stack(selected_updates, axis=0), axis=0)
    return agg



def median_aggregate(updates):

    if len(updates) == 0:
        return np.array([], dtype=np.float32)

    stacked = np.stack(updates, axis=0).astype(np.float32)
    agg = np.median(stacked, axis=0)
    return agg


def centered_clipping(updates, tau=10.0, iters=2):
    mu = np.mean(updates, axis=0)

    for _ in range(iters):
        new_updates = []
        for g in updates:
            diff = g - mu
            norm = np.linalg.norm(diff)
            if norm > tau:
                g = mu + diff / norm * tau
            new_updates.append(g)
        mu = np.mean(new_updates, axis=0)

    return mu

def fltrust_aggregate(updates, global_direction):
    g_ref = global_direction
    g_ref_norm = np.linalg.norm(g_ref) + 1e-12

    agg = np.zeros_like(updates[0])
    total_weight = 0.0

    for g in updates:
        cos_sim = np.dot(g, g_ref) / ((np.linalg.norm(g)+1e-12)*g_ref_norm)
        w = max(0.0, cos_sim)

        agg += w * g
        total_weight += w

    if total_weight > 0:
        agg /= total_weight

    return agg

def fedavg_aggregate(updates, client_weights):
    """
    纯 FedAvg：按 client_weights 做加权平均。
    updates: list of np.array (each is full-dim update)
    client_weights: list-like of positive weights (can be dataset sizes)
    """
    if len(updates) == 0:
        return np.array([], dtype=np.float32)
    w = np.array(client_weights, dtype=np.float64)
    w = w / (w.sum() + 1e-12)
    stacked = np.stack(updates, axis=0).astype(np.float32)
    agg = np.sum(stacked * w[:, None], axis=0)
    return agg.astype(np.float32)
def fedavg_globaldp_aggregate(updates, client_weights, sigma_global, dp_mech="gaussian", rng=None):
    """
    FedAvg + global DP noise added to the final aggregate.
    sigma_global: global noise std (float)
    """
    if rng is None:
        rng = np.random.RandomState()
    agg = fedavg_aggregate(updates, client_weights)
    if sigma_global is None or sigma_global <= 0.0:
        return agg.astype(np.float32)
    if dp_mech == "gaussian":
        noise = rng.normal(0.0, sigma_global, size=agg.shape).astype(np.float32)
    else:
        noise = np.zeros_like(agg, dtype=np.float32)
    return (agg + noise).astype(np.float32)

def fedavg_localdp_aggregate(updates, client_weights, per_client_sigmas, dp_mech="gaussian", rng=None):
    """
    FedAvg where each client's update is first noised with its own sigma,
    then aggregated (more faithful to local DP scenario).
    per_client_sigmas: list of same length as updates
    """
    if rng is None:
        rng = np.random.RandomState()
    if len(updates) == 0:
        return np.array([], dtype=np.float32)
    noised = []
    for u, s in zip(updates, per_client_sigmas):
        if s is None or s <= 0.0:
            noised.append(u.astype(np.float32))
        else:
            if dp_mech == "gaussian":
                n = rng.normal(0.0, s, size=u.shape).astype(np.float32)
            else:
                n = np.zeros_like(u, dtype=np.float32)
            noised.append((u + n).astype(np.float32))
    return fedavg_aggregate(noised, client_weights)
# ---------------- Top-K & Quantization ----------------
def topk_mask(vec, k):
    k_count=int(max(1,vec.size*k)) if k<=1.0 else int(min(vec.size,k))
    if k_count>=vec.size:
        return np.arange(vec.size)
    idx=np.argpartition(-np.abs(vec),k_count-1)[:k_count]
    return np.sort(idx)


def quantize_vector(vals, q, eps=1e-6, rng=None):
    if vals.size==0:
        return np.zeros_like(vals,dtype=np.int64),1.0
    mx=max(np.max(np.abs(vals)),eps)
    scale=(q/2-1)/mx
    qvals=np.clip(np.round(vals*scale),-q/2,q/2-1).astype(np.int64)
    return qvals,scale

def dequantize_vector(z,scale):
    return z.astype(np.float32)/scale if scale!=0 else z.astype(np.float32)

# ---------------- DP dual-adaptive ----------------

def _approx_gaussian_report_sigma(sensitivity, eps_report):
    eps_report = max(float(eps_report), 1e-8)
    return float(sensitivity / eps_report)

def _rdp_gaussian_alpha_sigma(alpha, sampling_rate, sigma):
    return (alpha * (sampling_rate ** 2)) / (2.0 * (sigma ** 2) + 1e-30)


# ---------------- SARA-adaptive aggregator ----------------
def sara_adaptive_aggregate(
    updates,
    client_weights,
    strength_factor=1.0,
    cos_factor=1.0,
    gamma_eff=None,
    fraud_cos_thresh=-0.2,
    round_num=1,
    median_correction=True,
    verbose=False,
    prev_client_vecs=None,
    lambda_shift_server=1.0,
    shift_clip=5.0
):



    n = len(updates)
    if n == 0:
        return np.array([], dtype=np.float32)

    # cast and stack safely
    stacked = np.stack([np.asarray(u, dtype=np.float32) for u in updates], axis=0)
    # basic statistics
    mean_vec = np.mean(stacked, axis=0)
    mean_norm = np.linalg.norm(mean_vec)
    if mean_norm < 1e-12:
        # if mean is (near) zero, fall back to simple average of updates
        if verbose:
            print(f"[SARA][R{round_num}] mean_vec nearly zero (norm={mean_norm:.6g}) -> fallback to weighted average")
        weighted = np.average(stacked, axis=0, weights=np.array(client_weights, dtype=np.float32))
        return np.nan_to_num(weighted, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    mean_norm = mean_norm + 1e-12
    mean_dir = mean_vec / mean_norm

    norms = np.linalg.norm(stacked, axis=1) + 1e-12
    cos_sims = np.sum(stacked * mean_vec[None, :], axis=1) / (norms * mean_norm)
    dists = np.linalg.norm(stacked - mean_vec[None, :], axis=1)

    if gamma_eff is None:
        gamma_eff = 1.0

    # Distance and cosine based components
    dist_norm = max(np.mean(dists), 1e-12)
    dist_weights = np.exp(-(abs(strength_factor) * dists) / dist_norm)
    cos_raw = np.clip(cos_factor * cos_sims, -20.0, 20.0)
    cos_weights = np.exp(cos_raw)

    # directional weights (from gamma_eff and cos)
    directional = np.ones(n, dtype=np.float32)

    for i in range(n):
        c = float(cos_sims[i])

        # ★ Step1: 强腐败过滤（强于你现在的 -0.2）
        if c < fraud_cos_thresh:  # e.g., -0.2
            directional[i] = 1e-4  # not zero, but extremely small
            continue

        # ★ Step2: 强化方向权重 gamma_eff*(cos)
        raw = 2.0 * gamma_eff * c  # 从 1.0 → 2.0，提高方向敏感度
        try:
            directional[i] = math.exp(raw)
        except OverflowError:
            directional[i] = 1e8

    # ★ Step3: 不要强 clip（改成软 clip）
    directional = np.clip(directional, 1e-6, 1e6)

    # ---------- server-side shift decay (prevents sudden drifters) ----------
    if prev_client_vecs is None:
        prev_client_vecs = [None] * n

    shift_vals = np.zeros(n, dtype=np.float32)
    for i in range(n):
        prev_v = prev_client_vecs[i]
        if prev_v is None:
            shift_vals[i] = 0.0
        else:
            try:
                num = np.linalg.norm(stacked[i] - prev_v)
                den = (np.linalg.norm(prev_v) + 1e-12)
                shift_vals[i] = float(num / den)
            except Exception:
                shift_vals[i] = 0.0

    shift_vals_clipped = np.clip(shift_vals, 0.0, shift_clip)

    try:
        shift_decay = np.exp(-float(lambda_shift_server) * shift_vals_clipped)
    except Exception:
        shift_decay = np.ones_like(shift_vals_clipped)

    directional = directional * shift_decay


    # ---------- combine weights  ----------

    client_w = np.array(client_weights, dtype=np.float32)
    # 原本：dist * cos * directional * weight
    combined = dist_weights * cos_weights * directional * client_w
    # ★ Step1: 强化“好客户端”（多数方向）
    # (★) 强化多数（幂运算会让大值更大，小值更小）
    gamma_power = 2.0  # 或 3.0，越大越鲁棒
    combined = np.power(combined, gamma_power, dtype=np.float64)
    # ---- Stable log-softmax reweighting ----
    combined = np.asarray(combined, dtype=np.float64)

    # 防止 -inf / nan：替换非法值为很小的负值（log-space）
    combined[np.isnan(combined)] = -1e9
    combined[np.isinf(combined)] = -1e9

    # 进入 log 域（避免出现 combined 全是负，但 log仍可处理）
    log_comb = np.log(np.clip(combined, 1e-30, None))  # clip 避免 log(0)

    # 经典 log-softmax: x_i = exp(log_comb_i - max - log(sum(exp(...))))
    m = np.max(log_comb)
    log_comb_stable = log_comb - m
    exp_comb = np.exp(log_comb_stable)
    soft_weights = exp_comb / (np.sum(exp_comb) + 1e-12)

    # 经过 softmax 后自动满足 sum=1, 不需要再重新 clip
    combined = soft_weights.astype(np.float32)

    # aggregated vector
    agg = np.sum(stacked * combined[:, None], axis=0)

    # ---------- median-correction (safe) ----------
    if median_correction:
        # protect median_dir computation
        unit_vecs = stacked / (norms[:, None] + 1e-12)
        median_dir = np.median(unit_vecs, axis=0)
        median_norm = np.linalg.norm(median_dir)
        if median_norm < 1e-8:
            # fallback to mean_dir (safer)
            median_dir = mean_dir.copy()
        else:
            median_dir = median_dir / (median_norm + 1e-12)

        agg_norm = np.linalg.norm(agg) + 1e-12

        # if round 1 or mean_cos very small, reduce median influence to avoid NaN
        mean_cos = float(np.mean(cos_sims))
        if round_num == 1 or abs(mean_cos) < 1e-3:
            # weaker correction on round 1 when history absent
            agg = 0.95 * agg + 0.05 * (agg_norm * mean_dir)
        else:
            agg = (0.8 * agg +
                   0.2 * (agg_norm * mean_dir) +
                   0.05 * (agg_norm * median_dir))


    # ---------- diagnostics ----------
    fraud_count = int(np.sum(cos_sims < fraud_cos_thresh))
    mean_cos_val = float(np.mean(cos_sims))
    mean_shift = float(np.mean(shift_vals))

    agg = np.nan_to_num(agg, nan=0.0, posinf=0.0, neginf=0.0)

    if verbose:
        print(f"[DARA][R{round_num}] n={n}, fraud_count={fraud_count}, mean_cos={mean_cos_val:.4f}, "
              f"agg_norm={np.linalg.norm(agg):.6f}, mean_shift={mean_shift:.4f}")
        # weight stats
        try:
            print(f"[DARA][R{round_num}] combined_min={combined.min():.6f}, combined_max={combined.max():.6f}, combined_sum={combined.sum():.6f}")
        except Exception:
            pass

    return agg.astype(np.float32)


# ---------------- Local training ----------------
def local_train(model, loader, epochs=1, lr=None):
    if lr is None: lr=LR
    m=copy.deepcopy(model).to(DEVICE)
    opt=optim.SGD(m.parameters(),lr=lr,momentum=MOMENTUM,weight_decay=WEIGHT_DECAY)
    loss_fn=nn.CrossEntropyLoss()
    m.train()
    total_loss=0.0; n_batches=0
    for _ in range(epochs):
        for xb,yb in loader:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE)
            opt.zero_grad()
            out=m(xb)
            loss=loss_fn(out,yb)
            loss.backward()
            opt.step()
            total_loss+=float(loss.item())
            n_batches+=1
    avg_loss=total_loss/max(1,n_batches)
    delta=get_model_vector(m)-get_model_vector(model)
    if np.isnan(delta).any() or np.isinf(delta).any():
        print(f"[DEBUG-LOCAL][cid UNKNOWN] local_train produced NaN/Inf delta, min={delta.min()}, max={delta.max()}")
    return delta, avg_loss

# ---------------- One round ----------------
def brea_round(global_model, client_loaders, attack_type, base_sigma, noise_type,
               round_num, controller, base_topk=TOPK, strength_factor=1.0,
               model_prev=None, agg_type="sara-adaptive", rng_round=None, accountant=None):
    # ----- setup -----
    n_total = len(client_loaders)
    if rng_round is None:
        rng_round = np.random.RandomState(controller.seed + round_num)
    client_ids = list(range(n_total))
    chosen = client_ids if CLIENT_FRACTION >= 1.0 else rng_round.choice(client_ids, CLIENTS_PER_ROUND, replace=False).tolist()
    n = len(chosen)
    # ----- global direction (for adaptive attack & FLTrust) -----
    global_direction = None
    if model_prev is not None:
        try:
            global_direction = get_model_vector(global_model) - get_model_vector(model_prev)
            if np.linalg.norm(global_direction) < 1e-12:
                global_direction = None
        except Exception:
            global_direction = None

    full_dim = get_model_vector(global_model).size if 'get_model_vector' in globals() else None

    clean_updates_for_ctrl = []     # dense clean updates (torch tensors) for detection/controller
    client_losses = []
    weights = []
    attacker_set = set()

    # optional: choose attackers
    if BYZANTINE > 0 and n > 0:
        rng_attack = np.random.RandomState(controller.seed + 5000 + round_num)
        k = min(BYZANTINE, n)
        picked = rng_attack.choice(chosen, size=k, replace=False)
        attacker_set = set(int(x) for x in picked)

    # ----- 0) collect local (clean) updates first (no DP yet) -----
        # ----- 0) collect local (clean) updates first (no DP yet) -----
    for cid in chosen:

        # ✅ 先统一训练（所有客户端）
        delta, local_loss = local_train(global_model, client_loaders[cid], LOCAL_EPOCHS, lr=LR)

        # ✅ 如果是攻击者，再修改 delta
        if cid in attacker_set:

            if attack_type == "sign-flip":
                delta = -SIGN_FLIP_ALPHA * delta

            elif attack_type == "random":
                rand = np.random.randn(full_dim).astype(np.float32)
                delta = rand * ((np.linalg.norm(delta) + 1e-12) / (np.linalg.norm(rand) + 1e-12))

            elif attack_type == "adaptive":
                 if global_direction is not None:
                    delta = simple_adaptive_attack(delta, global_direction)
                # 如果第一轮，没有方向，就不攻击（更稳定）

        client_losses.append(float(local_loss) if local_loss is not None else float('nan'))

        w = len(client_loaders[cid].dataset) if hasattr(client_loaders[cid], "dataset") else 1
        weights.append(w)

        # keep clean update for detection/controller (torch)
        try:
            clean_updates_for_ctrl.append(
                torch.tensor(delta.astype(np.float32), dtype=torch.float32, device=DEVICE))
        except Exception:
            clean_updates_for_ctrl.append(torch.tensor(delta.astype(np.float32), dtype=torch.float32))

    # ----- 1) DETECT suspicious clients (on clean updates) -----
    try:
        sus_idxs, soft_weights, diag = controller.detect_suspicious_clients_multifactor(
            client_updates=clean_updates_for_ctrl,
            client_ids=chosen,
            client_weights=weights,
            use_history=True,
            hard_neg_th=-0.7,
            w_cos=0.5, w_norm=0.3, w_drift=0.2, score_q=0.80
        )
    except Exception as e:
        if controller.verbose:
            print("[WARN] detection failed:", e)
        sus_idxs = []
        soft_weights = np.ones(len(weights), dtype=np.float32)
        diag = {}

    suspicious_clients = sus_idxs
    suspicious_count = len(suspicious_clients)
    controller.last_suspicious_count = suspicious_count

    if controller.verbose:
        print(f"[DETECT][R{round_num}] chosen={chosen}")
        for i, cid in enumerate(chosen):
            cos_i = diag.get(i, {}).get("cos", float('nan'))
            comp_i = diag.get(i, {}).get("comp", float('nan'))
            sw = float(soft_weights[i]) if i < len(soft_weights) else 1.0
            print(f"  client {cid:2d}: cos={cos_i:.4f}, comp={comp_i:.4f}, soft_w={sw:.4f}, weight(orig)={weights[i]}")
        print(f"[DETECT][R{round_num}] suspicious_idxs={sus_idxs}, suspicious_count={suspicious_count}")

    # normalize soft_weights fallback
    try:
        soft_weights = np.asarray(soft_weights, dtype=np.float32)
        if soft_weights.ndim == 0:
            soft_weights = np.ones(len(weights), dtype=np.float32)
    except Exception:
        soft_weights = np.ones(len(weights), dtype=np.float32)
    if soft_weights.shape[0] != len(weights):
        soft_weights = np.ones(len(weights), dtype=np.float32)
    adj_weights = [float(w) * float(soft_weights[i]) for i, w in enumerate(weights)]

    # ----- 2) per-client sigma plan (use controller.client_sigma_for_update) -----

    if FIXED_DP:
        sigma_clients = [FIXED_SIGMA for _ in chosen]
    else:
        sigma_clients = []
        for i, cid in enumerate(chosen):
            try:
                # pass clean dense numpy vector to planner (controller implementation may vary)
                u_np = clean_updates_for_ctrl[i].cpu().numpy() if isinstance(clean_updates_for_ctrl[i],
                                                                             torch.Tensor) else np.asarray(
                    clean_updates_for_ctrl[i], dtype=np.float32)
                sigma_c = controller.client_sigma_for_update(
                    client_update=u_np,
                    var_now=float(np.var(u_np)),
                    loss_now=float(client_losses[i]) if not np.isnan(client_losses[i]) else None,
                    loss_prev=float(controller.prev_loss) if getattr(controller, "prev_loss",
                                                                     None) is not None else None,
                    client_weight=weights[i],
                    global_sigma=base_sigma,
                    client_id=cid
                )
                if (not np.isfinite(sigma_c)) or sigma_c <= 0:
                    sigma_c = float(base_sigma)
            except Exception:
                sigma_c = float(base_sigma)
            sigma_clients.append(float(sigma_c))
            if controller.verbose:
                print(f"[DBG-sigma-plan] cid={cid} planned_sigma={sigma_clients[-1]:.6f}")

    # ----- 3) per-client DP: apply to TOP-K values, then reconstruct full dense -----
    raw_dense_updates = [x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, dtype=np.float32) for x in clean_updates_for_ctrl]
    updates = []
    per_client_sigmas = []

    for i, cid in enumerate(chosen):
        dense = raw_dense_updates[i].astype(np.float32)
        try:
            mask_idx = topk_mask(dense, base_topk)
            vals = dense[mask_idx].astype(np.float32)
            clip_norm = float(np.linalg.norm(vals)) + 1e-12
        except Exception:
            mask_idx = np.arange(dense.size)
            vals = dense.copy().astype(np.float32)
            clip_norm = float(np.linalg.norm(vals)) + 1e-12

        planned_sigma = float(sigma_clients[i])
        try:
            vals_dp, used_sigma, acct_info = apply_dp_dual_adaptive_secure(
                vec=vals,
                clip_norm=clip_norm,
                sigma=planned_sigma,
                rng=np.random.RandomState(controller.seed + 2000 + round_num + cid),
                noise_type=noise_type,
                sampling_rate=None
            )
            if accountant is not None and acct_info is not None:
                try:
                    accountant.accumulate_from_account_info(acct_info)
                except Exception:
                    pass
        except Exception as e:
            if controller.verbose:
                print(f"[WARN] apply_dp failed cid={cid}: {e}")
            vals_dp = vals.copy()
            used_sigma = planned_sigma

        # reconstruct dense
        dense_dp = np.zeros(dense.shape, dtype=np.float32)
        dense_dp[mask_idx] = vals_dp
        dense_dp = np.nan_to_num(dense_dp, nan=0.0, posinf=1e6, neginf=-1e6)

        updates.append(dense_dp)
        per_client_sigmas.append(float(used_sigma))

        if controller.verbose:
            before_l2 = np.linalg.norm(vals)
            after_l2 = np.linalg.norm(vals_dp)
            print(f"[INFO][Round {round_num}] cid={cid} mask_L2_before={before_l2:.6f}, mask_L2_afterDP={after_l2:.6f}, used_sigma={used_sigma:.6f}")

    # ----- 4) compute strength (fallback safe) -----
    try:
        strength = controller.compute_strength([torch.tensor(u, dtype=torch.float32) for u in updates])
    except Exception:
        norms_list = [np.linalg.norm(u) for u in updates] if len(updates) > 0 else [0.0]
        strength = float(0.5 + np.log1p(np.var(norms_list) + 1e-12))

    var_now_scalar = float(np.var([np.linalg.norm(u) for u in updates])) if len(updates) > 0 else 0.0
    avg_loss = float(np.mean([l for l in client_losses if l is not None])) if any([l is not None for l in client_losses]) else float('nan')

    # ----- 5) Aggregation -----
    byz_count = BYZANTINE
    byz_frac = byz_count / max(1, n_total)
    stacked = np.stack(updates, axis=0) if len(updates) > 0 else np.zeros((0, full_dim), dtype=np.float32)
    if controller.verbose:
        print(f"[DEBUG-AGG][Round {round_num}] stacked_shape={stacked.shape} anynan={np.isnan(stacked).any()}, anyinf={np.isinf(stacked).any()}")

    if agg_type == "fedavg":
        #agg = fedavg_localdp_aggregate(updates, adj_weights, per_client_sigmas)
        agg = fedavg_aggregate(updates, adj_weights)
        agg_method = "fedavg"
    elif agg_type == "multi_krum":
        agg = multi_krum_aggregate(updates, byz_frac=byz_frac, byz_count=byz_count)
        agg_method = "multi_krum"
    elif agg_type == "median":
        agg = median_aggregate(updates)
        agg_method = "median"
    elif agg_type == "fltrust":
        if global_direction is not None:
            agg = fltrust_aggregate(updates, global_direction)
        else:
            agg = fedavg_aggregate(updates, adj_weights)
        agg_method = "fltrust"
    else:
        prev_vecs = controller.prev_server_updates if (hasattr(controller, "prev_server_updates") and getattr(controller, "prev_chosen", None) == chosen) else None
        agg = sara_adaptive_aggregate(
            updates,
            client_weights=adj_weights,
            cos_factor=getattr(controller, "cos_factor", 1.0),
            strength_factor=strength_factor * strength,
            gamma_eff=getattr(controller, "last_gamma", 1.0),
            verbose=controller.verbose,
            round_num=round_num,
            prev_client_vecs=prev_vecs,
            lambda_shift_server=getattr(controller, "lambda_shift_server", 1.0),
            shift_clip=getattr(controller, "shift_clip", 5.0)
        )
        agg_method = "dara"

    # numeric sanitize
    if np.any(np.isnan(agg)) or np.any(np.isinf(agg)):
        if controller.verbose:
            print(f"[WARN][Round {round_num}] agg contains NaN/Inf -> zeroing")
        agg = np.nan_to_num(agg, nan=0.0, posinf=0.0, neginf=0.0)

    # ----- 6) simulate & controller.update -----
    try:
        model_sim = copy.deepcopy(model_prev) if model_prev is not None else copy.deepcopy(global_model)
        add_model_delta(model_sim, agg)
        median_sigma = float(np.median(per_client_sigmas)) if len(per_client_sigmas) > 0 else float(base_sigma)
        gamma_after, sigma_after = controller.update(
            var_now=var_now_scalar,
            loss_now=avg_loss,
            strength=strength,
            model=model_sim,
            model_prev=model_prev,
            rng=controller.rng_global,
            round_num=round_num,
            client_fractions=[len(client_loaders[c].dataset) for c in range(len(client_loaders))],
            client_updates=clean_updates_for_ctrl,
            suspicious_idxs=sus_idxs,
            soft_weights=soft_weights,
            median_client_sigma=median_sigma
        )
        if not np.isfinite(sigma_after) or sigma_after <= 0:
            sigma_after = float(controller.prev_sigma if getattr(controller, "prev_sigma", None) is not None else base_sigma)
        if not np.isfinite(gamma_after):
            gamma_after = float(getattr(controller, "last_gamma", 1.0))
        controller.last_gamma = float(gamma_after)
        controller.prev_sigma = float(sigma_after)
    except Exception as e:
        if controller.verbose:
            print("[WARN] post-agg controller.update failed:", e)
        sigma_after = float(getattr(controller, "prev_sigma", base_sigma))
        gamma_after = float(getattr(controller, "last_gamma", 1.0))

    controller.prev_server_updates = [u.copy() for u in updates]
    controller.prev_chosen = chosen.copy()

    # ----- Return (same order your run_experiment expects) -----
    return (
        agg_method,          # str
        agg,                 # aggregated update (vector)
        strength,            # float
        sigma_after,         # float
        gamma_after,         # float
        avg_loss,            # float
        client_losses,       # list
        attacker_set,        # set
        suspicious_clients,  # list
        suspicious_count,    # int
        per_client_sigmas,   # list
        soft_weights,        # list or np.array
        updates              # per-client full-dim updates (list)
    )

# ---------------- Evaluation ----------------
def evaluate(model, loader):
    model.eval()
    correct=0; total=0
    with torch.no_grad():
        for xb,yb in loader:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE)
            out=model(xb)
            pred=out.argmax(1)
            correct+=(pred==yb).sum().item()
            total+=yb.size(0)
    return 100.0*correct/total

# ---------------- Experiment ----------------
def run_experiment():
    controller = AdaptiveStabilityController(base_sigma=BASE_SIGMA, verbose=True, device=DEVICE)
    loaders, test_loader = partition_mnist_dirichlet(NUM_CLIENTS, alpha=0.5)
    global_model = CIFARResNet18(num_classes=10).to(DEVICE)
    csv_path = os.path.join(OUTDIR, "secure_sara_results_multibaseline.csv")
    os.makedirs(OUTDIR, exist_ok=True)

    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            'round', 'method', 'attack', 'noise_type', 'sigma',
            'strength', 'acc', 'avg_loss',
            'attackers', 'suspicious_count', 'suspicious_clients',
            'gamma_eff', 'eps', 'delta',
            'avg_client_sigma', 'median_client_sigma', 'softmean'
        ])
        agg_methods = ["fltrust"]
        noise_types = ["gaussian"]
        sigmas_map = {"gaussian": [0.05]}

        attacks = ["none", "sign-flip", "random","adaptive"]
        metrics_log_all = {}

        for agg_type in agg_methods:
            print(f"\n==============================")
            print(f"▶▶ Baseline: {agg_type.upper()}")
            print(f"==============================\n")

            metrics_log = []

            for atk in attacks:
                for nt in noise_types:
                    for sigma_init in sigmas_map[nt]:

                        # ===== reset controller FULL state =====
                        controller.prev_var = None
                        controller.prev_loss = None
                        controller.prev_strength = None
                        controller.prev_sigma = sigma_init
                        controller.prev_server_updates = [None] * NUM_CLIENTS

                        # enable/disable control
                        controller.controller_active = (agg_type == "dara")

                        if controller.controller_active:
                            # inject all DARA hyperparameters
                            controller.sigma_momentum = 0.8
                            controller.gamma_k = 0.03
                            controller.lambda_shift_server = 0.1
                            controller.shift_clip = 5.0
                            controller.sigma_clip = (0.0, 2.0)
                        print(f"\n>>> Method={agg_type}, Attack={atk}, Noise={nt}, init_sigma={sigma_init}")

                        q = CLIENTS_PER_ROUND / NUM_CLIENTS
                        accountant = GaussianPrivacyAccountant(
                            sampling_rate=q,
                            delta=1e-5
                        )

                        for r in range(1, ROUNDS + 1):

                            model_prev = copy.deepcopy(global_model)
                            rng = np.random.RandomState(SEED + r)

                            (agg_method, agg, st, sigma, gamma_eff, avg_loss,
                             client_losses, attacker_set, suspicious_clients,
                             suspicious_count, per_client_sigmas, soft_weights, updates) = brea_round(
                                    global_model, loaders, atk, sigma_init, nt, r,
                                    controller, model_prev=model_prev,
                                    agg_type=agg_type, rng_round=rng,
                                    accountant=accountant
                                )
                            if not controller.controller_active:
                                sigma = sigma_init

                            add_model_delta(global_model, agg)
                            acc = evaluate(global_model, test_loader)

                            # update DP accountant: only once per round
                            if sigma > 0:
                                accountant.update(sigma)

                            eps, delta_val = accountant.get_privacy_spent()

                            avg_client_sigma = float(np.mean(per_client_sigmas)) if per_client_sigmas else sigma
                            median_client_sigma = float(np.median(per_client_sigmas)) if per_client_sigmas else sigma
                            softmean = float(np.mean(soft_weights)) if soft_weights is not None else 1.0

                            print(f"[{agg_type}][Round {r}] acc={acc:.2f}%, sigma={sigma:.4f}, "
                                  f"strength={st:.3f}, loss={avg_loss:.4f}, suspicious={suspicious_count}, gamma={gamma_eff:.4f}")

                            writer.writerow([
                                r, agg_type, atk, nt, sigma,
                                st, acc, avg_loss,
                                json.dumps(sorted(list(attacker_set))),
                                suspicious_count,
                                json.dumps(suspicious_clients),
                                gamma_eff, eps, delta_val,
                                avg_client_sigma, median_client_sigma, softmean
                            ])

            metrics_log_all[agg_type] = metrics_log

    print("\n✅ Experiment finished, results saved to", csv_path)

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    t0=time.time()
    run_experiment()
    print("Elapsed: {:.1f}s".format(time.time()-t0))
