import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
import csv
import os
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
# =======================
# Config
# =======================

NUM_CLIENTS = 50
CLIENTS_PER_ROUND = 40
ROUNDS = 200
LOCAL_EPOCHS = 2
BATCH_SIZE = 64
LR = 0.05
SERVER_LR = 0.5
ATTACK_RATIO = 0.2
METHOD = "trimmed"  # fedavg / trimmed / median / krum / mcpr
TRIM_RATIO = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# =======================
# CSV Logger
# =======================
csv_path = "training_log2.csv"

# 如果文件存在就删除（避免追加旧实验）
if os.path.exists(csv_path):
    os.remove(csv_path)

csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)

# 写入表头
csv_writer.writerow([
    "Method"
    "round",
    "accuracy",
    "loss",
    "malicious_reputation",
    "benign_reputation",
    "lambda_proj",
    "grad_norm"
])

# 立即写入磁盘
csv_file.flush()

print("CSV logging to:", os.path.abspath(csv_path))
# =======================
# Model
# =======================

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64*6*6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# =======================
# Utils
# =======================

def flatten(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def load_flattened(model, flat):
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat[idx:idx+numel].view_as(p))
        idx += numel

# =======================
# Dataset (Dirichlet Non-IID)
# =======================
def partition_cifar10_dirichlet(num_clients, alpha=0.5, seed=42):

    tfm_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914,0.4822,0.4465),
            (0.2023,0.1994,0.2010)
        )
    ])

    tfm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914,0.4822,0.4465),
            (0.2023,0.1994,0.2010)
        )
    ])

    cifar_train = datasets.CIFAR10(
        './data', train=True, download=True, transform=tfm_train)

    cifar_test = datasets.CIFAR10(
        './data', train=False, download=True, transform=tfm_test)

    labels = np.array(cifar_train.targets)
    n_classes = 10

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

            client_indices[cid].extend(idx_c[start:start+count])
            start += count

    client_loaders = {}

    print("\n=== Client Label Histograms (Dirichlet) ===")

    for cid in range(num_clients):

        ds = Subset(cifar_train, client_indices[cid])

        loader = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )

        client_loaders[cid] = loader

        labels_c = [labels[i] for i in client_indices[cid]]

        hist = np.bincount(labels_c, minlength=n_classes)

        print(f"Client {cid}: total={len(client_indices[cid])}, hist={hist.tolist()}")

    test_loader = DataLoader(
        cifar_test,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print("================================\n")

    return client_loaders, test_loader
# =======================
# Local Training
# =======================
def local_train(model, loader, malicious=False):

    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    old_weights = flatten(model).clone()

    for _ in range(LOCAL_EPOCHS):
        for data, target in loader:

            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            loss = F.cross_entropy(model(data), target)

            loss.backward()

            optimizer.step()

    new_weights = flatten(model).clone()

    grad = new_weights - old_weights

    if malicious:
        grad = -grad

    return grad.detach()
# =======================
# Baselines
# =======================

def fedavg(grads):
    return torch.stack(grads).mean(dim=0)

def trimmed_mean(grads):
    grads = torch.stack(grads)
    sorted_grads, _ = torch.sort(grads, dim=0)
    n = grads.size(0)
    trim = int(TRIM_RATIO * n)
    return sorted_grads[trim:n-trim].mean(dim=0)

def median(grads):
    grads = torch.stack(grads)
    return grads.median(dim=0).values

def krum(grads):
    grads = torch.stack(grads)
    n = grads.size(0)
    f = int(ATTACK_RATIO * n)
    scores = []

    for i in range(n):
        distances = []
        for j in range(n):
            if i != j:
                distances.append(torch.norm(grads[i]-grads[j])**2)
        distances = torch.stack(distances)
        score = torch.sort(distances)[0][:n-f-2].sum()
        scores.append(score)

    idx = torch.argmin(torch.stack(scores))
    return grads[idx]

# =======================
# MCPR
# =======================

class MCPR:

    def __init__(self, model):

        self.model = model

        # 历史动量方向
        self.momentum = None

        # 客户端声誉
        self.reputation = [0.5] * NUM_CLIENTS

        # 投影强度（用于日志记录）
        self.lambda_proj = 0.0


    def aggregate(self, grads, selected_ids, current_round):

        grads_tensor = torch.stack(grads)

        if len(grads) == 0:
            return grads_tensor.mean(dim=0)

        # ==========================
        # Step 1 Robust Base Direction
        # ==========================

        g_base = trimmed_mean(grads)

        # ==========================
        # Step 2 Temporal Momentum
        # ==========================

        if self.momentum is None:

            m = g_base

        else:

            cos_sim = F.cosine_similarity(
                self.momentum.unsqueeze(0),
                g_base.unsqueeze(0),
                dim=1
            )[0]

            # 防止方向翻转
            if cos_sim < 0:
                m = g_base
            else:
                m = 0.7 * self.momentum + 0.3 * g_base

        # 归一化 momentum（关键）
        self.momentum = m / (torch.norm(m) + 1e-12)

        # ==========================
        # Step 3 Gradient Dispersion
        # ==========================

        disp = torch.norm(grads_tensor - g_base, dim=1).median()
        disp = disp / (torch.norm(g_base) + 1e-12)

        lam = disp / (1.0 + disp)
        lam = torch.clamp(lam, 0.0, 0.3)
        # ===== warm-up 阶段 =====
        if current_round < 10:
            lam = lam * (current_round / 10)

        self.lambda_proj = lam.item()

        # ==========================
        # Step 4 Adaptive Projection
        # ==========================

        proj_grads = []
        sims = []

        for g in grads:

            proj = (
                torch.dot(g, self.momentum) /
                (torch.norm(self.momentum) ** 2 + 1e-12)
            ) * self.momentum

            g_new = (1 - lam) * g + lam * proj

            proj_grads.append(g_new)

            sim = F.cosine_similarity(
                g_new.unsqueeze(0),
                self.momentum.unsqueeze(0),
                dim=1
            )[0]

            sims.append(sim)

        sims = torch.stack(sims)

        mu = sims.mean()
        sigma = sims.std()

        # ==========================
        # Step 5 Reputation Update
        # ==========================

        for i, cid in enumerate(selected_ids):

            sim = sims[i].item()

            rep = self.reputation[cid]

            # 奖励
            if sim > mu + sigma:
                rep += 0.04
            #惩罚
            elif sim < mu - sigma:
                rep -= 0.08

            sim_norm = (sim + 1) / 2
            rep = 0.95 * rep + 0.05 * sim_norm

            self.reputation[cid] = max(0.05, min(1.0, rep))

        # ==========================
        # Step 6 Weight Calculation
        # ==========================

        rep_tensor = torch.tensor(
            [self.reputation[cid] for cid in selected_ids],
            device=device
        )

        # similarity 权重
        sim_weight = (sims + 1) / 2

        weights = rep_tensor * sim_weight

        weights = weights / (weights.sum() + 1e-12)

        # ==========================
        # Step 7 Aggregation
        # ==========================

        proj_tensor = torch.stack(proj_grads)

        global_grad = torch.sum(
            weights.unsqueeze(1) * proj_tensor,
            dim=0
        )

        # ==========================
        # Step 8 Energy Restore
        # ==========================

        avg_norm = grads_tensor.norm(dim=1).median()

        global_grad = (
            global_grad /
            (torch.norm(global_grad) + 1e-12)
        ) * avg_norm

        return global_grad

# =======================
# Evaluation
# =======================

def evaluate(model):

    model.eval()
    loader = test_loader

    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in loader:

            data, target = data.to(device), target.to(device)

            out = model(data)

            loss = F.cross_entropy(out, target)

            total_loss += loss.item() * data.size(0)

            pred = out.argmax(dim=1)

            correct += pred.eq(target).sum().item()

            total += target.size(0)

    acc = correct / total
    avg_loss = total_loss / total

    return acc, avg_loss

# =======================
# Training Loop
# =======================
client_loaders, test_loader = partition_cifar10_dirichlet(
    NUM_CLIENTS,
    alpha=0.5
)
global_model = CNN().to(device)
mcpr = MCPR(global_model)

malicious_ids = random.sample(range(NUM_CLIENTS),
                               int(NUM_CLIENTS*ATTACK_RATIO))
print("Method:", METHOD)
print("Attack ratio:", ATTACK_RATIO)
print("Malicious clients:", len(malicious_ids))
for round in range(ROUNDS):
    mal_rep = 0.0
    ben_rep = 0.0
    selected = random.sample(range(NUM_CLIENTS), CLIENTS_PER_ROUND)
    grads = []

    for cid in selected:
        local_model = CNN().to(device)
        load_flattened(local_model, flatten(global_model))
        grad = local_train(
            local_model,
            client_loaders[cid],
            malicious=(cid in malicious_ids)
        )
        grads.append(grad)

    if METHOD == "fedavg":
        global_grad = fedavg(grads)
    elif METHOD == "trimmed":
        global_grad = trimmed_mean(grads)
    elif METHOD == "median":
        global_grad = median(grads)
    elif METHOD == "krum":
        global_grad = krum(grads)
    elif METHOD == "mcpr":
        global_grad = mcpr.aggregate(grads, selected, round)
    # ===== 记录指标 =====
    lambda_proj = mcpr.lambda_proj if METHOD == "mcpr" else 0.0
    grad_norm = global_grad.norm().item()

    flat = flatten(global_model)
    load_flattened(global_model, flat + SERVER_LR * global_grad)

    acc, loss = evaluate(global_model)

    if METHOD == "mcpr":
        if len(malicious_ids) > 0:
            mal_rep = np.mean([mcpr.reputation[i] for i in malicious_ids])
        else:
            mal_rep = 0.0

    benign_ids = [i for i in range(NUM_CLIENTS) if i not in malicious_ids]

    if len(benign_ids) > 0:
        ben_rep = np.mean([mcpr.reputation[i] for i in benign_ids])
    else:
        ben_rep = 0.0

    print(
        f"Round {round}: "
        f"Acc={acc:.4f}, "
        f"Loss={loss:.4f}, "
        f"MalRep={mal_rep:.3f}, "
        f"BenRep={ben_rep:.3f}, "
        f"λ={lambda_proj:.3f}, "
        f"GradNorm={grad_norm:.3f}"
    )
    # ======================
    # 写入 CSV
    # ======================
    csv_writer.writerow([
        METHOD,
        round,
        float(acc),
        float(loss),
        float(mal_rep),
        float(ben_rep),
        float(lambda_proj),
        float(grad_norm)
    ])
    # 强制写入磁盘
    csv_file.flush()
    # 调试信息
    print(f"CSV saved for round {round}")
# ======================
# 关闭 CSV
# ======================
csv_file.close()
print("Training finished. CSV saved.")