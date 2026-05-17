import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def log_training_metrics(metrics_log, round_num, acc, loss, strength, sigma, agg_method):
    """
    在每轮训练后调用此函数，记录当前轮指标。
    """
    metrics_log.append({
        "round": round_num,
        "acc": acc,
        "loss": loss,
        "strength": strength,
        "sigma": sigma,
        "agg": agg_method
    })
    return metrics_log


def visualize_training_metrics(metrics_log, save_dir="results7", file_name="training_dynamics.png", smooth_sigma=1):
    """
    绘制训练指标曲线 (acc / loss / sigma / strength)，自动区分聚合阶段。
    """
    if not metrics_log:
        print("[visualize] Warning: metrics_log is empty.")
        return

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rounds = [m["round"] for m in metrics_log]
    accs = [m["acc"] for m in metrics_log]
    losses = [m["loss"] for m in metrics_log]
    strengths = [m["strength"] for m in metrics_log]
    sigmas = [m["sigma"] for m in metrics_log]
    aggs = [m["agg"] for m in metrics_log]

    # 平滑处理
    accs_smooth = gaussian_filter1d(accs, sigma=smooth_sigma)
    losses_smooth = gaussian_filter1d(losses, sigma=smooth_sigma)
    strengths_smooth = gaussian_filter1d(strengths, sigma=smooth_sigma)
    sigmas_smooth = gaussian_filter1d(sigmas, sigma=smooth_sigma)

    # --- 绘图 ---
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    plt.subplots_adjust(hspace=0.3)

    # Accuracy
    axs[0].plot(rounds, accs_smooth, label="Accuracy", color="dodgerblue", linewidth=2)
    axs[0].set_ylabel("Accuracy (%)")
    axs[0].grid(True)
    axs[0].legend()

    # Loss
    axs[1].plot(rounds, losses_smooth, label="Loss", color="orange", linewidth=2)
    axs[1].set_ylabel("Loss")
    axs[1].grid(True)
    axs[1].legend()

    # Sigma (DP噪声)
    axs[2].plot(rounds, sigmas_smooth, label="σ (DP noise)", color="purple", linewidth=2)
    axs[2].set_ylabel("σ (DP scale)")
    axs[2].grid(True)
    axs[2].legend()

    # Strength + 聚合区间
    axs[3].plot(rounds, strengths_smooth, label="Strength", color="green", linewidth=2)
    axs[3].set_ylabel("Strength")
    axs[3].set_xlabel("Round")
    axs[3].grid(True)
    axs[3].legend()

    # 背景区块：区分聚合策略
    for i, agg in enumerate(aggs):
        color = (
            "lightblue" if "fedavg" in agg.lower() else
            "gold" if "trimmed" in agg.lower() else
            "lightcoral"
        )
        axs[3].axvspan(i - 0.5, i + 0.5, facecolor=color, alpha=0.25)

    # 保存图像
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[visualize] Training dynamics plot saved to: {save_path}")
