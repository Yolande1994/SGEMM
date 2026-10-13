# -*- coding: utf-8 -*-
"""
GEMM 基准测试数据图生成脚本
用法:
    python plot.py                # 生成全部图表
    python plot.py perf           # 只生成性能图
    python plot.py ncu            # 只生成 NCU 分析图
依赖: matplotlib, numpy
数据来源: test/benchmark_native.txt、test/benchmark_sm80.txt（RTX 5060 Laptop，冷缓存，中位数）
输出: images/ 下的 PNG（150 dpi）
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.edgecolor"] = "#374151"
plt.rcParams["axes.labelcolor"] = "#111827"
plt.rcParams["xtick.color"] = "#374151"
plt.rcParams["ytick.color"] = "#374151"

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(ROOT, "images")
TEST = os.path.join(ROOT, "test")
os.makedirs(IMG, exist_ok=True)

SIZES = [1024, 2048, 3072, 4096, 5120, 6144]

# 数据图收录的 11 个版本（README 主性能图口径）
KERNELS = ["GEMM1", "GEMM2", "GEMM3", "GEMM4", "GEMM6", "GEMM9",
           "GEMM10", "GEMM11", "GEMM11B", "GEMM12", "cuBLAS"]

# 按优化阶段统一配色（冷色=早期，暖色=后期，黑=cuBLAS 基准）
COLORS = {
    "GEMM1":   "#9ca3af",
    "GEMM2":   "#60a5fa",
    "GEMM3":   "#3b82f6",
    "GEMM4":   "#2563eb",
    "GEMM6":   "#0d9488",
    "GEMM9":   "#84cc16",
    "GEMM10":  "#f59e0b",
    "GEMM11":  "#f97316",
    "GEMM11B": "#ea580c",
    "GEMM12":  "#dc2626",
    "cuBLAS":  "#111827",
}
LABELS = {
    "GEMM1": "GEMM1 朴素移植", "GEMM2": "GEMM2 共享内存分块",
    "GEMM3": "GEMM3 一维线程粗化", "GEMM4": "GEMM4 二维线程粗化",
    "GEMM6": "GEMM6 float4 向量化", "GEMM9": "GEMM9 朴素双缓冲",
    "GEMM10": "GEMM10 寄存器流水", "GEMM11": "GEMM11 两级双缓冲",
    "GEMM11B": "GEMM11B 两级+cp.async", "GEMM12": "GEMM12 自定义分块",
    "cuBLAS": "cuBLAS(FP32)",
}

# ---- 中位数耗时（ms），行序与 KERNELS 一致，列序与 SIZES 一致 ----
MS = {
    "native": {
        "GEMM1":   [2.0104, 15.7235, 76.1610, 188.4857, 367.9026, 635.6741],
        "GEMM2":   [1.5264, 11.8947, 46.9620, 110.5193, 213.5812, 368.8877],
        "GEMM3":   [0.5600, 4.1436, 14.8341, 36.5387, 72.1806, 124.3201],
        "GEMM4":   [0.4594, 2.5459, 9.4197, 21.2540, 42.2307, 73.1048],
        "GEMM6":   [0.3788, 2.2943, 8.6957, 20.8843, 38.9292, 67.2306],
        "GEMM9":   [0.3653, 2.3418, 8.6256, 20.5894, 41.2961, 68.8747],
        "GEMM10":  [0.3181, 2.0172, 7.9789, 17.7899, 34.5951, 59.4656],
        "GEMM11":  [0.3021, 1.9737, 7.4459, 17.7999, 35.1413, 61.1089],
        "GEMM11B": [0.2876, 1.9563, 6.7233, 16.3854, 32.1591, 55.0281],
        "GEMM12":  [0.2383, 1.5952, 5.4973, 12.6700, 26.5469, 45.6144],
        "cuBLAS":  [0.2313, 1.8409, 6.5423, 13.9580, 30.9068, 52.4824],
    },
    "sm_80": {
        "GEMM1":   [1.9557, 15.3232, 62.9979, 149.7542, 291.5066, 523.2632],
        "GEMM2":   [1.5282, 11.8989, 46.9589, 108.2935, 213.2735, 368.4566],
        "GEMM3":   [0.7084, 5.1179, 16.6492, 39.7701, 96.7910, 176.0039],
        "GEMM4":   [0.8691, 3.9699, 18.9798, 40.9052, 78.5079, 134.4184],
        "GEMM6":   [0.4261, 2.4222, 8.9226, 19.6720, 39.1843, 68.9637],
        "GEMM9":   [0.4591, 2.8072, 12.4989, 28.8420, 57.0167, 98.0064],
        "GEMM10":  [0.3530, 2.3004, 8.5590, 19.1681, 37.8715, 65.1283],
        "GEMM11":  [0.3038, 2.0208, 7.4101, 17.8354, 34.9440, 59.7437],
        "GEMM11B": [0.2691, 1.8380, 6.7741, 15.9454, 31.7254, 54.5180],
        "GEMM12":  [0.3019, 1.6945, 6.1025, 13.7706, 30.1659, 49.9444],
        "cuBLAS":  [0.2325, 1.7878, 6.6624, 14.3241, 30.6949, 52.4888],
    },
}


def tflops(ms, n):
    """GFLOPs(2MNK) / 耗时(ms) 换算成 TFLOPS：2*n^3 / ms / 1e9"""
    return 2.0 * n ** 3 / ms / 1e9


# ---- 导出结构化数据 CSV ----
with open(os.path.join(TEST, "benchmark.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["arch", "kernel", "size_n", "median_ms", "tflops"])
    for arch in ("native", "sm_80"):
        for k in KERNELS:
            for n, ms in zip(SIZES, MS[arch][k]):
                w.writerow([arch, k, n, ms, round(tflops(ms, n), 3)])


def save(fig, name):
    # 正常绘制，不留裁剪白边
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.16)
    fig.savefig(os.path.join(IMG, name + ".png"), dpi=150)
    plt.close(fig)
    print("生成", name)


# 性能总览图终点右侧的短标签
SHORT = {
    "GEMM1": "GEMM1 朴素", "GEMM2": "GEMM2 共享内存", "GEMM3": "GEMM3 一维粗化",
    "GEMM4": "GEMM4 二维粗化", "GEMM6": "GEMM6 float4", "GEMM9": "GEMM9 双缓冲",
    "GEMM10": "GEMM10 寄存器流水", "GEMM11": "GEMM11 两级缓冲",
    "GEMM11B": "GEMM11B 两级+cp.async", "GEMM12": "GEMM12 自定义分块",
    "cuBLAS": "cuBLAS（FP32）",
}



# ============================================================
# 1. 性能演进图（仿 siboehm：灰底白网格、无图例、版本名标在每条线终点右侧）
# ============================================================
def plot_perf(arch, fname):
    fig, ax = plt.subplots(figsize=(11.5, 7.0), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    x = np.arange(len(SIZES))

    series = {}
    for k in KERNELS:
        vals = [tflops(ms, n) for ms, n in zip(MS[arch][k], SIZES)]
        series[k] = vals
        is_base = k == "cuBLAS"
        ax.plot(x, vals, marker="o", ms=4.2,
                lw=3.0 if is_base else 1.9,
                ls="--" if is_base else "-",
                color=COLORS[k], zorder=5 if is_base else 3)

    # 终点标签：严格按最后一个尺寸点的性能从低到高（图上从下到上）排序，
    # 标签顺序与曲线终点一一对应，引导线不分叉；再用最小间距防止标签重叠
    end_ys = {k: series[k][-1] for k in KERNELS}
    perf_order = sorted(KERNELS, key=lambda k: end_ys[k])
    label_y = {}
    prev = -10.0
    for k in perf_order:
        y = max(end_ys[k], prev + 0.42)
        label_y[k] = y
        prev = y
    xlast = len(SIZES) - 1
    for k in KERNELS:
        ax.plot([xlast, xlast + 0.22], [end_ys[k], label_y[k]],
                color=COLORS[k], lw=0.9, alpha=0.6, zorder=2)
        ax.text(xlast + 0.28, label_y[k], SHORT[k],
                color=COLORS[k], va="center", ha="left", fontsize=11.5)

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SIZES], fontsize=11, rotation=45, ha="right")
    ax.set_xlabel("M = N = K（方阵边长）", fontsize=12.5)
    ax.set_ylabel("TFLOPS（FP32，中位耗时）", fontsize=12.5)
    ax.set_xlim(-0.35, len(SIZES) - 1 + 2.6)
    ax.set_ylim(0, 12.1)
    ax.grid(True, color="white", lw=1.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    arch_title = "-arch=native（sm_120）" if arch == "native" else "-arch=sm_80"
    ax.set_title("手写 GEMM 各版本性能  ·  RTX 5060 Laptop  ·  冷缓存  ·  " + arch_title,
                 fontsize=14, pad=12)
    # 正常绘制，不留裁剪白边
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.18)
    fig.savefig(os.path.join(IMG, fname + ".png"), dpi=150)
    plt.close(fig)
    print("生成", fname)


# ============================================================
# 2. 4096 尺寸各版本占 cuBLAS 百分比，native vs sm_80 分组柱
# ============================================================
def plot_ratio():
    idx = SIZES.index(4096)
    hand = [k for k in KERNELS if k != "cuBLAS"]
    ratios = {}
    for arch in ("native", "sm_80"):
        base = tflops(MS[arch]["cuBLAS"][idx], 4096)
        ratios[arch] = [100.0 * tflops(MS[arch][k][idx], 4096) / base for k in hand]

    fig, ax = plt.subplots(figsize=(11.5, 6.0), dpi=150)
    x = np.arange(len(hand))
    w = 0.38
    b1 = ax.bar(x - w / 2, ratios["native"], w, label="-arch=native (sm_120)",
                color="#2563eb", ec="white")
    b2 = ax.bar(x + w / 2, ratios["sm_80"], w, label="-arch=sm_80",
                color="#f59e0b", ec="white")
    ax.axhline(100, color="#111827", lw=2, ls="--")
    ax.text(len(hand) - 0.5, 101.5, "cuBLAS = 100%", fontsize=11, color="#111827", ha="right")
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width() / 2, h + 1.2, f"{h:.0f}",
                    ha="center", fontsize=9.5, color="#374151")
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("GEMM", "G") for k in hand], fontsize=11)
    ax.set_ylabel("占 cuBLAS 性能百分比（%）", fontsize=13)
    ax.set_ylim(0, 125)
    ax.set_title("各版本相对 cuBLAS 的性能  ·  M=N=K=4096  ·  冷缓存", fontsize=14, pad=12)
    ax.grid(axis="y", ls="--", alpha=0.35)
    ax.legend(fontsize=12, frameon=False, loc="upper left")
    save(fig, "ratio_cublas_4096")


# ============================================================
# 3. NCU 分析（M=N=K=4096，锁定时钟）
# ============================================================
NCU = [
    # 名称, Duration ms, Compute %, Memory %, 寄存器/线程
    ("GEMM1",   276.03, 76.75, 76.75, 32),
    ("GEMM2",   179.23, 81.28, 81.28, 38),
    ("GEMM3",    57.96, 84.27, 84.27, 47),
    ("GEMM4",    34.57, 57.60, 82.05, 96),
    ("GEMM5",    34.91, 57.29, 81.48, 96),
    ("GEMM6",    31.57, 60.58, 89.88, 96),
    ("GEMM6B",   30.86, 57.99, 91.93, 96),
    ("GEMM7",    75.42, 20.70, 94.02, 128),
    ("GEMM7B",   37.67, 41.78, 74.22, 128),
    ("GEMM8",    24.12, 65.64, 54.23, 128),
    ("GEMM9",    33.24, 47.44, 38.82, 128),
    ("GEMM9B",   32.51, 48.65, 38.73, 128),
    ("GEMM10",   28.09, 54.58, 46.18, 155),
    ("GEMM10B",  27.36, 56.09, 46.34, 141),
    ("GEMM11",   27.23, 54.10, 77.84, 167),
    ("GEMM11B",  25.10, 58.87, 83.50, 149),
    ("GEMM12",   19.09, 76.29, 69.57, 247),
    ("cuBLAS",   21.05, 76.58, 65.30, 128),
]


def plot_ncu_throughput():
    names = [r[0] for r in NCU]
    comp = [r[2] for r in NCU]
    mem = [r[3] for r in NCU]
    fig, ax = plt.subplots(figsize=(12.5, 6.0), dpi=150)
    x = np.arange(len(names))
    w = 0.4
    ax.bar(x - w / 2, comp, w, label="Compute Throughput（SM 计算吞吐）", color="#2563eb", ec="white")
    ax.bar(x + w / 2, mem, w, label="Memory Throughput（显存吞吐）", color="#f59e0b", ec="white")
    # 高亮 GEMM7（bank 冲突）与 GEMM12
    ax.axvspan(7 - 0.5, 7 + 0.5, color="#fecaca", alpha=0.45, zorder=0)
    ax.text(7, 97, "GEMM7\nbank 冲突", ha="center", fontsize=10, color="#991b1b")
    ax.axvspan(16 - 0.5, 16 + 0.5, color="#dcfce7", alpha=0.55, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10.5, rotation=35, ha="right")
    ax.set_ylabel("NCU 吞吐利用率（%）", fontsize=13)
    ax.set_ylim(0, 105)
    ax.set_title("NCU 计算/显存吞吐  ·  M=N=K=4096（锁定时钟）", fontsize=14, pad=12)
    ax.grid(axis="y", ls="--", alpha=0.35)
    ax.legend(fontsize=12, frameon=False, loc="upper right")
    save(fig, "ncu_throughput")


def plot_ncu_registers():
    names = [r[0] for r in NCU]
    regs = [r[4] for r in NCU]
    colors = ["#dc2626" if r >= 128 else "#2563eb" for r in regs]
    fig, ax = plt.subplots(figsize=(12.5, 5.8), dpi=150)
    x = np.arange(len(names))
    bars = ax.bar(x, regs, 0.62, color=colors, ec="white")
    for r, v in zip(bars, regs):
        ax.text(r.get_x() + r.get_width() / 2, v + 4, str(v),
                ha="center", fontsize=10, color="#374151")
    ax.axhline(255, color="#7f1d1d", lw=1.5, ls=":")
    ax.text(len(names) - 0.5, 257, "上限 255", fontsize=10.5, color="#7f1d1d", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10.5, rotation=35, ha="right")
    ax.set_ylabel("每线程寄存器数", fontsize=13)
    ax.set_ylim(0, 280)
    ax.set_title("寄存器占用演进（红色 = ≥128，占用率开始下降）  ·  M=N=K=4096",
                 fontsize=14, pad=12)
    ax.grid(axis="y", ls="--", alpha=0.35)
    # 图例说明颜色含义
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#2563eb", label="寄存器 < 128"),
                       Patch(facecolor="#dc2626", label="寄存器 ≥ 128")],
              fontsize=11.5, frameon=False, loc="upper left")
    save(fig, "ncu_registers")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "perf"):
        plot_perf("native", "perf_native")
        plot_perf("sm_80", "perf_sm80")
        plot_ratio()
    if mode in ("all", "ncu"):
        plot_ncu_throughput()
        plot_ncu_registers()
    print("完成，图片输出目录:", IMG)
