# -*- coding: utf-8 -*-
"""
siboehm 风格中文原理图（严格对齐 siboehm.com CUDA-MMM 配图构图）
手绘 xkcd 风格；A 蓝 / B 青 / C 绿 / 线程块紫 / 线程橙。
每张 PNG 底部预留约 0.65 英寸白边，方便裁剪水印。
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch, Polygon

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- 颜色常量
C_INK = "#1f2937"
C_A = "#bfdbfe"; C_A_HI = "#3b82f6"; C_A_DK = "#1d4ed8"
C_B = "#99f6e4"; C_B_HI = "#0d9488"; C_B_DK = "#0f766e"
C_C = "#bbf7d0"; C_C_HI = "#16a34a"; C_C_DK = "#15803d"
C_BLOCK = "#ddd6fe"; C_PURPLE = "#7c3aed"
C_THR = "#fed7aa"; C_ORANGE = "#ea580c"
C_RED = "#dc2626"; C_GRID = "#9ca3af"
C_A_RED = "#fca5a5"  # 第二个线程的红

# ---------------------------------------------------------------- 辅助函数
def save(fig, name):
    """输出 PNG（150 dpi）。"""
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=150, facecolor="white")
    plt.close(fig)
    print("生成", name)

def cell(ax, x, y, w, h, fc="white", ec=C_INK, lw=1.2, hatch=None,
         alpha=1.0, ls="-", zorder=2):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, hatch=hatch,
                           alpha=alpha, ls=ls, zorder=zorder))

def dashed_grid(ax, x0, y0, cols, rows, c, lw=0.7, color=C_GRID):
    """在 (x0,y0) 起画 cols×rows 个边长 c 的虚线网格。"""
    for i in range(1, cols):
        ax.plot([x0 + i*c, x0 + i*c], [y0, y0 + rows*c],
                ls=(0, (3, 3)), color=color, lw=lw, zorder=3)
    for j in range(1, rows):
        ax.plot([x0, x0 + cols*c], [y0 + j*c, y0 + j*c],
                ls=(0, (3, 3)), color=color, lw=lw, zorder=3)

def arrow(ax, p0, p1, color=C_INK, lw=1.6, text=None, fs=11,
          text_xy=None, style="->", ls="-", ms=14):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, color=color, lw=lw,
                                 mutation_scale=ms, ls=ls, zorder=6,
                                 shrinkA=0, shrinkB=0))
    if text:
        tx, ty = text_xy if text_xy else ((p0[0]+p1[0])/2, (p0[1]+p1[1])/2)
        ax.text(tx, ty, text, fontsize=fs, color=color, ha="center",
                va="center", zorder=7)

def brace_h(ax, x0, x1, y, text, color=C_INK, fs=11, side="bottom", lw=1.6):
    """水平大括号（两端带钩）。"""
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y),
        arrowstyle="]-[ , angleA=0, angleB=0",
        color=color, lw=lw, mutation_scale=13, zorder=5))
    ty = y - 0.30 if side == "bottom" else y + 0.30
    ax.text((x0+x1)/2, ty, text, ha="center", va="center",
            fontsize=fs, color=color)

def brace_v(ax, x, y0, y1, text, color=C_INK, fs=11, side="left", lw=1.6):
    """垂直大括号。"""
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1),
        arrowstyle="]-[ , angleA=0, angleB=0",
        color=color, lw=lw, mutation_scale=13, zorder=5))
    tx = x - 0.32 if side == "left" else x + 0.32
    ax.text(tx, (y0+y1)/2, text, ha="center", va="center",
            fontsize=fs, color=color, rotation=90)

def new_fig(w, h, xlim, ylim):
    fig = plt.figure(figsize=(w, h), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    return fig, ax

def xkcd_on():
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]


# ================================================================
# 图 1：GEMM1 三级映射（严格仿 naive-kernel）
# ================================================================
def fig1_hierarchy():
    with plt.xkcd(scale=0.55, length=90):
        xkcd_on()
        fig, ax = new_fig(16.0, 7.0, (0, 16), (0, 7.0))
        ax.text(8.0, 6.65, "GEMM1 朴素版：一个线程算一个输出 —— Grid / Block / Thread 三级映射",
                ha="center", fontsize=14.5)

        # ---------------- GRID
        gx, gy, gs = 1.7, 2.05, 3.05
        cell(ax, gx, gy, gs, gs, fc="white", ec=C_C_HI, lw=2.0)
        dashed_grid(ax, gx, gy, 2, 2, gs/2)
        cell(ax, gx+gs/2, gy+gs/2, gs/2, gs/2, fc=C_BLOCK, ec=C_PURPLE,
             lw=1.8, hatch="////")
        ax.text(gx+gs*0.75, gy+gs*0.72, "Block", ha="center", va="center",
                fontsize=13, color=C_PURPLE)
        ax.text(gx+gs*0.30, gy+gs*0.30, "C", ha="center", va="center", fontsize=16)
        arrow(ax, (gx, gy+gs+0.55), (gx+gs, gy+gs+0.55), color=C_PURPLE, lw=2.0,
              text="blockIdx.x", fs=12, text_xy=(gx+gs/2, gy+gs+0.85))
        arrow(ax, (gx-0.55, gy+gs), (gx-0.55, gy), color=C_PURPLE, lw=2.0,
              text="blockIdx.y", fs=12, text_xy=(gx-0.95, gy+gs/2))
        brace_v(ax, gx-1.18, gy, gy+gs, "gridDim.y", color=C_INK, fs=10.5, side="left")
        ax.text(gx+gs/2, 1.45, "GRID（线程块网格）", ha="center", fontsize=14, weight="bold")
        ax.text(gx+gs/2, 0.78,
                "按输出矩阵 C 的尺寸铺满网格：\n"
                "grid = (N/32, M/32)，每个 block 负责 C 的一块",
                ha="center", va="top", fontsize=10.5, color=C_INK)

        arrow(ax, (gx+gs+0.15, gy+gs/2), (5.75, gy+gs/2),
              color=C_INK, lw=1.8, text="放大 1 个 block", fs=10.5,
              text_xy=(5.25, gy+gs/2-0.75))

        # ---------------- BLOCK
        bx, by, bs = 5.95, 2.05, 3.1
        cell(ax, bx, by, bs, bs, fc="white", ec=C_PURPLE, lw=2.0)
        n = 8
        c = bs/n
        dashed_grid(ax, bx, by, n, n, c)
        brace_h(ax, bx, bx+bs, by+bs+0.25, "blockDim（32×32 个线程）", fs=11,
                color=C_INK, side="top")
        tgx, tgy = bx+3.2*c, by+3.2*c
        cell(ax, tgx, tgy, c, c, fc=C_THR, ec=C_ORANGE, lw=1.8, hatch="////")
        ax.text(tgx+c/2, tgy+c/2, "thread", ha="center", va="center",
                fontsize=10, color=C_ORANGE)
        arrow(ax, (bx, by+bs+0.62), (bx+bs, by+bs+0.62), color=C_ORANGE, lw=1.8,
              text="threadIdx.x", fs=11.5, text_xy=(bx+bs/2, by+bs+0.92))
        arrow(ax, (bx-0.5, by+bs), (bx-0.5, by), color=C_ORANGE, lw=1.8,
              text="threadIdx.y", fs=11.5, text_xy=(bx-0.92, by+bs/2))
        ax.text(bx+bs/2, 1.45, "BLOCK（线程块，32×32）", ha="center",
                fontsize=14, weight="bold")
        ax.text(bx+bs/2, 0.78,
                "一个 block 负责 C 中 32×32 的输出块；\n"
                "块内共 1024 个线程，各自有 (tx, ty) 坐标",
                ha="center", va="top", fontsize=10.5, color=C_INK)

        arrow(ax, (bx+bs+0.15, by+bs/2), (10.35, by+bs/2),
              color=C_INK, lw=1.8, text="放大 1 个线程", fs=10.5,
              text_xy=(10.0, by+bs/2+0.45))

        # ---------------- THREAD
        rx, ry = 10.55, 3.2
        cell(ax, rx, ry, 0.55, 0.55, fc=C_C, ec=C_C_HI, lw=1.8, hatch="////")
        ax.text(rx+0.275, ry-0.35, "结果", ha="center", fontsize=10.5, color=C_C_DK)
        ax.text(11.45, 3.5, "=", ha="center", fontsize=21)
        ax0, ay0, aw, ah = 11.85, 4.25, 2.75, 0.5
        cell(ax, ax0, ay0, aw, ah, fc=C_A, ec=C_A_DK, lw=1.5, hatch="////")
        ax.text(ax0+aw/2, ay0+ah+0.28, "A 的一行", ha="center", fontsize=11.5, color=C_A_DK)
        brace_h(ax, ax0, ax0+aw, ay0-0.18, "K", fs=12, color=C_INK)
        bxx, byy, bw, bh = 14.70, 2.05, 0.5, 2.7
        cell(ax, bxx, byy, bw, bh, fc=C_B, ec=C_B_DK, lw=1.5, hatch="\\\\\\\\")
        ax.text(bxx+bw/2, byy+bh+0.28, "B 的一列", ha="center", fontsize=11.5,
                color=C_B_DK)
        brace_v(ax, bxx-0.18, byy, byy+bh, "K", fs=12, color=C_INK, side="left")
        ax.text(13.35, 3.05, "点积\n(dot product)", ha="center", va="center",
                fontsize=11, color=C_INK, rotation=28)
        ax.text(12.9, 1.45, "THREAD（一个线程）", ha="center",
                fontsize=14, weight="bold")
        ax.text(12.9, 0.78,
                "线程沿 K 串行做 K 次乘加：\n"
                "C[row,col] = Σ_k A[row,k] · B[k,col]",
                ha="center", va="top", fontsize=10.5, color=C_INK)
        save(fig, "sib_01_thread_hierarchy")


# ================================================================
# 图 2：GEMM1 访存模式（严格仿 naive_kernel_mem_access）
# ================================================================
def fig2_naive_mem():
    with plt.xkcd(scale=0.55, length=90):
        xkcd_on()
        fig, ax = new_fig(14.0, 10.6, (0, 14), (0, 10.6))
        ax.text(7.0, 10.3, "GEMM1 的访存：A 沿行读地址连续，B 沿列读跨步 N，缓存行被大量浪费",
                ha="center", fontsize=14)

        # ---------- B 矩阵（右上）
        Bx, By, Bw, Bh = 5.6, 5.7, 3.9, 3.8
        cell(ax, Bx, By, Bw, Bh, fc="white", ec=C_INK, lw=1.6)
        cell(ax, Bx+0.12, By+0.12, 0.55, Bh-0.24, fc=C_A_RED, ec=C_RED,
             lw=1.2, hatch="\\\\\\\\", alpha=0.75)
        cell(ax, Bx+0.72, By+0.12, 0.55, Bh-0.24, fc=C_B, ec=C_B_DK,
             lw=1.2, hatch="\\\\\\\\", alpha=0.75)
        ax.text(Bx+0.28, By+Bh-0.42, "B", ha="center", fontsize=17)
        ax.text(Bx+Bw/2+0.25, By+Bh*0.52,
                "地址不连续：\n"
                "每读一个元素跨过 N 个 float\n"
                "一条缓存行只用上 1 个数",
                ha="center", va="center", fontsize=11, color=C_B_DK)
        arrow(ax, (Bx+0.99, By+Bh-0.35), (Bx+0.99, By+0.35), color=C_B_DK, lw=1.6,
              ms=12)
        ax.text(Bx-0.25, By+Bh+0.32, "两个线程读相邻两列", ha="left", va="center",
                fontsize=11, color=C_INK)

        # ---------- A 矩阵（左下）
        Ax, Ay, Aw, Ah = 0.5, 1.5, 4.3, 3.1
        cell(ax, Ax, Ay, Aw, Ah, fc=C_A, ec=C_A_DK, lw=1.2, hatch="////", alpha=0.35)
        cell(ax, Ax, Ay+Ah-0.78, Aw, 0.40, fc=C_A_RED, ec=C_RED, lw=1.2,
             hatch="////", alpha=0.7)
        cell(ax, Ax, Ay+Ah-1.24, Aw, 0.40, fc=C_B, ec=C_B_DK, lw=1.2,
             hatch="////", alpha=0.7)
        ax.text(Ax+0.3, Ay+Ah-0.42, "A", ha="center", fontsize=17)
        ax.text(Ax+Aw/2, Ay+Ah*0.42,
                "地址连续：\n"
                "沿行读取，缓存行里的 4 个数都能用上",
                ha="center", va="center", fontsize=11, color=C_A_DK)
        arrow(ax, (Ax+0.9, Ay+Ah-1.55), (Ax+Aw-0.8, Ay+Ah-1.55),
              color=C_INK, lw=1.6, ms=12)
        ax.text(Ax+Aw/2, Ay+Ah+0.32, "每个线程读不同的行", ha="center", fontsize=11)

        # ---------- C 矩阵（中下）
        Cx, Cy, Cw, Ch = 5.6, 1.5, 3.9, 3.1
        cell(ax, Cx, Cy, Cw, Ch, fc="white", ec=C_INK, lw=1.6)
        dashed_grid(ax, Cx, Cy, 6, 6, Cw/6, color="#d1d5db")
        g = Cw/6
        cell(ax, Cx, Cy+Ch-g, g, g, fc=C_A_RED, ec=C_RED, lw=1.4, hatch="////")
        cell(ax, Cx+g, Cy+Ch-g, g, g, fc=C_B, ec=C_B_DK, lw=1.4, hatch="\\\\\\\\")
        cell(ax, Cx, Cy+Ch-2*g, 2*g, 2*g, fc="none", ec=C_PURPLE, lw=1.8)
        ax.text(Cx+Cw/2, Cy-0.35, "C", ha="center", fontsize=17)

        # ---------- BLOCK 放大（右）
        Xx, Xy, Xw, Xh = 10.6, 2.0, 2.3, 2.3
        cell(ax, Xx, Xy, Xw, Xh, fc="white", ec=C_PURPLE, lw=1.8)
        dashed_grid(ax, Xx, Xy, 4, 4, Xw/4, color="#d1d5db")
        cc = Xw/4
        cell(ax, Xx, Xy+Xh-cc, cc, cc, fc=C_A_RED, ec=C_RED, lw=1.4, hatch="////")
        cell(ax, Xx+cc, Xy+Xh-cc, cc, cc, fc=C_B, ec=C_B_DK, lw=1.4, hatch="\\\\\\\\")
        ax.text(Xx+cc/2, Xy+Xh-cc/2, "A", ha="center", va="center",
                fontsize=12, color=C_RED, weight="bold")
        ax.text(Xx+cc*1.5, Xy+Xh-cc/2, "B", ha="center", va="center",
                fontsize=12, color=C_B_DK, weight="bold")
        ax.text(Xx+Xw/2, Xy-0.32, "BLOCK（红=线程 A(0,0)，青=线程 B(0,1)）",
                ha="center", fontsize=11.5, weight="bold")

        # C 与 BLOCK 的放大关系
        ax.add_patch(FancyArrowPatch((Cx+Cw+0.1, Cy+Ch-g/2),
            (Xx-0.05, Xy+Xh/2),
            connectionstyle="arc3,rad=0.15", color=C_PURPLE, lw=1.1,
            arrowstyle="->", ls=(0, (4, 3)), zorder=5))
        ax.text(10.35, 3.85, "放大", ha="center", fontsize=10, color=C_PURPLE)

        # 红线：线程 A → A 红行（走 C 与 B 之间的空隙）
        for p0, p1 in [
            ((Xx, Xy+Xh-cc/2), (Xx, 4.85)),
            ((Xx, 4.85), (Ax+Aw+0.25, 4.85)),
            ((Ax+Aw+0.25, 4.85), (Ax+Aw, Ay+Ah-0.58)),
        ]:
            arrow(ax, p0, p1, color=C_RED, lw=1.2, ms=11)
        # 青线：线程 B → B 青列（走上侧空隙）
        for p0, p1 in [
            ((Xx+cc, Xy+Xh-cc/2), (Xx+cc, 5.28)),
            ((Xx+cc, 5.28), (Bx+0.99, 5.28)),
            ((Bx+0.99, 5.28), (Bx+0.99, By+0.15)),
        ]:
            arrow(ax, p0, p1, color=C_B_DK, lw=1.2, ms=11)
        ax.text(7.4, 5.08, "线程 A：A 的第 0 行", ha="left", fontsize=10, color=C_RED)
        ax.text(7.5, 5.52, "线程 B：B 的第 1 列", ha="left", fontsize=10, color=C_B_DK)

        ax.text(7.0, 0.78,
                "A、B、C 都按行主序存储，最后一个下标连续（步长 1）。\n"
                "线程 A(0,0) 读 A 的第 0 行、B 的第 0 列；线程 B(0,1) 读 A 的第 0 行、B 的第 1 列。\n"
                "B 沿列方向跨步 N，单个线程的访问序列不连续，缓存行利用率低——这是 GEMM1 的第一个性能问题。",
                ha="center", va="top", fontsize=11, color=C_INK)
        save(fig, "sib_02_naive_mem")


# ================================================================
# 图 3：warp 合并访存（严格仿 GMEM_coalescing）
# ================================================================
def fig3_coalescing():
    with plt.xkcd(scale=0.55, length=90):
        xkcd_on()
        fig, ax = new_fig(14.0, 7.6, (0, 14), (0, 7.6))
        ax.text(7.0, 7.25, "合并访存（Coalescing）：一个 warp 内地址连续，32 次访存才能被合并成少数几条指令",
                ha="center", fontsize=13.5)

        # 线程排：16 个代表线程（warp0 画 0-7，warp1 画 8-15）
        c = 0.62
        x0 = 2.6
        ty = 4.5
        for i in range(16):
            color = C_PURPLE if i < 8 else C_ORANGE
            cell(ax, x0+i*c, ty, c*0.92, c*0.92, fc="white", ec=color, lw=1.4)
            ax.text(x0+i*c+c*0.46, ty+c*0.46, str(i), ha="center",
                    va="center", fontsize=11)
        # 两个 warp 的大括号（画在格子上方）
        for (i0, i1, color, lab) in [
            (0, 8, C_PURPLE, "WARP 0"), (8, 16, C_ORANGE, "WARP 1")
        ]:
            ax.add_patch(FancyArrowPatch((x0+i0*c-0.15, ty+c+0.20),
                (x0+i1*c-0.10, ty+c+0.20),
                arrowstyle="]-[ , angleA=0, angleB=0", color=color, lw=1.8,
                mutation_scale=13, zorder=4))
            ax.text((x0+i0*c+x0+(i1-1)*c)/2, ty+c+0.52, lab, ha="center",
                    fontsize=12, color=color, weight="bold")

        # 内存行：16 个格子，与线程垂直对齐
        my = 2.1
        cell(ax, x0-0.62, my-0.02, 0.62, 0.62, fc="white", ec=C_INK, lw=1.0)
        ax.text(x0-0.31, my+0.29, "显存", ha="center", va="center",
                fontsize=11, rotation=90)
        for i in range(16):
            fc = C_BLOCK if i < 8 else C_THR
            ec = C_PURPLE if i < 8 else C_ORANGE
            cell(ax, x0+i*c, my, c*0.92, c*0.92, fc=fc, ec=ec, lw=1.0)
        # 垂直连线
        for i in range(16):
            col = C_PURPLE if i < 8 else C_ORANGE
            arrow(ax, (x0+i*c+c*0.46, ty), (x0+i*c+c*0.46, my+c*0.92),
                  color=col, lw=0.8, ms=8)
        # 每 4 格一条事务（小括号）
        for g in range(4):
            xs = x0+g*4*c
            xe = x0+(g*4+4)*c
            ax.add_patch(FancyArrowPatch((xs, my-0.18), (xe-0.05, my-0.18),
                arrowstyle="]-[ , angleA=0, angleB=0", color=C_INK, lw=1.3,
                mutation_scale=10, zorder=4))
        ax.text(x0+8*c, my-0.58, "每 4 个连续 float（16B）合并为 1 条访存指令",
                ha="center", fontsize=11.5)
        # 每个 warp 一条 32B 大括号（画在内存上方）
        for (i0, i1, color) in [
            (0, 8, C_PURPLE), (8, 16, C_ORANGE)
        ]:
            ax.add_patch(FancyArrowPatch((x0+i0*c-0.15, my+c+0.45),
                (x0+i1*c-0.10, my+c+0.45),
                arrowstyle="]-[ , angleA=0, angleB=0", color=color, lw=1.6,
                mutation_scale=12, zorder=4))
        ax.text(7.0, 6.85,
                "同一 warp 的 32 个线程访问连续地址时，32 个 float（128B）合并成 1 条访存事务；\n"
                "若地址跨步（如沿列读 B，步长为 N），每条缓存行只用 1 个数，指令数和流量成倍放大。\n"
                "（真实 warp 为 32 个线程，图中每个 warp 只画 8 个代表）",
                ha="center", va="top", fontsize=11.5)
        save(fig, "sib_03_coalescing")


# ================================================================
# 图 4：GEMM2 共享内存分块（严格仿 cache-blocking）
# ================================================================
def fig4_cache_blocking():
    with plt.xkcd(scale=0.55, length=90):
        xkcd_on()
        fig, ax = new_fig(13.5, 9.2, (0, 13.5), (0, 9.2))
        ax.text(6.75, 8.9, "GEMM2：沿 K 把 A、B 的 32×32 小块搬进共享内存，块内线程共享，全局访存降 32 倍",
                ha="center", fontsize=13.5)

        # B 矩阵（右上）
        Bx, By, Bw, Bh = 5.2, 5.55, 3.3, 3.05
        cell(ax, Bx, By, Bw, Bh, fc="white", ec=C_INK, lw=1.5)
        cell(ax, Bx+1.15, By, 0.55, Bh, fc=C_B, ec=C_B_DK, lw=1.0,
             hatch="\\\\\\\\", alpha=0.40)
        cell(ax, Bx+1.15, By+1.05, 0.55, 0.85, fc=C_B_HI, ec=C_B_DK, lw=1.5,
             hatch="\\\\\\\\")
        ax.text(Bx+0.28, By+Bh-0.42, "B", ha="center", fontsize=17)
        ax.text(Bx+Bw+0.28, By+Bh/2, "N", ha="left", va="center",
                fontsize=13, color=C_GRID)
        brace_v(ax, Bx+1.95, By+1.05, By+1.9, "32", fs=11, color=C_INK, side="right")
        brace_h(ax, Bx+1.15, Bx+1.7, By+2.02, "32", fs=11, color=C_INK, side="top")
        arrow(ax, (Bx+1.42, By+0.92), (Bx+1.42, By+0.28), color=C_INK, lw=1.6)
        ax.text(Bx+0.5, By+1.5, "Bs", fontsize=13, color=C_B_DK, weight="bold")

        # A 矩阵（左下）
        Ax, Ay, Aw, Ah = 0.5, 2.45, 4.0, 2.2
        cell(ax, Ax, Ay, Aw, Ah, fc="white", ec=C_INK, lw=1.5)
        cell(ax, Ax, Ay+0.72, Aw, 0.8, fc=C_A, ec=C_A_DK, lw=1.0,
             hatch="////", alpha=0.40)
        cell(ax, Ax+0.8, Ay+0.72, 0.85, 0.8, fc=C_A_HI, ec=C_A_DK, lw=1.5,
             hatch="////")
        ax.text(Ax+0.3, Ay+Ah-0.4, "A", ha="center", fontsize=17)
        ax.text(Ax+Aw/2, Ay+Ah+0.25, "K", ha="center", fontsize=13, color=C_GRID)
        ax.text(Ax-0.3, Ay+Ah/2, "M", ha="center", fontsize=13, color=C_GRID)
        brace_v(ax, Ax+0.62, Ay+0.72, Ay+1.52, "32", fs=11, color=C_INK, side="left")
        brace_h(ax, Ax+0.8, Ax+1.65, Ay+0.58, "32", fs=11, color=C_INK)
        arrow(ax, (Ax+1.8, Ay+1.12), (Ax+3.2, Ay+1.12), color=C_INK, lw=1.6)
        ax.text(Ax+1.05, Ay+1.08, "As", fontsize=13, color=C_A_DK, weight="bold")

        # C 矩阵（中下）
        Cx, Cy, Cw, Ch = 5.2, 2.0, 3.6, 2.4
        cell(ax, Cx, Cy, Cw, Ch, fc="white", ec=C_INK, lw=1.5)
        cell(ax, Cx+0.55, Cy+0.95, 0.85, 0.85, fc=C_C, ec=C_C_DK, lw=1.5,
             hatch="////")
        ax.text(Cx+0.3, Cy+Ch-0.4, "C", ha="center", fontsize=17)
        brace_h(ax, Cx+0.55, Cx+1.4, Cy+0.82, "32", fs=11, color=C_INK)
        brace_v(ax, Cx+0.37, Cy+0.95, Cy+1.8, "32", fs=11, color=C_INK, side="left")
        ax.plot([Ax+Aw, Cx+0.55], [Ay+1.12, Cy+1.8], color=C_GRID, lw=0.9)
        ax.plot([Bx+1.42, Cx+1.4], [By, Cy+1.8], color=C_GRID, lw=0.9)

        ax.text(9.35, 8.55,
                "外层循环：tile 沿 K 推进，步长 BK = 32\n\n"
                "1. 块内 1024 个线程协作，把 As、Bs\n"
                "   搬到共享内存（SMEM）\n"
                "2. 线程同步栅栏，等大家都搬完\n"
                "3. 32×32 个线程从 SMEM 读数、算 C\n"
                "4. 再次同步，进入下一个 tile",
                ha="left", va="top", fontsize=11.5, color=C_INK)
        ax.text(5.0, 1.25,
                "As 中每个元素被 32 个线程共用，Bs 中每个元素被 32 个线程共用；\n"
                "全局内存总访问量从 O(MNK) 降到 O(MNK/32)，反复使用发生在片上共享内存里。",
                ha="center", va="top", fontsize=11)
        save(fig, "sib_04_cache_blocking")


def _outer_panel(ax, BN):
    """blocktiling 图的上半部分：outer loop（B 右上、A 左下、C 右下）。
    BN=32 对应 GEMM3，BN=128 对应 GEMM4。BM=128、BK=32 固定。"""
    ax.add_patch(FancyBboxPatch((0.15, 6.05), 13.2, 4.8,
        boxstyle="round,pad=0.02,rounding_size=0.15", fc="none",
        ec=C_GRID, lw=1.3, ls=(0, (4, 3))))
    ax.text(0.4, 10.62, "outer loop（外层循环）：tile 沿 K 移动", fontsize=12, color=C_GRID)

    # B 大矩阵（右上）
    Bx, By, Bw, Bh = 6.8, 8.45, 3.2, 2.15
    cell(ax, Bx, By, Bw, Bh, fc="white", ec=C_INK, lw=1.5)
    lane_w = 0.7 if BN == 32 else 3.0
    lane_x = Bx + 1.15 if BN == 32 else Bx + 0.1
    cell(ax, lane_x, By+0.25, lane_w, Bh-0.5, fc=C_B, ec=C_B_DK, lw=0.9,
         hatch="\\\\\\\\", alpha=0.35)
    chk_h = 0.62
    cell(ax, lane_x, By+Bh/2-chk_h/2, lane_w, chk_h, fc=C_B_HI, ec=C_B_DK,
         lw=1.5, hatch="\\\\\\\\")
    ax.text(Bx+0.22, By+Bh-0.35, "B", fontsize=15)
    ax.text(Bx+Bw+0.18, By+Bh/2, "N", fontsize=12, color=C_GRID,
            ha="left", va="center")
    brace_v(ax, Bx-0.18, By+0.25, By+Bh-0.25, "K", fs=11, color=C_GRID)
    brace_h(ax, lane_x, lane_x+lane_w, By+Bh/2+chk_h/2+0.1, "BK", fs=10.5,
            color=C_INK, side="top")
    ax.text(lane_x+lane_w/2, By+0.18, "BN", fontsize=10.5, color=C_INK, ha="center")
    arrow(ax, (lane_x+lane_w/2, By+Bh/2-chk_h/2-0.15),
          (lane_x+lane_w/2, By+0.42), color=C_INK, lw=1.5)

    # C 输出块（右下，与 A 顶部对齐）
    Cw = 0.9 if BN == 32 else 3.0
    Cx, Cy, Ch = 6.8, 6.4, 1.7
    cell(ax, Cx, Cy, Cw, Ch, fc="white", ec=C_INK, lw=1.5)
    cell(ax, Cx, Cy, Cw, Ch, fc=C_C, ec=C_C_DK, lw=1.5,
         hatch="////", alpha=0.55)
    ax.text(Cx+Cw+0.18, Cy+Ch-0.25, "C", fontsize=15)
    brace_v(ax, Cx-0.16, Cy, Cy+Ch, "BM", fs=10.5, color=C_INK)
    ax.text(Cx+Cw/2, Cy+0.12, "BN", fontsize=10.5, color=C_INK, ha="center")

    # A 大矩阵（左下，与 C 中心对齐）
    Ax, Ay, Aw, Ah = 0.5, 6.5, 3.9, 1.5
    cell(ax, Ax, Ay, Aw, Ah, fc="white", ec=C_INK, lw=1.5)
    cell(ax, Ax+0.35, Ay+0.25, 3.1, 0.55, fc=C_A, ec=C_A_DK, lw=0.9,
         hatch="////", alpha=0.35)
    cell(ax, Ax+0.35, Ay+0.25, 0.75, 0.55, fc=C_A_HI, ec=C_A_DK, lw=1.5,
         hatch="////")
    ax.text(Ax+Aw-0.25, Ay+0.18, "A", fontsize=15, ha="right")
    brace_v(ax, Ax+0.22, Ay+0.25, Ay+0.8, "BM", fs=10.5, color=C_INK)
    brace_h(ax, Ax+0.35, Ax+1.1, Ay+0.13, "BK", fs=10.5, color=C_INK)
    arrow(ax, (Ax+1.2, Ay+0.52), (Ax+3.3, Ay+0.52), color=C_INK, lw=1.5)
    ax.text(Ax+Aw+0.2, Ay+Ah/2, "K", ha="left", va="center", fontsize=11, color=C_GRID)

    # 参考线：As→C、Bs→C
    ax.plot([Ax+0.35, Cx], [Ay+0.8, Cy+Ch], color=C_GRID, lw=0.9)
    ax.plot([lane_x+lane_w/2, Cx+Cw/2], [By, Cy+Ch], color=C_GRID, lw=0.9)
    ax.text(10.4, 9.5, "每个 tile 先缓存到\n共享内存（SMEM）",
            ha="left", va="center", fontsize=11.5, color=C_GRID)
    return Cx, Cy, Cw, Ch


# ================================================================
# 图 5：GEMM3 一维线程粗化（严格仿 kernel_4_1D）
# ================================================================
def fig5_1d_blocktiling():
    with plt.xkcd(scale=0.55, length=90):
        xkcd_on()
        fig, ax = new_fig(13.5, 11.5, (0, 13.5), (0, 11.5))
        ax.text(6.75, 11.2, "GEMM3 一维线程粗化：BM=128、BN=32、BK=32，每个线程算 TM=8 行 × 1 列",
                ha="center", fontsize=13.5)
        _outer_panel(ax, 32)

        ax.add_patch(FancyBboxPatch((0.15, 0.8), 13.2, 5.0,
            boxstyle="round,pad=0.02,rounding_size=0.15", fc="none",
            ec=C_GRID, lw=1.3, ls=(0, (4, 3))))
        ax.text(0.4, 5.55, "inner loops（内层循环）：块内线程分工 + dotIdx 沿 K 累加",
                fontsize=12, color=C_GRID)

        # As（BK×BM 竖条）
        Asx, Asy, Asw, Ash = 3.2, 1.4, 1.3, 3.4
        cell(ax, Asx, Asy, Asw, Ash, fc="white", ec=C_A_DK, lw=1.6)
        dashed_grid(ax, Asx, Asy, 4, 16, Asw/4, color="#d1d5db")
        cell(ax, Asx+0.06, Asy+Ash-0.95, Asw-0.12, 0.8, fc="none", ec=C_ORANGE, lw=1.8)
        ax.text(Asx+Asw/2, Asy-0.28, "As（BK×BM）", ha="center", fontsize=12, color=C_A_DK)
        brace_v(ax, Asx-0.22, Asy, Asy+Ash, "BM", fs=11, color=C_INK)
        brace_h(ax, Asx, Asx+Asw, Asy+Ash+0.12, "BK", fs=11, color=C_INK, side="top")
        ax.text(Asx+Asw+0.18, Asy+Ash-0.55, "TM", ha="left", fontsize=10.5, color=C_ORANGE)
        arrow(ax, (Asx+0.25, Asy+Ash-0.05), (Asx+0.25, Asy+0.3),
              color=C_A_DK, lw=1.4)
        ax.text(Asx+0.35, Asy+0.12, "dotIdx", ha="left", fontsize=10.5, color=C_A_DK)

        # Bs（BK×BN 正方形）
        Bsx, Bsy, Bsw, Bsh = 5.3, 4.15, 1.1, 0.8
        cell(ax, Bsx, Bsy, Bsw, Bsh, fc="white", ec=C_B_DK, lw=1.6)
        dashed_grid(ax, Bsx, Bsy, 4, 3, Bsw/4, color="#d1d5db")
        ax.text(Bsx+Bsw/2, Bsy+Bsh+0.18, "Bs（BK×BN）", ha="center", fontsize=12, color=C_B_DK)
        brace_v(ax, Bsx-0.18, Bsy, Bsy+Bsh, "BK", fs=10.5, color=C_INK)
        arrow(ax, (Bsx+Bsw+0.12, Bsy-0.02), (Bsx+Bsw+0.12, Bsy-0.4), color=C_B_DK, lw=1.4)
        ax.text(Bsx+Bsw+0.22, Bsy-0.48, "dotIdx", ha="left", fontsize=10.5, color=C_B_DK)

        # C 块（BM×BN 竖长）：16 个 warp 横条
        Cx, Cy, Cw, Ch = 5.3, 1.4, 1.1, 2.7
        cell(ax, Cx, Cy, Cw, Ch, fc="white", ec=C_C_DK, lw=1.6)
        for r in range(16):
            ax.plot([Cx, Cx+Cw], [Cy+Ch*(r+1)/16, Cy+Ch*(r+1)/16],
                    color=C_GRID, lw=0.6, ls=(0, (2, 2)))
        hr = 6
        cell(ax, Cx, Cy+Ch*hr/16, Cw, Ch/16, fc=C_THR, ec=C_ORANGE,
             lw=1.6, hatch="////")
        ax.text(Cx+Cw/2, Cy+Ch*(hr+0.5)/16, "warp", ha="center", va="center",
                fontsize=9, color=C_ORANGE)
        ax.text(Cx+Cw+0.2, Cy+Ch/2, "C 块（BM×BN）", ha="left", va="center",
                rotation=90, fontsize=11.5, color=C_C_DK)
        brace_v(ax, Cx-0.18, Cy, Cy+Ch, "BM", fs=11, color=C_INK)
        ax.text(Cx+Cw/2, Cy+0.1, "BN", fontsize=10.5, color=C_INK, ha="center")

        ax.text(6.95, 4.6,
                "一个 warp 32 个线程\n"
                "覆盖 BN=32 列（每线程 1 列），\n"
                "每个线程沿 M 连续算 TM=8 个结果",
                ha="left", va="center", fontsize=11.5)
        ax.text(6.95, 2.6,
                "块内 512 个线程（32×16）：\n"
                "threadCol = threadIdx.x（0..31）\n"
                "threadRow = threadIdx.y（0..15）\n"
                "每线程 8 个累加器\n"
                "A 元素在线程间复用 8 次",
                ha="left", va="center", fontsize=11.5)
        ax.text(6.75, 1.0,
                "计算访存比：GEMM2 每 1 个结果读 2 个 SMEM 元素（1:2），GEMM3 每 8 个结果共读 9 个元素（8:9）",
                ha="center", va="center", fontsize=11)
        save(fig, "sib_05_1d_blocktiling")


# ================================================================
# 图 6：GEMM4 二维线程粗化（严格仿 kernel_5_2D）
# ================================================================
def fig6_2d_blocktiling():
    with plt.xkcd(scale=0.55, length=90):
        xkcd_on()
        fig, ax = new_fig(13.5, 11.5, (0, 13.5), (0, 11.5))
        ax.text(6.75, 11.2, "GEMM4 二维线程粗化：BM=BN=128、BK=32，每个线程算 TM×TN = 8×8 个结果",
                ha="center", fontsize=13.5)
        _outer_panel(ax, 128)

        ax.add_patch(FancyBboxPatch((0.15, 0.8), 13.2, 5.0,
            boxstyle="round,pad=0.02,rounding_size=0.15", fc="none",
            ec=C_GRID, lw=1.3, ls=(0, (4, 3))))
        ax.text(0.4, 5.55, "inner loops（内层循环）：块内线程分工 + dotIdx 沿 K 累加",
                fontsize=12, color=C_GRID)

        # As（BK×BM 竖条）
        Asx, Asy, Asw, Ash = 3.0, 1.4, 1.3, 3.4
        cell(ax, Asx, Asy, Asw, Ash, fc="white", ec=C_A_DK, lw=1.6)
        dashed_grid(ax, Asx, Asy, 4, 16, Asw/4, color="#d1d5db")
        cell(ax, Asx+0.06, Asy+Ash-0.95, Asw-0.12, 0.8, fc="none", ec=C_ORANGE, lw=1.8)
        ax.text(Asx+Asw/2, Asy-0.28, "As（BK×BM）", ha="center", fontsize=12, color=C_A_DK)
        brace_v(ax, Asx-0.22, Asy, Asy+Ash, "BM", fs=11, color=C_INK)
        brace_h(ax, Asx, Asx+Asw, Asy+Ash+0.12, "BK", fs=11, color=C_INK, side="top")
        ax.text(Asx+Asw+0.18, Asy+Ash-0.55, "TM", ha="left", fontsize=10.5, color=C_ORANGE)
        arrow(ax, (Asx+0.25, Asy+Ash-0.05), (Asx+0.25, Asy+0.3),
              color=C_A_DK, lw=1.4)
        ax.text(Asx+0.35, Asy+0.12, "dotIdx", ha="left", fontsize=10.5, color=C_A_DK)

        # Bs（BK×BN 横条）
        Bsx, Bsy, Bsw, Bsh = 5.3, 4.15, 3.2, 0.8
        cell(ax, Bsx, Bsy, Bsw, Bsh, fc="white", ec=C_B_DK, lw=1.6)
        dashed_grid(ax, Bsx, Bsy, 16, 4, Bsw/16, color="#d1d5db")
        cell(ax, Bsx+Bsw-0.45, Bsy+0.05, 0.35, Bsh-0.1, fc="none", ec=C_ORANGE, lw=1.8)
        ax.text(Bsx+Bsw/2, Bsy+Bsh+0.18, "Bs（BK×BN）", ha="center", fontsize=12, color=C_B_DK)
        brace_v(ax, Bsx-0.2, Bsy, Bsy+Bsh, "BK", fs=10.5, color=C_INK)
        ax.text(Bsx+Bsw-0.28, Bsy+Bsh/2, "TN", ha="center", fontsize=10, color=C_ORANGE)

        # C 块（BM×BN 正方形），4×4 tile 示意网格，高亮 2×2 线程块
        Cx, Cy, Cw, Ch = 5.3, 1.4, 3.2, 2.7
        cell(ax, Cx, Cy, Cw, Ch, fc="white", ec=C_C_DK, lw=1.6)
        gx_, gy_ = Cw/4, Ch/4
        for i in range(1, 4):
            ax.plot([Cx+i*gx_, Cx+i*gx_], [Cy, Cy+Ch], color=C_GRID, lw=0.8, ls=(0, (3, 3)))
            ax.plot([Cx, Cx+Cw], [Cy+i*gy_, Cy+i*gy_], color=C_GRID, lw=0.8, ls=(0, (3, 3)))
        hx, hy = Cx+gx_, Cy+Ch-2*gy_
        cell(ax, hx, hy, 2*gx_, 2*gy_, fc=C_THR, ec=C_ORANGE, lw=2.0,
             hatch="////", alpha=0.75)
        cell(ax, hx, hy, gx_, gy_, fc="none", ec=C_RED, lw=1.6)
        cell(ax, hx+gx_, hy, gx_, gy_, fc="none", ec=C_B_DK, lw=1.6)
        ax.text(hx+gx_/2, hy+gy_/2, "A", ha="center", fontsize=11, color=C_RED, weight="bold")
        ax.text(hx+1.5*gx_, hy+gy_/2, "B", ha="center", fontsize=11, color=C_B_DK, weight="bold")
        ax.text(Cx+Cw/2, Cy+0.1, "BN", fontsize=10.5, color=C_INK, ha="center")
        ax.text(Cx+Cw+0.2, Cy+Ch/2, "C 块（BM×BN）", ha="left", va="center",
                rotation=90, fontsize=11.5, color=C_C_DK)
        brace_v(ax, Cx-0.2, Cy+gy_, Cy+3*gy_, "TM", fs=10.5, color=C_INK)
        ax.text(Cx+Cw-0.08, Cy+Ch-2.55*gy_, "TN", ha="right", fontsize=10.5, color=C_INK)
        ax.text(Cx+Cw+0.2, Cy+0.65, "红 A、青 B 是\n相邻两个线程",
                ha="left", va="center", fontsize=10)

        ax.text(9.0, 3.1,
                "块内 256 个线程（16×16）：\n"
                "threadCol = threadIdx.x（0..15）\n"
                "threadRow = threadIdx.y（0..15）\n"
                "每线程持有 64 个累加器（8×8）\n"
                "A 元素横向复用 8 次、\n"
                "B 元素纵向复用 8 次",
                ha="left", va="center", fontsize=11.5)
        ax.text(6.75, 1.0,
                "计算访存比：64 个结果只从 SMEM 读 16 个元素（8 个 A + 8 个 B），即 64:16",
                ha="center", va="center", fontsize=11)
        save(fig, "sib_06_2d_blocktiling")


# ================================================================
# 图 7：协作加载（严格仿 kernel_5_GMEM_loading）
# ================================================================
def fig7_gmem_loading():
    with plt.xkcd(scale=0.55, length=90):
        xkcd_on()
        fig, ax = new_fig(10.5, 9.0, (0, 10.5), (0, 9.0))
        ax.text(5.25, 8.7, "协作加载：把块内线程平铺成一维，每一步各取一个元素，跨步跳着把 As 填满",
                ha="center", fontsize=13.5)

        # As 竖网格（BK 宽 × BM 高）
        Asx, Asy, Asw, Ash = 3.9, 1.5, 2.3, 6.4
        cell(ax, Asx, Asy, Asw, Ash, fc="white", ec=C_A_DK, lw=1.8)
        cols, rows = 4, 16
        cw, ch = Asw/cols, Ash/rows
        dashed_grid(ax, Asx, Asy, cols, rows, cw, color="#d1d5db")
        # 本步加载的一条（横跨整个 BK 宽度，ceil(BK*BM/blockDim) 个元素；示意 4 个）
        for i in range(cols):
            cell(ax, Asx+i*cw, Asy+Ash-2*ch, cw, ch, fc=C_THR, ec=C_ORANGE,
                 lw=1.1, hatch="////", alpha=0.8)
        ax.text(Asx+Asw/2, Asy+Ash-1.5*ch, "t0  t1  t2  t3", ha="center",
                va="center", fontsize=9.5, color=C_ORANGE)
        ax.text(Asx+Asw/2, Asy-0.3, "As（BK×BM）", ha="center", fontsize=13, color=C_A_DK)
        brace_h(ax, Asx, Asx+Asw, Asy+Ash+0.18, "BK", fs=13, color=C_INK, side="top")
        brace_v(ax, Asx-0.25, Asy, Asy+Ash, "BM", fs=13, color=C_INK)

        # loadOffset 箭头
        arrow(ax, (2.0, Asy+Ash-1.5*ch), (Asx-0.05, Asy+Ash-1.5*ch),
              color=C_ORANGE, lw=1.8)
        ax.text(2.0, Asy+Ash-1.15, "loadOffset：本步起点\n（线程平铺编号）", ha="center",
                fontsize=11, color=C_ORANGE)
        # strideA 大括号
        ax.add_patch(FancyArrowPatch((Asx-0.1, Asy+Ash-2*ch),
            (Asx-0.1, Asy+Ash-6*ch),
            connectionstyle="arc3,rad=-0.6",
            arrowstyle="->", color=C_ORANGE, lw=1.6, mutation_scale=14))
        ax.text(1.55, Asy+Ash-3.8*ch, "strideA\n= 块内线程数", ha="center",
                va="center", fontsize=11, color=C_ORANGE)

        # 下一步箭头
        arrow(ax, (Asx+Asw/2, Asy+Ash-2*ch-0.05),
              (Asx+Asw/2, Asy+Ash-6*ch), color=C_INK, lw=1.6)
        ax.text(Asx+Asw/2+0.15, Asy+Ash-6.3*ch, "下一步", ha="left", fontsize=11)

        ax.text(7.2, 7.4,
                "每个线程每一步\n从全局内存取 1 个元素\n放入 As 对应位置\n（Bs 同理）",
                ha="left", va="center", fontsize=11.5)
        ax.text(7.2, 5.4,
                "加载行横跨 As 的整个 BK 宽度，\n"
                "warp 内 32 个线程访问连续地址，\n"
                "天然满足合并访存；\n"
                "行与行之间按线程数跨步",
                ha="left", va="center", fontsize=11.5)
        ax.text(5.25, 0.9,
                "代码：i = tid；for (i = tid; i < BM×BK; i += blockDim.x)\n"
                "          m = i / BK；k = i % BK；As[m][k] = A[…][…]",
                ha="center", va="center", fontsize=11, color=C_INK,
                family="monospace")
        save(fig, "sib_07_gmem_loading")


# ================================================================
# 图 8：GEMM5 寄存器复用 / 外积展开（严格仿 kernel_5_reg_blocking）
# ================================================================
def fig8_reg_blocking():
    with plt.xkcd(scale=0.55, length=90):
        xkcd_on()
        fig, ax = new_fig(15.5, 7.2, (0, 15.9), (0, 7.2))
        ax.text(0.5, 6.9, "GEMM5 寄存器复用：展开 dotIdx 循环，每步只把一行 A、一列 B 读进寄存器",
                ha="left", fontsize=13.5)

        frame_w = 3.75
        x0 = 0.7
        for f in range(4):
            fx = x0 + f*(frame_w+0.05)
            # A 横条（TM 高 × BK 长，4 格示意）
            ay, aw, ah = 2.0, 2.0, 0.8
            cell(ax, fx, ay, aw, ah, fc="white", ec=C_ORANGE, lw=1.6)
            dashed_grid(ax, fx, ay, 4, 2, aw/4, color="#d1d5db")
            cell(ax, fx+f*aw/4, ay, aw/4, ah, fc=C_A, ec=C_A_DK, lw=1.5,
                 hatch="////")
            ax.text(fx+aw-0.12, ay-0.22, "A", fontsize=14, color=C_A_DK, ha="right")
            if f == 0:
                brace_h(ax, fx, fx+aw, ay+ah+0.12, "BK", fs=10.5, color=C_INK, side="top")
                brace_v(ax, fx-0.15, ay, ay+ah, "TM", fs=10.5, color=C_INK)
            # B 竖条（TN 宽 × BK 高，4 格示意），位于 A 右端上方
            bx, by, bw, bh = fx+1.95, 3.0, 0.65, 2.2
            cell(ax, bx, by, bw, bh, fc="white", ec=C_ORANGE, lw=1.6)
            dashed_grid(ax, bx, by, 2, 4, bw/2, color="#d1d5db")
            cell(ax, bx, by+bh-f*bh/4-bh/4, bw, bh/4, fc=C_B, ec=C_B_DK,
                 lw=1.5, hatch="\\\\\\\\")
            ax.text(bx+bw+0.1, by+0.12, "B", fontsize=14, color=C_B_DK)
            if f == 0:
                brace_h(ax, bx, bx+bw, by+bh+0.12, "TN", fs=10.5, color=C_INK, side="top")
                ax.text(bx-0.2, by+bh/2-0.3, "BK", rotation=90, ha="right",
                        va="center", fontsize=10.5, color=C_INK)
            # 结果网格（2×2，逐帧加深）
            rx, ry, rg = fx+2.75, 2.2, 0.4
            for ii in range(2):
                for jj in range(2):
                    alpha = 0.18 + 0.22*f
                    cell(ax, rx+jj*rg, ry+(1-ii)*rg, rg, rg, fc=C_THR,
                         ec=C_ORANGE, lw=1.0, hatch="////", alpha=alpha)
            ax.text(rx+rg, ry-0.3, "结果", ha="center", fontsize=10.5, color=C_C_DK)
            ax.text(fx+1.7, 1.35, "dotIdx=%d" % f, ha="center", fontsize=13, weight="bold")
            if f < 3:
                ax.plot([fx+frame_w+0.02, fx+frame_w+0.02], [1.7, 6.0],
                        color=C_GRID, lw=1.2)

        # regM / regN 箭头指向第一帧
        arrow(ax, (1.5, 6.35), (x0+0.12, 2.85), color=C_A_DK, lw=1.4)
        ax.text(0.7, 6.45, "regM：A 的一列（寄存器）", fontsize=11, color=C_A_DK)
        arrow(ax, (3.55, 6.3), (3.22, 5.35), color=C_B_DK, lw=1.4)
        ax.text(3.4, 6.45, "regN：B 的一行", fontsize=11, color=C_B_DK)

        ax.text(0.7, 0.95,
                "每一步把相关的 A 行、B 列读入 regM、regN 寄存器，做一次外积并累加进 threadResults；\n"
                "示意 2×2 粗化、4 步：每步 4 次 SMEM 读，总共 16 次 SMEM 加载（GEMM4 是 64 个结果 512 次）。",
                ha="left", va="center", fontsize=11.5)
        ax.text(0.7, 0.42,
                "GEMM5 的改进：B 元素先存进 b_val[TN] 寄存器，同一个 B 被 TM 行复用，\n"
                "每步 SMEM 读次数从 16 降到 8，把“跨线程复用”进一步做成“线程内复用”。",
                ha="left", va="center", fontsize=11.5, color=C_ORANGE)
        save(fig, "sib_08_reg_blocking")


# ================================================================
# 图 9：算术强度对比（严格仿 raising_arith_inten）
# ================================================================
def _mem_grid(ax, x, y, s, n=8, hi_rows=None, hi_cols=None, hi_cells=None):
    """画 n×n 虚线网格，hi_rows 高亮行、hi_cols 高亮列、hi_cells 高亮亮格。"""
    cell(ax, x, y, s, s, fc="white", ec=C_INK, lw=1.4)
    dashed_grid(ax, x, y, n, n, s/n, color=C_GRID)
    g = s/n
    if hi_rows:
        for r, col in hi_rows:
            cell(ax, x, y+s-(r+1)*g, s, g, fc=col, ec=col, lw=0.8,
                 hatch="////", alpha=0.75)
    if hi_cols:
        for c, col in hi_cols:
            cell(ax, x+c*g, y, g, s, fc=col, ec=col, lw=0.8,
                 hatch="\\\\\\\\", alpha=0.75)
    if hi_cells:
        for (r, c), col in hi_cells:
            cell(ax, x+c*g, y+s-(r+1)*g, g, g, fc=col, ec=col, lw=0.8,
                 hatch="////", alpha=0.85)


def fig9_arith_intensity():
    with plt.xkcd(scale=0.55, length=90):
        xkcd_on()
        fig, ax = new_fig(13.5, 10.2, (0, 13.5), (0, 10.2))
        ax.text(6.75, 10.0, "线程粗化为什么能加速：同一份数据被更多结果共享，每个结果分摊的访存大幅下降",
                ha="center", fontsize=13.5)

        # ---------------- 上半：一个线程 1 个结果
        ax.text(0.4, 9.15, "每线程只算 1 个结果（GEMM1）：", fontsize=13)
        ax.text(0.4, 8.55, "· 从 A 读 1 整行：8 次加载\n"
                "· 从 B 读 1 整列：8 次加载\n"
                "· C 读 1 次、写 1 次\n"
                "→ 共 18 次访存，只为 1 个结果服务",
                fontsize=11.5, va="top")
        # B 网格（右上，高亮 1 列）
        _mem_grid(ax, 10.4, 7.7, 2.0, hi_cols=[(3, C_B)])
        ax.text(10.4+1.0, 7.42, "B", fontsize=14, ha="center")
        # A 网格（左中，高亮 1 行）
        _mem_grid(ax, 6.8, 5.4, 2.0, hi_rows=[(3, C_A)])
        ax.text(6.8+1.0, 5.12, "A", fontsize=14, ha="center")
        # C 网格（右中，高亮 1 格）
        _mem_grid(ax, 10.4, 5.4, 2.0, hi_cells=[((3, 3), C_C)])
        ax.text(10.4+1.0, 5.12, "C", fontsize=14, ha="center")

        ax.plot([0.4, 13.1], [4.95, 4.95], color=C_GRID, lw=1.4, ls=(0, (6, 4)))

        # ---------------- 下半：一个线程 8×8 个结果
        ax.text(0.4, 4.55, "每线程算 8×8 个结果（GEMM4）：", fontsize=13)
        ax.text(0.4, 3.95, "· A 读 8 行（每行被 8 个结果共享）：64 次\n"
                "· B 读 8 列（每列被 8 个结果共享）：64 次\n"
                "· C 读 64 次、写 64 次\n"
                "→ 共 256 次访存，服务 64 个结果\n"
                "→ 平均每结果 4 次访存（18 → 4）",
                fontsize=11.5, va="top")
        # B 网格（高亮 2 列示意 8 列）
        _mem_grid(ax, 10.4, 2.75, 1.9, hi_cols=[(3, C_B), (4, C_THR)])
        ax.text(10.4+0.95, 2.48, "B", fontsize=14, ha="center")
        # A 网格（高亮 2 行示意 8 行）
        _mem_grid(ax, 6.9, 0.55, 1.9, hi_rows=[(3, C_A), (4, C_THR)])
        ax.text(6.9+0.95, 0.28, "A", fontsize=14, ha="center")
        # C 网格（高亮 2×2 示意 8×8）
        _mem_grid(ax, 10.4, 0.55, 1.9,
                  hi_cells=[((3, 3), C_C), ((3, 4), C_C),
                            ((4, 3), C_A), ((4, 4), C_B)])
        ax.text(10.4+0.95, 0.28, "C", fontsize=14, ha="center")
        ax.text(0.4, 0.6, "网格中只画 2 行 / 2 列 / 2×2 作示意，实际是 8 行、8 列、8×8",
                fontsize=10.5, color=C_GRID, ha="left")
        save(fig, "sib_09_arith_intensity")


# ================================================================
# 图 13：三层循环结构（严格仿 Loop_structure）
# ================================================================
def fig13_loop_structure():
    with plt.xkcd(scale=0.55, length=90):
        xkcd_on()
        fig, ax = new_fig(12.5, 7.2, (0, 12.5), (0, 7.2))
        ax.text(6.25, 7.0, "三层循环 × 三级存储：循环每深入一层，数据就向计算单元靠近一级",
                ha="center", fontsize=13.5)

        # 三个竖条
        bars = [
            (1.5, 0.85, 5.7, C_A, "BLOCKTILING"),
            (2.55, 0.85, 4.0, C_B, "THREADTILING"),
            (3.5, 0.85, 2.2, C_THR, "COMPUTE"),
        ]
        for x, y, h, col, name in bars:
            cell(ax, x, y, 0.75, h, fc=col, ec=C_INK, lw=1.3,
                 hatch="xxxx", alpha=0.7)
        ax.text(1.875, 4.5, "块级", ha="center", fontsize=11, rotation=90, color=C_A_DK)
        ax.text(2.925, 3.4, "线程级", ha="center", fontsize=11, rotation=90, color=C_B_DK)

        rows = [
            (5.65, "Loop1：每个 block 沿 K 循环", C_A_DK,
                  "访存：把 As、Bs 从 GMEM 搬到 SMEM"),
            (3.9, "Loop2：每个线程沿 BK（dotIdx）循环", C_B_DK,
                  "访存：把 As、Bs 从 SMEM 搬到寄存器"),
            (2.1, "Loop3 & Loop4：寄存器内容做外积累加", C_ORANGE,
                  "访存：标量从寄存器进入 CUDA Core，做 FMA"),
        ]
        for y, t1, col, t2 in rows:
            ax.text(5.0, y, t1, fontsize=13, va="center", color=col)
            ax.text(5.2, y-0.6, t2, fontsize=11.5, va="center", color=C_INK)

        ax.text(1.875, 0.55, "BLOCKTILING", fontsize=11, ha="center", weight="bold")
        ax.text(2.925, 0.55, "THREADTILING", fontsize=11, ha="center", weight="bold")
        ax.text(3.875, 0.55, "COMPUTE", fontsize=11, ha="center", weight="bold")
        ax.text(7.9, 0.25,
                "越靠外的循环搬运数据越大、越慢；越靠内的循环数据越小、越快。",
                ha="center", fontsize=11)
        save(fig, "sib_13_loop_structure")


# ================================================================
# 图 10：float4 向量化（GEMM6）
# ================================================================
def fig10_float4():
    with plt.xkcd(scale=0.55, length=90):
        xkcd_on()
        fig, ax = new_fig(12.5, 6.6, (0, 12.5), (0, 6.6))
        ax.text(6.25, 6.4, "float4 向量化：一条指令搬 4 个连续 float，加载指令数降到 1/4",
                ha="center", fontsize=13.5)

        def mem_row(y, vector):
            x, w, n = 3.7, 6.4, 8
            cw = w/n
            for i in range(n):
                fc = C_A if i < 4 else "#f3f4f6"
                cell(ax, x+i*cw, y, cw, 0.72, fc=fc, ec=C_INK, lw=1.1)
            if vector:
                # 一个大箭头覆盖 4 格
                ax.add_patch(FancyArrowPatch((1.7, y-0.7), (x+4*cw-0.05, y-0.02),
                    arrowstyle="->", color=C_ORANGE, lw=3.0, mutation_scale=20))
                ax.text(1.6, y-0.8, "t0：1 条指令", ha="right", fontsize=11.5, color=C_ORANGE)
            else:
                for i in range(4):
                    ax.add_patch(FancyArrowPatch(
                        (1.7+i*0.02, y-0.62), (x+(i+0.5)*cw, y-0.05),
                        arrowstyle="->", color=C_GRID, lw=1.2, mutation_scale=11))
                ax.text(1.6, y-0.72, "t0：4 条指令", ha="right", fontsize=11.5, color=C_GRID)

        ax.text(0.5, 5.6, "普通加载：每条 LDG 搬 1 个 float（32bit = 4B）", fontsize=12.5)
        mem_row(4.2, False)
        ax.text(3.7, 5.15, "连续 4 个 float", fontsize=11, color=C_A_DK)

        ax.text(0.5, 3.0, "float4：1 条 LDG.128 一次搬 4 个 float（128bit = 16B）", fontsize=12.5)
        mem_row(1.6, True)
        ax.text(3.7, 2.55, "连续 4 个 float", fontsize=11, color=C_A_DK)

        ax.text(10.5, 4.3,
                "指令数：4 → 1\n"
                "GEMM6 中 A、B 加载\n"
                "和 C 写回都改用 float4",
                ha="left", va="center", fontsize=11.5)
        ax.text(10.5, 1.8,
                "前提：地址 16B 对齐\n"
                "（矩阵列数按 16 对齐分配）",
                ha="left", va="center", fontsize=11.5)
        save(fig, "sib_10_float4")


# ================================================================
# 图 11：Bank 冲突反例（GEMM7）
# ================================================================
def fig11_bank_conflict():
    with plt.xkcd(scale=0.55, length=90):
        xkcd_on()
        fig, ax = new_fig(13.5, 6.8, (0, 13.5), (0, 6.8))
        ax.text(6.75, 6.6, "共享内存 Bank 冲突：32 个 Bank 本可并行，访问同一个 Bank 就被迫串行",
                ha="center", fontsize=13.5)

        def banks(y, conflict):
            x, n, w, h = 2.0, 8, 8.4, 1.9
            bw = w/n
            for i in range(n):
                cell(ax, x+i*bw, y, bw-0.02, h, fc="white", ec=C_INK, lw=1.1)
                ax.text(x+i*bw+bw/2-0.01, y+h-0.16, "Bank %d" % i,
                        ha="center", fontsize=9, color=C_GRID)
            if not conflict:
                for i in range(8):
                    cell(ax, x+i*bw+0.05, y+h/2-0.3, bw-0.12, 0.6,
                         fc=C_C, ec=C_C_DK, lw=1.2, hatch="////")
                    ax.text(x+i*bw+bw/2-0.01, y+h/2, "t%d" % i,
                            ha="center", va="center", fontsize=9)
            else:
                for i in range(8):
                    cell(ax, x+0.05, y+h-0.4-i*0.2, bw-0.12, 0.18,
                         fc=C_A_RED, ec=C_RED, lw=1.0)
                ax.text(x+bw+0.25, y+0.25, "t0~t7 全在这里\n（同列重叠）",
                        ha="left", va="bottom", fontsize=9.5, color=C_RED)

        ax.text(0.5, 6.15, "无冲突：8 个线程访问 8 个不同 Bank（真实 warp 为 32 个）",
                fontsize=12.5)
        banks(4.0, False)
        ax.text(11.0, 4.95, "并行完成\n≈ 1 个周期", ha="left", va="center",
                fontsize=11.5, color=C_C_DK)

        ax.text(0.5, 3.25, "冲突：8 个线程访问同一个 Bank（GEMM7 实测为 32-way）",
                fontsize=12.5, color=C_RED)
        banks(1.0, True)
        ax.text(11.0, 1.95, "硬件串行化\n≈ 8 个周期\n（真实 32 个）",
                ha="left", va="center", fontsize=11.5, color=C_RED)
        ax.text(6.75, 0.5,
                "GEMM7 把 Bds 改成 [BN][BK] 转置布局后，同一 k、不同 n 的地址相差 BK=32（Bank 数的整数倍），"
                "Bank 号全部相同",
                ha="center", fontsize=10.5)
        save(fig, "sib_11_bank_conflict")


# ================================================================
# 图 12：XOR Swizzle（GEMM8）
# ================================================================
def fig12_swizzle():
    with plt.xkcd(scale=0.55, length=90):
        xkcd_on()
        fig, ax = new_fig(13.5, 6.8, (0, 13.5), (0, 6.8))
        ax.text(6.75, 6.6, "XOR Swizzle：用一个异或把 SMEM 列号交错开，加载、计算两边都不撞 Bank",
                ha="center", fontsize=13.5)
        ax.text(6.75, 6.15, "swizzle(n,k) = ((k>>2) ^ ((n>>3) & 7)) << 2 | (k & 3)，以 float4（4 个 float）为一组",
                ha="center", fontsize=11, family="monospace")

        def bank_grid(x, y, s, xor_mode):
            n = 8
            cell(ax, x, y, s, s, fc="white", ec=C_INK, lw=1.3)
            g = s/n
            for r in range(n):
                for c in range(n):
                    b = ((r ^ c) if xor_mode else c) % 8
                    hi = (c == 2)
                    fc = C_A_RED if (hi and not xor_mode) else (C_C if hi else "white")
                    cell(ax, x+c*g, y+(n-1-r)*g, g, g, fc=fc, ec=C_GRID, lw=0.5)
                    ax.text(x+c*g+g/2, y+(n-1-r)*g+g/2-0.01, str(b),
                            ha="center", va="center", fontsize=8.5,
                            color=C_RED if hi else C_INK)

        bank_grid(1.8, 1.4, 3.6, False)
        ax.text(3.6, 1.1, "swizzle 前：Bank 号按列排", ha="center", fontsize=12)
        ax.text(3.6, 5.25, "计算阶段同 k 的一列，Bank 号全相同 → 冲突",
                ha="center", fontsize=10.5, color=C_RED)

        ax.text(6.3, 3.2, "XOR\n交错", ha="center", va="center", fontsize=15, color=C_ORANGE)
        arrow(ax, (5.7, 3.2), (6.9, 3.2), color=C_ORANGE, lw=2.2)

        bank_grid(7.6, 1.4, 3.6, True)
        ax.text(9.4, 1.1, "swizzle 后：每行 XOR 偏移", ha="center", fontsize=12)
        ax.text(9.4, 5.25, "同一列被打散到 8 个不同 Bank → 无冲突",
                ha="center", fontsize=10.5, color=C_C_DK)

        ax.text(11.7, 3.5,
                "格子里的数字是 Bank 号\n"
                "（示意 8×8，实际 32 Bank）\n\n"
                "红色列为计算阶段\n"
                "同一 k、不同 n 的访问\n\n"
                "GEMM8 一处 swizzle，\n"
                "加载阶段 4-way、计算阶段\n"
                "2-way 的冲突同时消除",
                ha="left", va="center", fontsize=11)
        save(fig, "sib_12_swizzle")


# ================================================================
# 图 14：双缓冲甘特（GEMM9）
# ================================================================
def fig14_double_buffer():
    with plt.xkcd(scale=0.55, length=90):
        xkcd_on()
        fig, ax = new_fig(13.5, 6.8, (0, 13.5), (0, 6.8))
        ax.text(6.75, 6.6, "双缓冲：两块 SMEM 交替使用，算第 k 块的同时预取第 k+1 块",
                ha="center", fontsize=13.5)

        def lane(y, label):
            ax.text(1.9, y+0.25, label, ha="right", va="center", fontsize=11)
            ax.plot([2.0, 9.8], [y, y], color=C_GRID, lw=0.8)

        def bar(x, y, w, fc, ec, t):
            cell(ax, x, y, w, 0.5, fc=fc, ec=ec, lw=1.3, hatch="////", alpha=0.8)
            ax.text(x+w/2, y+0.25, t, ha="center", va="center", fontsize=9.5)

        ax.text(0.5, 6.0, "单缓冲：加载与计算严格串行（加载时计算单元空转）", fontsize=12.5)
        lane(4.95, "加载")
        lane(4.2, "计算")
        for k in range(4):
            xl = 2.0 + k*1.4
            bar(xl, 4.75, 0.7, C_A, C_A_DK, "加载 %d" % k)
            bar(xl+0.7, 4.0, 0.7, C_C, C_C_DK, "计算 %d" % k)

        ax.text(0.5, 3.25, "双缓冲：两块 SMEM（buf[0]/buf[1]）轮流，计算 k 与加载 k+1 重叠", fontsize=12.5)
        lane(2.2, "加载")
        lane(1.45, "计算")
        for k in range(4):
            bar(2.0+k*0.7, 2.0, 0.7, C_A, C_A_DK, "加载 %d" % k)
            bar(2.7+k*0.7, 1.25, 0.7, C_C, C_C_DK, "计算 %d" % k)
        ax.text(10.2, 4.5, "4 个块要 8 段时间", fontsize=11, ha="left")
        ax.text(10.2, 1.8, "约 5 段时间\n（首尾各 1 段 + 4 段）", fontsize=11, ha="left", color=C_C_DK)
        ax.text(6.75, 0.55,
                "GEMM9 申请两块 [2][BM][BK] 的动态共享内存（共 64KB，需 cudaFuncSetAttribute 提高上限）；"
                "GEMM9 是软件预取，还没有真正的异步流水",
                ha="center", fontsize=10.5)
        save(fig, "sib_14_double_buffer")


# ================================================================
# 图 15：cp.async DMA 路径（GEMM9B / GEMM10B）
# ================================================================
def fig15_cp_async():
    with plt.xkcd(scale=0.55, length=90):
        xkcd_on()
        fig, ax = new_fig(13.5, 6.8, (0, 13.5), (0, 6.8))
        ax.text(6.75, 6.6, "cp.async：让 DMA 引擎把数据直接搬进 SMEM，不经过寄存器、不占用线程",
                ha="center", fontsize=13.5)

        ax.add_patch(FancyBboxPatch((0.3, 1.1), 5.9, 5.0,
            boxstyle="round,pad=0.02,rounding_size=0.15", fc="none", ec=C_GRID,
            lw=1.2, ls=(0, (4, 3))))
        ax.text(0.5, 5.85, "传统路径（LDG + STS）", fontsize=12.5, color=C_GRID)
        cell(ax, 1.2, 4.6, 4.0, 0.75, fc=C_A, ec=C_A_DK, lw=1.4)
        ax.text(3.2, 4.97, "GMEM", ha="center", fontsize=13)
        cell(ax, 2.0, 3.25, 2.4, 0.7, fc=C_THR, ec=C_ORANGE, lw=1.5)
        ax.text(3.2, 3.6, "线程寄存器", ha="center", fontsize=12)
        cell(ax, 1.2, 1.9, 4.0, 0.75, fc=C_C, ec=C_C_DK, lw=1.4)
        ax.text(3.2, 2.27, "SMEM", ha="center", fontsize=13)
        arrow(ax, (3.2, 4.55), (3.2, 4.0), color=C_INK, lw=2.0, text="LDG", fs=11,
              text_xy=(3.55, 4.28))
        arrow(ax, (3.2, 3.2), (3.2, 2.7), color=C_INK, lw=2.0, text="STS", fs=11,
              text_xy=(3.55, 2.95))
        ax.text(3.25, 1.45,
                "两次访存都经过寄存器；\n线程必须等 LDG 完成才能继续",
                ha="center", fontsize=11)

        ax.add_patch(FancyBboxPatch((6.8, 1.1), 6.4, 5.0,
            boxstyle="round,pad=0.02,rounding_size=0.15", fc="none", ec=C_GRID,
            lw=1.2, ls=(0, (4, 3))))
        ax.text(7.0, 5.85, "cp.async 异步路径（Ampere sm_80 起）", fontsize=12.5, color=C_GRID)
        cell(ax, 7.6, 4.6, 4.6, 0.75, fc=C_A, ec=C_A_DK, lw=1.4)
        ax.text(9.9, 4.97, "GMEM", ha="center", fontsize=13)
        cell(ax, 8.6, 3.25, 2.6, 0.7, fc=C_BLOCK, ec=C_PURPLE, lw=1.5)
        ax.text(9.9, 3.6, "DMA 异步引擎", ha="center", fontsize=12)
        cell(ax, 7.6, 1.9, 4.6, 0.75, fc=C_C, ec=C_C_DK, lw=1.4)
        ax.text(9.9, 2.27, "SMEM", ha="center", fontsize=13)
        arrow(ax, (9.9, 4.55), (9.9, 4.0), color=C_PURPLE, lw=2.0,
              text="cp.async", fs=11, text_xy=(10.95, 4.28))
        arrow(ax, (9.9, 3.2), (9.9, 2.7), color=C_PURPLE, lw=2.0)
        ax.text(7.2, 3.6, "提交后立即返回，\n线程去做计算", ha="left", fontsize=10.5,
                color=C_PURPLE)
        ax.text(9.9, 1.45,
                "commit_group 标记提交；wait_group 等待整批到齐\n"
                "搬运全程不经过寄存器，线程与 DMA 并行",
                ha="center", fontsize=11)
        ax.text(6.75, 0.5,
                "GEMM9B 的 A 走 cp.async（B 因转置仍需 LDG+STS，半流水）；GEMM10B 同理。"
                "架构低于 sm_80 时回退 GEMM9",
                ha="center", fontsize=10.5)
        save(fig, "sib_15_cp_async")


# ================================================================
# 图 16：寄存器流水（GEMM10）
# ================================================================
def fig16_register_pipeline():
    with plt.xkcd(scale=0.55, length=90):
        xkcd_on()
        fig, ax = new_fig(13.5, 6.8, (0, 13.5), (0, 6.8))
        ax.text(6.75, 6.6, "寄存器流水：先把加载请求发出去不等待，计算把延迟“藏”起来，落 SMEM 时才收口",
                ha="center", fontsize=13.5)

        def lane(y, label, ytag=6.0):
            ax.text(3.0, y+0.25, label, ha="right", va="center", fontsize=11)
            ax.plot([3.1, 12.4], [y, y], color=C_GRID, lw=0.8)

        lane(4.95, "issue_load：发 LDG.128（不等）")
        lane(4.2, "在途数据（预取寄存器）")
        lane(3.0, "计算：FMA")
        lane(2.25, "commit_store：STS 落 SMEM")
        for k in range(4):
            x0 = 3.1 + k*1.3
            cell(ax, x0, 4.75, 0.85, 0.45, fc=C_A, ec=C_A_DK, lw=1.3, hatch="////")
            ax.text(x0+0.42, 4.97, "发 %d" % k, ha="center", va="center", fontsize=9.5)
            # 在途：虚线从 issue 到 commit
            cell(ax, x0+0.85, 4.0, 0.85, 0.45, fc="white", ec=C_A_DK, lw=1.0,
                 ls=(0, (3, 2)))
            # 计算条（与在途重叠）
            cell(ax, x0+0.2, 2.8, 1.5, 0.45, fc=C_C, ec=C_C_DK, lw=1.2,
                 hatch="////", alpha=0.85)
            # commit
            cell(ax, x0+1.7, 2.05, 0.5, 0.45, fc=C_THR, ec=C_ORANGE, lw=1.3)
            ax.text(x0+1.95, 2.27, "落", ha="center", va="center", fontsize=9)
        arrow(ax, (11.0, 4.2), (11.7, 4.2), color=C_INK, lw=1.5)
        ax.text(11.8, 4.2, "多个请求同时在途", ha="left", va="center", fontsize=10.5)
        ax.text(11.8, 3.0, "计算不等待", ha="left", va="center", fontsize=10.5, color=C_C_DK)

        ax.text(0.5, 1.35,
                "issue_load：一次性发 4 个 float4 的 LDG（A、B 各 4 个），不等待返回，继续往下计算；",
                fontsize=11)
        ax.text(0.5, 0.9,
                "commit_store：把预取寄存器里的数据写进 SMEM，此时才隐式等待未到的数据；",
                fontsize=11)
        ax.text(0.5, 0.45,
                "GEMM10 每线程多持有 8 个 float4 预取寄存器，A、B 两路全流水（不依赖 cp.async，各架构可用）",
                fontsize=11, color=C_ORANGE)
        save(fig, "sib_16_register_pipeline")


# ================================================================
# 图 17：两级流水（GEMM11 / GEMM11B）
# ================================================================
def fig17_two_stage_pipeline():
    with plt.xkcd(scale=0.55, length=90):
        xkcd_on()
        fig, ax = new_fig(13.5, 7.0, (0, 13.5), (0, 7.0))
        ax.text(6.75, 6.8, "两级流水：GMEM→SMEM、SMEM→寄存器 fragment 都做双缓冲，三级动作像流水线一样重叠",
                ha="center", fontsize=13.5)

        lanes = [
            (5.3, "GMEM → SMEM", C_A, C_A_DK, 0),
            (4.0, "SMEM → fragment", C_B, C_B_DK, 1),
            (2.7, "FMA 计算", C_C, C_C_DK, 2),
        ]
        sw = 1.05
        for y, label, fc, ec, offset in lanes:
            ax.text(1.9, y+0.28, label, ha="right", va="center", fontsize=11.5)
            ax.plot([2.0, 11.6], [y, y], color=C_GRID, lw=0.8)
            for k in range(5):
                x0 = 2.0 + k*1.05 + offset*sw
                cell(ax, x0, y, sw-0.08, 0.55, fc=fc, ec=ec, lw=1.3,
                     hatch="////", alpha=0.8)
                ax.text(x0+(sw-0.08)/2, y+0.27, "stage %d" % (k+offset),
                        ha="center", va="center", fontsize=8.5)

        # 垂直对齐参考线
        for k in range(4):
            ax.plot([2.0+3*sw+k*sw, 2.0+3*sw+k*sw], [2.6, 5.9],
                    color=C_GRID, lw=0.7, ls=(0, (2, 3)))
        ax.text(2.0, 6.15, "同一时刻三级各干各的（加载 k+2、取 frag k+1、算 k）",
                ha="left", fontsize=11, color=C_ORANGE)

        ax.text(0.5, 1.75,
                "第一级：SMEM 开两份 [2][BK][BM] / [2][BK][BN]，配合 cp.async 或寄存器预取交替填充；",
                fontsize=11)
        ax.text(0.5, 1.3,
                "第二级：每线程开两份 a_frag[2] / b_frag[2] 寄存器片段，从 SMEM 交替读取；",
                fontsize=11)
        ax.text(0.5, 0.85,
                "GEMM11：SMEM 用转置布局（B 半流水）；GEMM11B：B 保持 [BK][BN] 与全局对齐、走 cp.async，性能更好",
                fontsize=11, color=C_ORANGE)
        ax.text(0.5, 0.4,
                "三级同时推进，访存延迟被计算填满，是手写 kernel 在 RTX 5060 上最接近 cuBLAS 的版本（GEMM12 之前）",
                fontsize=11)
        save(fig, "sib_17_two_stage_pipeline")


# ================================================================
# 图 18：GPU 存储层级金字塔
# ================================================================
def fig18_memory_hierarchy():
    with plt.xkcd(scale=0.55, length=90):
        xkcd_on()
        fig, ax = new_fig(13.5, 7.6, (0, 13.5), (0, 7.6))
        ax.text(6.75, 7.35, "GPU 存储层级：越往上越快、越小、越私有；GEMM 优化就是把数据往上层搬并反复用",
                ha="center", fontsize=13.5)

        cx = 4.2
        layers = [
            (4.9, 6.3, 1.0, 1.5, C_THR, C_ORANGE, "寄存器（Register）",
             "线程私有 · 单周期访问",
             "每线程最多 255 个 32 位寄存器"),
            (3.5, 4.9, 1.5, 2.3, C_B, C_B_DK, "共享内存（SMEM）",
             "Block 内共享 · 数十周期 · 片上 SRAM",
             "每 SM 上百 KB（动态共享内存，GEMM9 起 64KB）"),
            (2.1, 3.5, 2.3, 3.1, C_A, C_A_DK, "L2 Cache",
             "全芯片共享 · 数十 MB · 对所有 SM 可见",
             "由硬件自动管理"),
            (0.7, 2.1, 3.1, 3.9, "#93c5fd", C_A_DK, "Global Memory（显存 HBM/GDDR）",
             "所有线程可见 · 数百周期 · 容量 GB 级",
             "A、B、C 原始矩阵所在\n带宽 ~1 TB/s 量级，是整个优化要减少访问的对象"),
        ]
        for y0, y1, wtop, wbot, fc, ec, name, spec, note in layers:
            poly = Polygon([
                (cx-wtop, y1), (cx+wtop, y1),
                (cx+wbot, y0), (cx-wbot, y0)],
                closed=True, fc=fc, ec=ec, lw=1.6, alpha=0.6, hatch="////")
            ax.add_patch(poly)
            ax.text(cx, (y0+y1)/2+0.12, name, ha="center", va="center",
                    fontsize=13, weight="bold")
            ax.text(cx, (y0+y1)/2-0.28, spec, ha="center", va="center", fontsize=10)

        ax.annotate("", xy=(0.25, 6.4), xytext=(0.25, 1.0),
                    arrowprops=dict(arrowstyle="->", color=C_INK, lw=1.8))
        ax.text(0.15, 3.7, "速度：越往上越快", rotation=90, ha="center",
                va="center", fontsize=11.5)
        ax.annotate("", xy=(8.15, 0.9), xytext=(8.15, 6.4),
                    arrowprops=dict(arrowstyle="->", color=C_INK, lw=1.8))
        ax.text(8.45, 3.7, "容量：越往下越大", rotation=90, ha="center",
                va="center", fontsize=11.5)

        for y0, y1, _, _, _, _, name, spec, note in layers:
            ax.text(9.0, (y0+y1)/2, note, ha="left", va="center", fontsize=10.8)
        save(fig, "sib_18_memory_hierarchy")


# ================================================================
# 图 19：三级 tiling 总图（grid → block tile → thread tile）
# ================================================================
def fig19_tiling_overview():
    with plt.xkcd(scale=0.55, length=90):
        xkcd_on()
        fig, ax = new_fig(13.5, 7.2, (0, 13.5), (0, 7.2))
        ax.text(6.75, 6.9, "三级 tiling 总图：Grid 铺满 C，Block 负责 128×128，线程负责 8×8",
                ha="center", fontsize=13.5)

        # C 大矩阵（10×12 网格）
        cx, cy, cw, ch = 0.8, 1.8, 2.8, 3.6
        cell(ax, cx, cy, cw, ch, fc="white", ec=C_INK, lw=1.6)
        dashed_grid(ax, cx, cy, 8, 10, cw/8, color=C_GRID)
        cell(ax, cx+3*cw/8, cy+ch-5*ch/10, 2*cw/8, 2*ch/10,
             fc=C_C, ec=C_C_DK, lw=1.6, hatch="////", zorder=4)
        ax.text(cx+cw/2, cy-0.32, "Grid = C 矩阵\n（按 block tile 铺满）",
                ha="center", fontsize=11.5)
        arrow(ax, (cx+cw+0.15, cy+ch/2), (4.5, cy+ch/2),
              color=C_INK, lw=2.0, text="放大 1 个 block", fs=11,
              text_xy=(2.9, 4.6))

        # Block tile（16×16 个 thread tile，画 8×8 示意）
        bx, by, bs = 4.7, 2.1, 3.0
        cell(ax, bx, by, bs, bs, fc="white", ec=C_C_DK, lw=1.6)
        n = 8
        dashed_grid(ax, bx, by, n, n, bs/n, color=C_GRID)
        cell(ax, bx+3*bs/n, by+bs-4*bs/n, bs/n, bs/n,
             fc=C_THR, ec=C_ORANGE, lw=1.8, hatch="////", zorder=4)
        ax.text(bx+bs/2, by+bs+0.2, "Block tile：128×128",
                ha="center", fontsize=12, color=C_C_DK)
        brace_h(ax, bx, bx+bs, by-0.14, "BN=128（16 个 thread tile）", fs=10, color=C_INK)
        brace_v(ax, bx-0.16, by, by+bs, "BM=128", fs=10, color=C_INK)
        arrow(ax, (bx+bs+0.15, by+bs/2), (8.5, by+bs/2),
              color=C_INK, lw=2.0, text="放大 1 个线程", fs=11,
              text_xy=(6.6, by+bs/2+0.5))

        # Thread tile（8×8 网格）
        tx, ty, ts = 8.7, 2.5, 2.4
        cell(ax, tx, ty, ts, ts, fc="white", ec=C_ORANGE, lw=1.6)
        dashed_grid(ax, tx, ty, 8, 8, ts/8, color=C_GRID)
        ax.text(tx+ts/2, ty+ts+0.2, "Thread tile：8×8",
                ha="center", fontsize=12, color=C_ORANGE)
        brace_h(ax, tx, tx+ts, ty-0.14, "TN=8", fs=10, color=C_INK)
        brace_v(ax, tx-0.15, ty, ty+ts, "TM=8", fs=10, color=C_INK)

        # 输入 A/B 碎片
        cell(ax, 11.55, ty+0.2, 0.55, 2.0, fc=C_A, ec=C_A_DK, lw=1.5,
             hatch="////")
        ax.text(11.82, ty+2.35, "A frag", ha="center", fontsize=11, color=C_A_DK)
        cell(ax, tx+0.3, ty+ts+0.55, 2.4, 0.55, fc=C_B, ec=C_B_DK, lw=1.5,
             hatch="\\\\\\\\")
        ax.text(tx+ts+0.45, ty+ts+0.82, "B frag", ha="center", fontsize=11, color=C_B_DK)
        ax.text(12.3, ty+1.1, "dotIdx：\n8 个 A × 8 个 B\n外积进 8×8",
                ha="left", va="center", fontsize=10)

        ax.text(6.75, 1.2,
                "GEMM4 起：grid = (N/128, M/128)，block = 16×16 个线程，每线程 TM×TN = 8×8 个累加器；"
                "K 方向每次推进 BK=32",
                ha="center", fontsize=11)
        ax.text(6.75, 0.7,
                "GEMM12 进一步把 tile 放大到 BM=BN=128、TM=16、TN=8，每线程 128 个累加器，多数尺寸超过 cuBLAS FP32",
                ha="center", fontsize=11, color=C_ORANGE)
        save(fig, "sib_19_tiling_overview")


if __name__ == "__main__":
    fig1_hierarchy()
    fig2_naive_mem()
    fig3_coalescing()
    fig4_cache_blocking()
    fig5_1d_blocktiling()
    fig6_2d_blocktiling()
    fig7_gmem_loading()
    fig8_reg_blocking()
    fig9_arith_intensity()
    fig13_loop_structure()
    fig10_float4()
    fig11_bank_conflict()
    fig12_swizzle()
    fig14_double_buffer()
    fig15_cp_async()
    fig16_register_pipeline()
    fig17_two_stage_pipeline()
    fig18_memory_hierarchy()
    fig19_tiling_overview()
    print("完成")
