"""Export presentation figures from the verified common Ant evaluation CSV."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    with args.input.open() as file:
        rows = list(csv.DictReader(file))
    indexed = {row["label"]: row for row in rows}
    fourth = "4_curriculum_ray" if "4_curriculum_ray" in indexed else "4_ray_only"
    order = ["1_flat", "2_hard", "3_curriculum", fourth, "5_recovery"]
    if set(indexed) != set(order):
        raise ValueError("Expected the five verified development-stage model labels.")
    rows = [indexed[label] for label in order]
    for key in ("seed", "num_envs", "duration_limit_s", "torso_height_enabled", "shared_config_sha256", "terrain_mesh_sha256", "initial_state_sha256"):
        if len({row[key] for row in rows}) != 1:
            raise ValueError(f"Mixed evaluation conditions: {key}")
    if rows[0]["duration_limit_s"] != "16" or rows[0]["torso_height_enabled"] != "True":
        raise ValueError("This figure layout is labeled for 16s, torso_height ON.")
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if not font.is_file():
        raise FileNotFoundError("Install Noto Sans CJK, or change the font path for your PC.")
    font_manager.fontManager.addfont(str(font))
    family = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({
        "font.family": family, "font.size": 15, "axes.unicode_minus": False,
        "svg.fonttype": "path", "pdf.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    means = np.array([float(row["reward_mean"]) for row in rows])
    stds = np.array([float(row["reward_std"]) for row in rows])
    distances = np.array([float(row["grounded_distance_mean_m"]) for row in rows])
    fourth_name = "④ 커리큘럼\n+ RayCaster" if fourth == "4_curriculum_ray" else "④ RayCaster\n(커리큘럼 없음)"
    labels = ["① 평지 학습", "② 험지 학습", "③ 커리큘럼\n(센서 없음)", fourth_name, "⑤ Recovery\n(추가 보상 튜닝)"]
    colors = ["#99A8B6", "#507596", "#3E67AC", "#248A9B", "#124D78"]
    ink, muted = "#18344D", "#526575"
    condition = "L10 혼합 험지  ·  원본 Ant 보상  ·  seed 24  ·  100개 환경  ·  최대 16초  ·  높이 종료 ON"
    note = (
        f"체크포인트: ①~③ {rows[0]['checkpoint_iter']} / ④ {rows[3]['checkpoint_iter']} / ⑤ {rows[4]['checkpoint_iter']}. "
        "훈련 지형·보상·예산이 달라 개발 단계별 비교로 해석."
    )

    def style(ax):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#DDE5ED", linewidth=0.8)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#AAB7C3")
        ax.tick_params(axis="both", length=0, colors=muted)
        ax.tick_params(axis="x", pad=13, labelsize=13)
        ax.set_xticks(np.arange(5), labels)
        ax.set_xlim(-0.6, 4.6)

    def reward_plot(ax):
        style(ax)
        ax.bar(np.arange(5), means, width=0.58, color=colors,
               yerr=stds, error_kw={"ecolor": ink, "elinewidth": 1.7, "capsize": 5, "capthick": 1.7})
        ax.set_title("누적 보상 평균 ± 표준편차", fontsize=21, color=ink, loc="left", pad=22)
        ax.set_ylabel("누적 보상", fontsize=15, color=muted, labelpad=10)
        lower = min(0, float((means - stds).min())) - 1.0
        ax.set_ylim(lower, float((means + stds).max()) + 4.5)
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.axhline(0, color="#AAB7C3", linewidth=0.8)
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.65, f"{mean:.2f} ± {std:.2f}", ha="center", va="bottom",
                    fontsize=13, fontweight="bold", color=ink)

    def distance_plot(ax):
        style(ax)
        ax.bar(np.arange(5), distances, width=0.58, color=colors)
        ax.set_title("접지 최대 전진 거리 평균", fontsize=21, color=ink, loc="left", pad=22)
        ax.set_ylabel("전진 거리 (m)", fontsize=15, color=muted, labelpad=10)
        ax.set_ylim(0, float(distances.max()) * 1.27)
        ax.yaxis.set_major_locator(MultipleLocator(2))
        for i, distance in enumerate(distances):
            ax.text(i, distance + 0.20, f"{distance:.2f} m", ha="center", va="bottom",
                    fontsize=15, fontweight="bold", color=ink)

    def save(fig, stem):
        for ext in ("png", "svg", "pdf"):
            fig.savefig(args.output_dir / f"{stem}.{ext}", dpi=300, facecolor="white")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.subplots_adjust(left=0.065, right=0.975, top=0.76, bottom=0.245, wspace=0.29)
    fig.text(0.055, 0.92, "공통 평가 환경에서의 모델별 성능", fontsize=29, fontweight="bold", color=ink)
    fig.text(0.055, 0.855, condition, fontsize=15, color=muted)
    reward_plot(axes[0])
    distance_plot(axes[1])
    fig.text(0.055, 0.135, "오차막대: 100개 환경의 첫 에피소드 보상 표준편차.", fontsize=12, color=muted)
    fig.text(0.055, 0.098, "접지 최대 전진 거리: 발 접촉이 있는 시점에 도달한 최대 +X 거리.", fontsize=12, color=muted)
    fig.text(0.055, 0.05, note, fontsize=11.5, color=muted)
    save(fig, "ant_common_results_16s")

    for stem, draw, extra in (
        ("reward_mean_std", reward_plot, "오차막대: 100개 환경의 첫 에피소드 누적 보상 표준편차."),
        ("grounded_forward_mean", distance_plot, "발 접촉이 있는 시점에 도달한 최대 +X 거리의 환경별 평균."),
    ):
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.subplots_adjust(left=0.10, right=0.97, top=0.76, bottom=0.255)
        fig.text(0.075, 0.92, "공통 평가 결과", fontsize=28, fontweight="bold", color=ink)
        fig.text(0.075, 0.85, condition, fontsize=11.5, color=muted)
        draw(ax)
        fig.text(0.075, 0.12, extra, fontsize=11.5, color=muted)
        fig.text(0.075, 0.065, note, fontsize=9.4, color=muted)
        save(fig, stem)
    print(f"Saved combined and individual PNG / SVG / PDF figures to {args.output_dir}")


if __name__ == "__main__":
    main()
