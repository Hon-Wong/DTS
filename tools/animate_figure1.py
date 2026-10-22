#!/usr/bin/env python3
"""Animate Figure 1 with layer-wise arrows followed by a rotating view.

Uses the plot geometry and provenance in assets/figure1_paths.json.
Interpolation between the 13 saved vertices is for animation only.
No new experimental samples or measurements are generated.

Usage: python tools/animate_figure1.py
Dependencies: numpy, matplotlib, Pillow.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["Baseline", "Deep Supervision", "DTS (Ours)"]
COLORS = ["#e63946", "#1d3557", "#006d77", "#8338ec"]


def prefix(vertices, position):
    step = min(int(position), 12)
    result = vertices[:step + 1]
    if step < 12 and position > step:
        point = vertices[step] + (position - step) * (vertices[step + 1] - vertices[step])
        result = np.vstack([result, point])
    return result


def render(data_path, output_dir, preview_only=False):
    data = json.loads(data_path.read_text())
    # Put evidence on the same geometric scale as x/y so arrowheads stay small.
    evidence_scale = 15.
    floor = data["floor"] / evidence_scale
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    figure = plt.figure(figsize=(14, 5.4), dpi=100, facecolor="white")
    axes, artists = [], []
    for panel, method in enumerate(METHODS):
        ax = figure.add_axes([.005 + panel * .33, .15, .33, .73], projection="3d")
        ax.set(xlim=(-1.12, 1.12), ylim=(-1.12, 1.12), zlim=(floor, 32 / evidence_scale))
        ax.set_box_aspect((1, 1, 1.05), zoom=1.16)
        ax.set_axis_off()
        ax.view_init(elev=25, azim=-55)
        ax.set_title(method, y=1.00, pad=0, fontsize=17, fontweight="bold", color="#172b3a")
        theta = np.linspace(0, 2 * np.pi, 160)
        for radius in [1 / 3, 2 / 3, 1.]:
            ax.plot(radius * np.cos(theta), radius * np.sin(theta),
                    np.full_like(theta, floor), color="#cdd5db", lw=.7, alpha=.8)
        ax.plot([0, 0], [0, 0], [floor, 31 / evidence_scale], color="#536471", lw=.8)
        for z in [0, 15, 30]:
            ax.text(.035, .035, z / evidence_scale, str(z), color="#536471", fontsize=9)
        ax.text(0, 0, 33 / evidence_scale, "Evidence", ha="center", color="#536471", fontsize=10)
        for index, color in enumerate(COLORS):
            angle = index * np.pi / 2
            ax.text(1.12 * np.cos(angle), 1.12 * np.sin(angle), floor,
                    f"C{index + 1}", color=color, ha="center", fontsize=10)
        paths = []
        for trace in data["traces"]:
            if trace["method"] != method:
                continue
            vertices = np.asarray(trace["xyz"])
            vertices[:, 2] /= evidence_scale
            color = trace["color"]
            line, = ax.plot([], [], [], color=color, lw=1.0, alpha=.65)
            shadow, = ax.plot([], [], [], color=color, lw=1.2, alpha=.13)
            head, = ax.plot([], [], [], marker="o", ms=2.7, color=color,
                            markeredgewidth=0, linestyle="none", alpha=.9)
            paths.append((vertices, color, line, shadow, head))
        axes.append(ax)
        artists.append(paths)

    figure.text(.5, .96, "Deep Trajectory Supervision", ha="center", va="center",
                fontsize=21, fontweight="bold", color="#172b3a")
    status = figure.text(.5, .135, "", ha="center", fontsize=12, color="#172b3a")
    progress = figure.add_axes([.30, .085, .40, .011])
    progress.set(xlim=(0, 12), ylim=(0, 1))
    progress.axis("off")
    progress.axhline(.5, color="#e6ecef", lw=5, solid_capstyle="round")
    bar, = progress.plot([], [], color="#007d72", lw=5, solid_capstyle="round")
    figure.text(.5, .035, "ViT-Small / CIFAR-100     ·     Color: class     ·     Height: semantic evidence",
                ha="center", color="#627281", fontsize=10)

    arrows = [None] * 3

    def frame(position, azimuth, rotating=False):
        for panel, (ax, paths) in enumerate(zip(axes, artists)):
            ax.view_init(elev=25, azim=azimuth)
            starts, vectors, colors = [], [], []
            for vertices, color, line, shadow, head in paths:
                current = prefix(vertices, position)
                line.set_data_3d(current[:, 0], current[:, 1], current[:, 2])
                shadow.set_data_3d(current[:, 0], current[:, 1], np.full(len(current), floor))
                head.set_data_3d(current[-1:, 0], current[-1:, 1], current[-1:, 2])
                for start in range(0, 12, 2):
                    end = min(start + 2., position)
                    if end <= start + .05:
                        continue
                    tip = prefix(vertices, end)[-1]
                    starts.append(vertices[start])
                    vectors.append(tip - vertices[start])
                    colors.append((*color, .65))
            if arrows[panel] is not None:
                arrows[panel].remove()
                arrows[panel] = None
            if starts:
                starts, vectors = np.array(starts), np.array(vectors)
                # Each quiver has a shaft and two arrowhead edges.
                arrow_colors = colors + [color for color in colors for _ in range(2)]
                arrows[panel] = ax.quiver(*starts.T, *vectors.T, length=1,
                                         normalize=False, arrow_length_ratio=.14,
                                         linewidth=.7, colors=arrow_colors)
        if rotating:
            status.set_text("Layer 12 / 12  ·  Full trajectories")
        else:
            status.set_text(f"Layer {min(12, int(position))} / 12")
        bar.set_data([0, position], [.5, .5])
        figure.canvas.draw()
        return Image.fromarray(np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy())

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots = []
    for name, depth, angle in [("early", 3, -55), ("middle", 7, -55),
                                ("full", 12, -55), ("rotated", 12, 35)]:
        result = frame(depth, angle, name == "rotated")
        snapshots.append(result)
        if name == "full":
            result.save(output_dir / "figure1.png")
    if preview_only:
        contact = Image.new("RGB", (1400, 540), "white")
        for index, snapshot in enumerate(snapshots):
            contact.paste(snapshot.resize((700, 270)), ((index % 2) * 700, (index // 2) * 270))
        contact.save(output_dir / "figure1_preview.png")
        plt.close(figure)
        return

    # Build a stable palette from multiple views to avoid color flicker.
    palette_source = Image.new("RGB", (700, 1080), "white")
    for index, snapshot in enumerate(snapshots):
        palette_source.paste(snapshot.resize((700, 270)), (0, index * 270))
    palette = palette_source.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    frames, durations = [], []
    states = [(float(depth), -55., False, 80) for depth in np.linspace(0, 12, 49)]
    states[0] = (0., -55., False, 600)
    states[-1] = (12., -55., False, 1200)
    # A complete first frame also serves as a useful non-animated thumbnail.
    states.insert(0, (12., -55., False, 1200))
    states.extend((12., float(angle), True, 80) for angle in np.linspace(-55, 305, 91)[1:])
    states[-1] = (12., 305., True, 1600)
    for index, (depth, angle, rotating, duration) in enumerate(states):
        result = frame(depth, angle, rotating)
        frames.append(result.quantize(palette=palette, dither=Image.Dither.NONE))
        durations.append(duration)
        if index % 25 == 0:
            print(f"Rendered frame {index + 1}/{len(states)}", flush=True)
    frames[0].save(output_dir / "figure1.gif", save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True, disposal=1)
    plt.close(figure)
    print(f"Saved {output_dir / 'figure1.gif'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "assets/figure1_paths.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets")
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()
    render(args.data, args.output_dir, args.preview_only)


if __name__ == "__main__":
    main()
