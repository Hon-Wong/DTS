#!/usr/bin/env python3
"""Recover the plotted geometry of Figure 1 from the published vector PDF.

This recovers the figure's 3-D display coordinates, not model activations,
logits, sample IDs, or an original experiment dump. Floor shadows determine
the horizontal coordinates; the evidence axis determines the vertical scale.
Every recovered vertex is checked against its source PDF projection.

Usage:
  python tools/recover_figure1.py wang26ib.pdf --output assets/figure1_paths.json

Dependencies: numpy, scipy, PyMuPDF.
"""
import argparse
import hashlib
import json
from pathlib import Path

import fitz
import numpy as np
from scipy.optimize import least_squares

SOURCE_URL = "https://raw.githubusercontent.com/mlresearch/v306/main/assets/wang26ib/wang26ib.pdf"
METHODS = ["Baseline", "Deep Supervision", "DTS (Ours)"]


def points(path):
    if not all(item[0] == "l" for item in path["items"]):
        raise ValueError("Expected a polyline in the source figure")
    return np.array([tuple(path["items"][0][1])] + [tuple(i[2]) for i in path["items"]])


def transform(matrix, vertices):
    projected = np.column_stack([vertices, np.ones(len(vertices))]) @ matrix.T
    return projected[:, :2] / projected[:, 2:]


def recover(pdf):
    with fitz.open(pdf) as document:
        paths = document[0].get_drawings()
    shadows = [p for p in paths if p["type"] == "s" and len(p["items"]) == 12
               and abs(p["stroke_opacity"] - .15) < .001]
    traces = [p for p in paths if p["type"] == "s" and len(p["items"]) == 12
              and abs(p["stroke_opacity"] - .4) < .001]
    if len(shadows) != 120 or len(traces) != 120:
        raise ValueError("This PDF does not have the expected Figure 1 paths")

    origin = points(shadows[0])[0]
    rings = [points(p) for p in paths if p["type"] == "s" and len(p["items"]) > 50
             and p["color"] == (0., 0., 0.)][:3]
    # Equal-radius intervals in the original plot, with the outside ring set to 1.
    radii = [1 / 3, 2 / 3, 1.]

    def homography(parameters):
        affine = parameters[:4].reshape(2, 2)
        perspective = parameters[4:]
        return np.block([
            [affine + origin[:, None] * perspective[None, :], origin[:, None]],
            [perspective[None, :], np.ones((1, 1))],
        ])

    def residual(parameters):
        matrix = homography(parameters)
        inverse = np.linalg.inv(matrix)
        errors = [(np.linalg.norm(transform(inverse, ring), axis=1) - radius) * 50
                  for ring, radius in zip(rings, radii)]
        # The first vertex of each ring fixes the otherwise arbitrary rotation.
        errors.append((transform(matrix, np.array([[r, 0] for r in radii]))
                       - np.array([ring[0] for ring in rings])).ravel())
        return np.concatenate(errors)

    fit = least_squares(residual, [48, 32, 14, -21, 0, 0],
                        xtol=1e-13, gtol=1e-13, ftol=1e-13)
    matrix = homography(fit.x)
    if not fit.success or np.max(np.abs(fit.fun)) > .001:
        raise ValueError("Floor-circle calibration failed")

    # Intersect the three tick segments with the central axis: evidence 0, 15, 30.
    ticks = [p for p in paths if p["type"] == "s" and len(p["items"]) == 1
             and p["color"] == (0., 0., 0.)
             and abs(p["width"] - .418305) < 1e-4][:3]
    tick_y = []
    for tick in ticks:
        a, b = points(tick)
        tick_y.append(a[1] + (b[1] - a[1]) * (origin[0] - a[0]) / (b[0] - a[0]))
    evidence = np.array([0., 15., 30.])
    intercept, slope, perspective = np.linalg.solve(
        np.column_stack([np.ones(3), evidence, -evidence * tick_y]), tick_y)
    floor = (origin[1] - intercept) / (slope - origin[1] * perspective)
    vertical = np.array([origin[0] * perspective, slope, perspective]) / (1 + perspective * floor)

    recovered, errors = [], []
    for index, (shadow, trace) in enumerate(zip(shadows, traces)):
        if shadow["color"] != trace["color"]:
            raise ValueError("Shadow and trajectory colors do not match")
        floor_points, source_points = points(shadow), points(trace)
        panel_shift = floor_points[0] - origin
        xy = transform(np.linalg.inv(matrix), floor_points - panel_shift)
        projected_floor = np.column_stack([xy, np.ones(len(xy))]) @ matrix.T
        target = source_points - panel_shift
        coefficients = vertical[:2] - target * vertical[2]
        rhs = target * projected_floor[:, 2:] - projected_floor[:, :2]
        height = np.sum(coefficients * rhs, axis=1) / np.sum(coefficients ** 2, axis=1)
        projected = projected_floor + height[:, None] * vertical
        projected = projected[:, :2] / projected[:, 2:] + panel_shift
        errors.extend(np.linalg.norm(projected - source_points, axis=1))
        recovered.append({
            "method": METHODS[index // 40],
            "color": list(trace["color"]),
            "xyz": np.column_stack([xy, height + floor]).round(7).tolist(),
        })
    max_error = float(np.max(errors))
    if max_error > .001:
        raise ValueError(f"Reprojection error too large: {max_error} PDF points")

    return {
        "source": {"url": SOURCE_URL, "figure": 1, "page": 1,
                   "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest()},
        "data_kind": "3-D plot coordinates reconstructed from the published vector figure",
        "not_included": ["original activations", "original logits", "sample IDs"],
        "coordinates": {"xy": "figure coordinates; outer floor ring has radius 1",
                        "z": "semantic evidence, calibrated from the 0, 15, 30 axis ticks"},
        "vertices_per_path": 13,
        "paths_per_method": 40,
        "floor": float(floor),
        "validation": {"reprojection_max_pdf_points": max_error,
                       "reprojection_rms_pdf_points": float(np.sqrt(np.mean(np.square(errors)))),
                       "circle_calibration_max_residual": float(np.max(np.abs(fit.fun)))},
        "traces": recovered,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, default=Path("assets/figure1_paths.json"))
    args = parser.parse_args()
    data = recover(args.pdf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data["validation"], indent=2))


if __name__ == "__main__":
    main()
