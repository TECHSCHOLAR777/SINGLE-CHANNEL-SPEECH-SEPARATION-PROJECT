"""Generate P3 speaker-count confusion and calibration artifacts from RunLog.

The report is intentionally dependency-free beyond the project's NumPy stack:
JSON and CSV are machine-readable, Markdown is reviewable in GitHub, and SVG
plots can be embedded directly in the final report without notebook state.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from eval.reporting import RunLog, RunRecord, calibration_curve, counting_report


@dataclass(frozen=True)
class CountingReportArtifacts:
    """Paths produced by :func:`generate_counting_report`."""

    summary_json: Path
    confusion_csv: Path
    calibration_csv: Path | None
    markdown: Path
    confusion_svg: Path
    calibration_svg: Path | None


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_confusion_csv(
    path: Path,
    labels: Sequence[int],
    matrix: Sequence[Sequence[int]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\estimated", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *row])


def _write_calibration_csv(path: Path, calibration: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bin_center", "mean_confidence", "empirical_accuracy", "count"])
        rows = zip(
            calibration["bin_centers"],
            calibration["bin_confidence"],
            calibration["bin_accuracy"],
            calibration["bin_counts"],
            strict=True,
        )
        for center, confidence, accuracy, count in rows:
            writer.writerow(
                [
                    center,
                    "" if not math.isfinite(float(confidence)) else confidence,
                    "" if not math.isfinite(float(accuracy)) else accuracy,
                    count,
                ]
            )


def _svg_document(width: int, height: int, body: Sequence[str]) -> str:
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                f'height="{height}" viewBox="0 0 {width} {height}">'
            ),
            '<rect width="100%" height="100%" fill="white"/>',
            *body,
            "</svg>",
        ]
    )


def _write_confusion_svg(
    path: Path,
    labels: Sequence[int],
    matrix: Sequence[Sequence[int]],
) -> None:
    n = len(labels)
    cell = 70
    left = 105
    top = 75
    width = left + n * cell + 25
    height = top + n * cell + 45
    maximum = max((value for row in matrix for value in row), default=0)
    body = [
        '<text x="20" y="28" font-size="20" font-family="sans-serif">Count confusion matrix</text>',
        (
            f'<text x="{left + n * cell / 2:.1f}" y="52" text-anchor="middle" '
            'font-size="14" font-family="sans-serif">Estimated speaker count</text>'
        ),
        (
            f'<text x="22" y="{top + n * cell / 2:.1f}" text-anchor="middle" '
            'font-size="14" font-family="sans-serif" '
            'transform="rotate(-90 22 '
            f'{top + n * cell / 2:.1f})">True speaker count</text>'
        ),
    ]
    for col, label in enumerate(labels):
        x = left + col * cell + cell / 2
        body.append(
            f'<text x="{x:.1f}" y="{top - 12}" text-anchor="middle" '
            f'font-size="13" font-family="sans-serif">{label}</text>'
        )
    for row_index, (label, row) in enumerate(zip(labels, matrix, strict=True)):
        y = top + row_index * cell
        body.append(
            f'<text x="{left - 18}" y="{y + cell / 2 + 5:.1f}" text-anchor="middle" '
            f'font-size="13" font-family="sans-serif">{label}</text>'
        )
        for col_index, value in enumerate(row):
            x = left + col_index * cell
            intensity = 0.0 if maximum == 0 else float(value) / maximum
            shade = int(round(255 - 150 * intensity))
            body.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="rgb({shade},{shade},{shade})" stroke="black"/>'
            )
            body.append(
                f'<text x="{x + cell / 2:.1f}" y="{y + cell / 2 + 5:.1f}" '
                f'text-anchor="middle" font-size="15" font-family="sans-serif">{value}</text>'
            )
    path.write_text(_svg_document(width, height, body), encoding="utf-8")


def _write_calibration_svg(path: Path, calibration: dict[str, Any]) -> None:
    width = 620
    height = 520
    left = 75
    top = 55
    size = 400

    def point(confidence: float, accuracy: float) -> tuple[float, float]:
        return left + confidence * size, top + (1.0 - accuracy) * size

    body = [
        '<text x="20" y="28" font-size="20" font-family="sans-serif">Count calibration</text>',
        f'<rect x="{left}" y="{top}" width="{size}" height="{size}" fill="none" stroke="black"/>',
        (
            f'<line x1="{left}" y1="{top + size}" x2="{left + size}" y2="{top}" '
            'stroke="gray" stroke-dasharray="6 5"/>'
        ),
        (
            f'<text x="{left + size / 2}" y="{top + size + 42}" text-anchor="middle" '
            'font-size="14" font-family="sans-serif">Mean confidence</text>'
        ),
        (
            f'<text x="24" y="{top + size / 2}" text-anchor="middle" font-size="14" '
            f'font-family="sans-serif" transform="rotate(-90 24 {top + size / 2})">Accuracy</text>'
        ),
        (
            f'<text x="{left + size + 25}" y="{top + 18}" font-size="13" '
            f'font-family="sans-serif">ECE: {calibration["ece"]:.4f}</text>'
        ),
    ]
    plotted: list[tuple[float, float]] = []
    rows = zip(
        calibration["bin_confidence"],
        calibration["bin_accuracy"],
        calibration["bin_counts"],
        strict=True,
    )
    for confidence, accuracy, count in rows:
        if count and math.isfinite(float(confidence)) and math.isfinite(float(accuracy)):
            plotted.append(point(float(confidence), float(accuracy)))
    if plotted:
        coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in plotted)
        body.append(
            f'<polyline points="{coordinates}" fill="none" stroke="black" stroke-width="2"/>'
        )
        for x, y in plotted:
            body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="black"/>')
    path.write_text(_svg_document(width, height, body), encoding="utf-8")


def _markdown(summary: dict[str, Any]) -> str:
    labels = summary["counting"]["labels"]
    matrix = summary["counting"]["confusion"]
    lines = [
        "# Speaker-count report",
        "",
        f"- Records: **{summary['n_records']}**",
        f"- Exact count accuracy: **{summary['counting']['accuracy']:.2%}**",
        "",
        "## Confusion matrix",
        "",
        "Rows are true count; columns are estimated count.",
        "",
        "| true \\ estimated | " + " | ".join(str(label) for label in labels) + " |",
        "|---|" + "---|" * len(labels),
    ]
    for label, row in zip(labels, matrix, strict=True):
        lines.append(f"| {label} | " + " | ".join(str(value) for value in row) + " |")
    calibration = summary.get("calibration")
    lines.extend(["", "## Calibration", ""])
    if calibration is None:
        lines.append("No finite count-confidence values were present in the run log.")
    else:
        lines.append(f"Expected calibration error (ECE): **{calibration['ece']:.4f}**")
    return "\n".join(lines) + "\n"


def generate_counting_report(
    records: Sequence[RunRecord],
    output_dir: str | Path,
    *,
    count_range: tuple[int, int] = (2, 5),
    n_bins: int = 10,
) -> CountingReportArtifacts:
    """Generate all P3-C3/C4 artifacts from canonical run records."""
    if not records:
        raise ValueError("cannot generate counting report from an empty record list")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    count_stats = counting_report(records, count_range=count_range)

    confidence_rows = [
        record
        for record in records
        if math.isfinite(float(record.count_confidence))
        and 0.0 <= float(record.count_confidence) <= 1.0
    ]
    calibration: dict[str, Any] | None = None
    if confidence_rows:
        calibration = calibration_curve(
            [float(record.count_confidence) for record in confidence_rows],
            [record.n_true == record.n_estimated for record in confidence_rows],
            n_bins=n_bins,
        )

    summary = {
        "n_records": len(records),
        "count_range": list(count_range),
        "counting": count_stats,
        "calibration": calibration,
    }
    summary_json = out / "counting_summary.json"
    summary_json.write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    confusion_csv = out / "count_confusion_matrix.csv"
    _write_confusion_csv(confusion_csv, count_stats["labels"], count_stats["confusion"])
    confusion_svg = out / "count_confusion_matrix.svg"
    _write_confusion_svg(confusion_svg, count_stats["labels"], count_stats["confusion"])

    calibration_csv: Path | None = None
    calibration_svg: Path | None = None
    if calibration is not None:
        calibration_csv = out / "count_calibration_curve.csv"
        _write_calibration_csv(calibration_csv, calibration)
        calibration_svg = out / "count_calibration_curve.svg"
        _write_calibration_svg(calibration_svg, calibration)

    markdown = out / "counting_report.md"
    markdown.write_text(_markdown(summary), encoding="utf-8")
    return CountingReportArtifacts(
        summary_json=summary_json,
        confusion_csv=confusion_csv,
        calibration_csv=calibration_csv,
        markdown=markdown,
        confusion_svg=confusion_svg,
        calibration_svg=calibration_svg,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-log", required=True, help="RunLog JSONL path")
    parser.add_argument("--output-dir", default="outputs/counting_report")
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--max-count", type=int, default=5)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    records = RunLog(args.run_log).load()
    artifacts = generate_counting_report(
        records,
        args.output_dir,
        count_range=(args.min_count, args.max_count),
        n_bins=args.bins,
    )
    printable = {
        name: str(value) if value is not None else None for name, value in asdict(artifacts).items()
    }
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
