from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data import Histogram1D, load_2d_histograms, load_histogram, load_histograms
from .optimise import DEFAULT_MU_BOUNDS, DEFAULT_TARGET_TS, MuOptimisationResult, optimise_mu
from .plotting import plot_mu_by_sin_dec, plot_tail_fit


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    histograms = load_input(args.input, args.format)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[MuOptimisationResult] = []
    failures: list[dict[str, str]] = []

    for index, histogram in enumerate(histograms):
        try:
            result = optimise_mu(
                histogram,
                bounds=tuple(args.bounds),
                objective=args.objective,
                method=args.method,
                target_ts_values=args.target_ts,
                sigma_levels=args.sigma_levels,
                min_tail_points=args.min_tail_points,
                fit_loc=args.fit_loc,
                random_state=args.random_state,
            )
        except Exception as exc:  # noqa: BLE001 - CLI should report every failed bin.
            failures.append(
                {
                    "index": str(index),
                    "source": "" if histogram.source is None else str(histogram.source),
                    "error": str(exc),
                }
            )
            continue

        results.append(result)
        figure_path = output_dir / f"fit_{index:04d}.png"
        fig, _ = plot_tail_fit(histogram, result, figure_path)
        plt.close(fig)

    if not results:
        write_failures(output_dir / "failures.json", failures)
        raise SystemExit("no declination bins were optimised successfully")

    records = [result.to_record() for result in results]
    write_csv(output_dir / "optimised_mu.csv", records)
    write_json(output_dir / "optimised_mu.json", {"results": records, "failures": failures})
    write_failures(output_dir / "failures.json", failures)

    if any(result.sin_dec_mid is not None for result in results):
        fig, _ = plot_mu_by_sin_dec(results, output_dir / "mu_vs_sin_dec.png")
        plt.close(fig)

    print(f"Optimised {len(results)} declination bins")
    if failures:
        print(f"Skipped {len(failures)} bins; see {output_dir / 'failures.json'}")
    print(f"Wrote outputs to {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimise gamma-tail mu cuts for IceCube TS histograms."
    )
    parser.add_argument("--input", required=True, type=Path, help="Histogram directory or .npz file.")
    parser.add_argument(
        "--format",
        choices=("auto", "1d", "2d"),
        default="auto",
        help="Input format. Auto detects a directory, Gnosis-style 1D file, or GammaMu-style 2D file.",
    )
    parser.add_argument("--output-dir", default="outputs/mu_scan", type=Path)
    parser.add_argument("--bounds", nargs=2, type=float, default=list(DEFAULT_MU_BOUNDS))
    parser.add_argument("--objective", choices=("log-p", "critical-ts"), default="log-p")
    parser.add_argument("--method", choices=("bounded", "differential_evolution"), default="bounded")
    parser.add_argument("--target-ts", nargs="*", type=float, default=list(DEFAULT_TARGET_TS))
    parser.add_argument("--sigma-levels", nargs="*", type=float, default=[3.0, 4.0, 4.5])
    parser.add_argument("--min-tail-points", type=int, default=8)
    parser.add_argument("--fit-loc", action="store_true")
    parser.add_argument("--random-state", type=int, default=0)
    return parser


def load_input(input_path: Path, input_format: str) -> list[Histogram1D]:
    if input_format == "1d":
        return load_histograms(input_path) if input_path.is_dir() else [load_histogram(input_path)]
    if input_format == "2d":
        return load_2d_histograms(input_path)

    if input_path.is_dir():
        return load_histograms(input_path)

    with np.load(input_path) as data:
        keys = set(data.files)
    if {"histogram_2d", "x_edges", "y_edges"} <= keys:
        return load_2d_histograms(input_path)
    return [load_histogram(input_path)]


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    fieldnames = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_failures(path: Path, failures: list[dict[str, str]]) -> None:
    write_json(path, failures)


if __name__ == "__main__":
    raise SystemExit(main())
