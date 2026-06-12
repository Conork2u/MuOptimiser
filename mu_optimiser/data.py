from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np


HISTOGRAM_FILE_RE = re.compile(
    r"histogram_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)\.npz$"
)


@dataclass(frozen=True)
class Histogram1D:
    """One TS histogram for one declination or sin(declination) band."""

    counts: np.ndarray
    bin_edges: np.ndarray
    sin_dec_start: float | None = None
    sin_dec_stop: float | None = None
    source: Path | None = None

    def __post_init__(self) -> None:
        counts = np.asarray(self.counts, dtype=float)
        bin_edges = np.asarray(self.bin_edges, dtype=float)

        if counts.ndim != 1:
            raise ValueError("counts must be one-dimensional")
        if bin_edges.ndim != 1:
            raise ValueError("bin_edges must be one-dimensional")
        if bin_edges.size != counts.size + 1:
            raise ValueError("bin_edges must have exactly one more entry than counts")
        if np.any(counts < 0):
            raise ValueError("counts must be non-negative")
        if np.any(np.diff(bin_edges) <= 0):
            raise ValueError("bin_edges must be strictly increasing")

        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "bin_edges", bin_edges)

    @property
    def bin_centers(self) -> np.ndarray:
        return 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

    @property
    def bin_centres(self) -> np.ndarray:
        return self.bin_centers

    @property
    def total(self) -> float:
        return float(np.sum(self.counts))

    @property
    def sin_dec_mid(self) -> float | None:
        if self.sin_dec_start is None or self.sin_dec_stop is None:
            return None
        return 0.5 * (self.sin_dec_start + self.sin_dec_stop)

    @property
    def dec_mid_deg(self) -> float | None:
        if self.sin_dec_mid is None:
            return None
        return float(np.degrees(np.arcsin(np.clip(self.sin_dec_mid, -1.0, 1.0))))


def load_histogram(path: str | Path) -> Histogram1D:
    """Load a Gnosis-style one-dimensional TS histogram."""

    path = Path(path)
    with np.load(path) as data:
        missing = {"hist", "bin_edges"} - set(data.files)
        if missing:
            raise KeyError(f"{path} is missing keys: {sorted(missing)}")
        counts = data["hist"]
        bin_edges = data["bin_edges"]

    sin_dec_start = None
    sin_dec_stop = None
    match = HISTOGRAM_FILE_RE.search(path.name)
    if match:
        sin_dec_start = float(match.group(1))
        sin_dec_stop = float(match.group(2))

    return Histogram1D(
        counts=counts,
        bin_edges=bin_edges,
        sin_dec_start=sin_dec_start,
        sin_dec_stop=sin_dec_stop,
        source=path,
    )


def load_histograms(directory: str | Path) -> list[Histogram1D]:
    """Load all Gnosis-style histograms from a directory."""

    directory = Path(directory)
    histograms = [load_histogram(path) for path in directory.glob("histogram_*.npz")]

    def sort_key(histogram: Histogram1D) -> tuple[float, str]:
        sin_dec = histogram.sin_dec_mid
        return (float("inf") if sin_dec is None else sin_dec, histogram.source.name)

    return sorted(histograms, key=sort_key)


def load_2d_histograms(path: str | Path) -> list[Histogram1D]:
    """Load a GammaMu-style 2D declination-by-TS histogram as 1D bands."""

    path = Path(path)
    with np.load(path) as data:
        missing = {"histogram_2d", "x_edges", "y_edges"} - set(data.files)
        if missing:
            raise KeyError(f"{path} is missing keys: {sorted(missing)}")
        histogram_2d = np.asarray(data["histogram_2d"], dtype=float)
        x_values = np.asarray(data["x_edges"], dtype=float)
        y_edges = np.asarray(data["y_edges"], dtype=float)

    if histogram_2d.ndim != 2:
        raise ValueError("histogram_2d must be two-dimensional")

    dec_edges = _declination_edges_from_values(x_values, histogram_2d.shape[0])
    histograms: list[Histogram1D] = []
    for row_index, counts in enumerate(histogram_2d):
        dec_start = dec_edges[row_index]
        dec_stop = dec_edges[row_index + 1]
        sin_start = float(np.sin(np.deg2rad(dec_start)))
        sin_stop = float(np.sin(np.deg2rad(dec_stop)))
        histograms.append(
            Histogram1D(
                counts=counts,
                bin_edges=y_edges,
                sin_dec_start=min(sin_start, sin_stop),
                sin_dec_stop=max(sin_start, sin_stop),
                source=path,
            )
        )

    return histograms


def _declination_edges_from_values(values: np.ndarray, n_rows: int) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError("x_edges must be one-dimensional")
    if values.size == n_rows + 1:
        return values
    if values.size != n_rows:
        raise ValueError(
            "x_edges must contain either declination bin edges or one centre per row"
        )

    centers = values
    if centers.size == 1:
        width = 1.0
        return np.array([centers[0] - 0.5 * width, centers[0] + 0.5 * width])

    midpoints = 0.5 * (centers[:-1] + centers[1:])
    first_width = centers[1] - centers[0]
    last_width = centers[-1] - centers[-2]
    return np.concatenate(
        [
            [centers[0] - 0.5 * first_width],
            midpoints,
            [centers[-1] + 0.5 * last_width],
        ]
    )


def histogram_sources(histograms: Iterable[Histogram1D]) -> list[str]:
    return ["" if histogram.source is None else str(histogram.source) for histogram in histograms]
