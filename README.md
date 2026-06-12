# MuOptimiser

I use this project to calibrate p-values for IceCube public neutrino-data point-source searches.

The core problem is that the test-statistic background distribution changes with declination. Direct frequentist counting is reliable where there are many pseudoexperiments, but it becomes noisy in the high-TS tail where the p-values I care about are smallest. This code fits a gamma survival model to the tail of each declination band and optimises the low-TS cut, `mu`, used for that fit.

The method is based on my DESY work on p-value estimation for an unbinned likelihood analysis of IceCube public data. I keep a short research-context note here:

- [DESY research context](docs/desy-research.md)
- [Method notes](docs/method.md)

## What This Does

- loads TS histograms for declination or sin(declination) bands
- computes empirical survival curves, `P(TS >= t)`
- fits a gamma survival function above a candidate `mu` cut
- optimises `mu` for each declination band
- writes reproducible CSV/JSON outputs
- produces diagnostic plots for the fitted tails and `mu(sin(dec))`

## Data Formats

The main format is the one used in my cleaned IceCube histogram files:

```text
histogram_<sin_dec_start>_<sin_dec_stop>.npz
```

Each file should contain:

```text
hist       # one-dimensional TS-bin counts
bin_edges  # TS bin edges
```

I also support the older 2D format used during optimiser development:

```text
histogram_2d  # shape: declination bins x TS bins
x_edges       # declination bin edges or centres, in degrees
y_edges       # TS bin edges
```

## Install

```bash
python -m pip install -e .[dev]
```

## Run

For a directory of one-dimensional sin(declination) histograms:

```bash
mu-optimiser --input data/10y --output-dir outputs/mu_scan
```

For a single 2D histogram file:

```bash
mu-optimiser --input data/histogram_data_1000_bins.npz --format 2d --output-dir outputs/mu_scan
```

Useful options:

```bash
mu-optimiser \
  --input data/10y \
  --bounds 0.5 6.0 \
  --objective log-p \
  --target-ts 5 9 12.5 16 \
  --method bounded
```

The output directory contains:

```text
optimised_mu.csv
optimised_mu.json
mu_vs_sin_dec.png
fit_<index>.png
```

## Notes

The exploratory notebooks were useful while I was testing the analysis, but I keep this repo focused on the cleaned implementation: the Gnosis histogram format, the GammaMu optimiser idea, and one consistent p-value definition based on empirical and fitted survival functions.
