# Method

## Empirical P-Values

For each declination band, I start from a histogram of background pseudoexperiment TS values. The empirical p-value curve is the survival function:

```text
p(t) = count(TS >= t) / count(all trials)
```

This is computed directly from the reverse cumulative sum of the histogram counts.

## Gamma Tail Fit

The low-TS part of the distribution is not the region I want to use for extrapolating small p-values. It is affected by the `ns >= 0` boundary, finite-sample effects, and declination-dependent detector behaviour.

For a candidate cut `mu`, I keep only points where:

```text
TS >= mu
```

Then I fit a gamma survival function:

```text
P_gamma(TS >= t; shape, loc, scale)
```

The fit is performed in log10 p-value space. That makes the high-TS tail matter more than it would under a raw squared-error loss.

## Mu Optimisation

The default objective compares empirical and fitted p-values at fixed TS values:

```text
5, 9, 12.5, 16
```

For each candidate `mu`, the code:

1. fits the gamma survival model above `mu`
2. evaluates empirical and fitted p-values at the target TS values that remain above `mu`
3. minimises the mean squared difference in log10 p-value

There is also a `critical-ts` objective. It compares the empirical and fitted TS values corresponding to one-sided Gaussian tail probabilities at 3, 4, and 4.5 sigma.

## Values Below Mu

The gamma fit is intended for the tail. If I need p-values below the optimised cut, I use the empirical survival curve directly rather than forcing the analytic model into the distorted low-TS region.

## Why Gamma Survival Instead Of PDF-To-P-Value

The old exploratory work mixed density fits and survival-curve comparisons. This version keeps the definition consistent:

```text
empirical p-value  = empirical survival
fitted p-value     = gamma survival
optimisation loss  = distance between those p-values
```

That avoids fitting a PDF to a curve that is already an anti-cumulative probability.
