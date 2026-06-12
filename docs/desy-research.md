# DESY Research Context

This repository comes from my DESY work on estimating p-values for an unbinned likelihood analysis of public IceCube neutrino data.

The analysis uses background-only pseudoexperiments to interpret a point-source test statistic:

```text
TS = -2 log(L(ns = 0) / L(ns_hat, gamma_hat))
```

For a tested sky position, the p-value is the probability that background gives a TS at least as large as the observed value:

```text
p(TS_obs, dec) = P_background(TS >= TS_obs | dec)
```

The important detail is the declination dependence. IceCube's detector acceptance and event rate change with declination, so one global TS distribution is not enough. I treat each declination band separately and calibrate the high-TS tail in that band.

The DESY presentation motivated three design choices used here:

1. Use empirical background survival curves from pseudoexperiments.
2. Avoid the distorted low-TS region by fitting only above a cut, `mu`.
3. Optimise `mu` per declination instead of relying only on a fixed global value.

The reference values from that work were:

```text
declination -30 deg -> optimised mu about 2.62
declination  +5 deg -> optimised mu about 3.20
declination +30 deg -> optimised mu about 3.10
```

Those declinations correspond roughly to:

```text
sin(dec) = -0.5
sin(dec) =  0.086
sin(dec) =  0.5
```

I use the same structure here: load the histograms, build empirical survival curves, fit the gamma tail, optimise `mu`, and write outputs that I can reproduce later.
