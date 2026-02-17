# Experiments

This directory contains the configuration and results for different
scientific experiments conducted using the `pasim` framework.

Each subdirectory represents a self-contained experiment,
tying together parameters, raw output, and processed results.
This structure ensures reproducibility and clear scientific bookkeeping.

## Note on Manuscript Lifespans

As of recent changes, manuscript death ticks are no longer configured via a fixed list in `params.yaml`.
Instead, the lifespan of each new manuscript is probabilistically sampled at its creation time,
based on its material and region, using a lognormal distribution. This enhances the
realism and flexibility of the simulation.
