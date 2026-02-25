# Experiments

This directory contains the configuration and results for different
scientific experiments conducted using the `pasim` framework.

Each subdirectory represents a self-contained experiment,
tying together parameters, raw output, and processed results.
This structure ensures reproducibility and clear scientific bookkeeping.

## Demand Schedule

The demand for new manuscripts is now specified as an *aggregate* demand per tick across all regions,
rather than broken down by individual regions in the `params.yaml`. The simulation engine internally
distributes this aggregate demand deterministically across regions based on historical allocation rules
that vary by century.

**Allocation Rules by Century (1 tick = 1 year, century = floor(tick / 100)):**
*   **Centuries 0-2 (0-299 years)**: Asia Minor: 70%, Levant: 25%, Egypt: 5%
*   **Centuries 3-5 (300-599 years)**: Asia Minor: 55%, Levant: 25%, Egypt: 20%
*   **Century 6 onwards (>= 600 years)**: Asia Minor: 100%, Levant: 0%, Egypt: 0%

**Rounding Rule:**
When computing regional demand, each region's demand is rounded *up* (ceiling) independently.
This means the total allocated demand across all regions may slightly exceed the initial
aggregate demand. This is an intentional design choice to ensure minimum demand is met
without complex re-normalization.

## Note on Manuscript Lifespans

As of recent changes, manuscript death ticks are no longer configured via a fixed list in `params.yaml`.
Instead, the lifespan of each new manuscript is probabilistically sampled at its creation time,
based on its material and region, using a lognormal distribution. This enhances the
realism and flexibility of the simulation.
