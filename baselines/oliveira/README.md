# Oliveira et al. baseline

This directory contains the validation artifacts for the unmodified
Oliveira et al. / DOCES innovation-probability baseline used in this
project.

## Model configuration

Network:
- directed Erdos-Renyi
- N = 1000
- target mean in-degree = 10.21

Dynamics:
- phi = pi / 2
- delta = 0.1
- feed size = 1
- iterations = 100000
- initial opinions sampled uniformly from [-1, 1]

Conditions:
- mu = 1.0
- mu = 0.1

## Fixed-state validation

A single network and initial opinion distribution were held fixed
across 50 stochastic simulation trajectories.

Seeds:
- network seed = 42
- opinion seed = 43
- simulation seeds = 0..49

For every simulation seed, both mu conditions use identical initial
conditions.

Aggregate results:

mu = 1.0
- mean BC = 0.1920
- median BC = 0.1749
- bimodal runs = 1/50 (2%)
- mean opinion standard deviation = 0.1817
- median maximum cascade size = 1

mu = 0.1
- mean BC = 0.4307
- median BC = 0.2106
- bimodal runs = 15/50 (30%)
- mean opinion standard deviation = 0.3177
- median maximum cascade size = 2232.5

Paired results:
- mean delta BC = +0.2386
- 72% of trajectories have higher BC at mu = 0.1
- mean delta opinion-sign assortativity = +0.0444
- mean delta opinion standard deviation = +0.1360
- median delta maximum cascade size = +2231.5

These results validate the qualitative sensitivity of information
diffusion and opinion dynamics to innovation probability. They do not
imply that every low-innovation realization becomes polarized.

## Representative realization

Simulation seed 6 was selected after the full 50-seed sweep as a
representative Figure-2-like realization for visualization.

mu = 1.0:
- BC = 0.2709
- opinion standard deviation = 0.2373
- positive fraction = 0.476
- negative fraction = 0.524
- maximum cascade size = 1

mu = 0.1:
- BC = 0.9840
- opinion standard deviation = 0.9676
- positive fraction = 0.513
- negative fraction = 0.487
- maximum cascade size = 3620

The representative trajectory is used only for visualization.
The 50-seed aggregate experiment is the basis for validation.

## Files

- fixed_state_summary.csv
  Aggregate statistics across the 50 fixed-state trajectories.

- paired_differences.csv
  Per-seed paired differences between mu = 0.1 and mu = 1.0.

- representative_summary.json
  Detailed output for representative simulation seed 6.

- bc_by_mu.png
  Distribution of bimodality coefficients across conditions.

- assortativity_by_mu.png
  Distribution of opinion-sign assortativity across conditions.

- paired_bc.png
  Paired change in bimodality for each simulation seed.

## Simulator version

doces-slop commit:

25f92b3c4f2ba33d3ef2f0bb7889f5d6a30058b1