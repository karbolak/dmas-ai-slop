# DMAS AI-Slop

Research repository for the University of Groningen **Design of Multi-Agent Systems (DMAS)** project on synthetic-content amplification in social-media networks.

This repository contains **experiments, configuration files, analysis, figures, and paper-related research code**.

The simulator itself lives in a separate repository:

```text
DMAS/
├── doces-slop/       # modified DOCES simulator
└── dmas-ai-slop/     # this repository
```

## Research question

> **Under what conditions does the interaction between high-volume synthetic content production, adaptive attention-seeking behaviour, and algorithmic recommendation lead to disproportionate synthetic-content exposure in a social-media network?**

The project builds primarily on:

1. Oliveira, de Arruda, and Moreno, *Mechanistic interplay between information spreading and opinion polarization*.
2. Tang et al., *Behind the “AI slop”: exploring the AI-generated viral short video production in China*.

Pendergrass, Johnson, and Bacarella, *A strategic cycle of slop*, provides supporting conceptual grounding for the attention-reward feedback loop.

---

## Current goal

Before modifying the DOCES simulator, we first reproduce a qualitative baseline from Oliveira et al.

The target result is the paper's comparison in which the same network and initial state can produce qualitatively different outcomes under different innovation probabilities:

- `mu = 1.0`
- `mu = 0.1`

Oliveira et al. use an Erdős-Rényi network with:

- `N = 1000`
- mean in-degree `z = 10.21`
- initial opinions sampled uniformly from `[-1, 1]`

The main paper does **not fully specify every parameter needed to reproduce Figure 2 exactly**. Therefore, the provided configuration should currently be treated as a **qualitative replication scaffold**, not an exact reproduction claim. Exact reproduction should use the paper's Supporting Information and/or the parameterisation from the referenced earlier model.

---

# Quick start

## Environment setup

The project uses Python 3.12 and Conda. DOCES contains a native C
extension, so a C compiler is also required.

Create the environment:

```bash
    conda env create -f environment.yml
    conda activate dmas-ai-slop
```

Verify Python:

```bash
    python --version
```

Expected:

```bash
    Python 3.12.x
```

Because DOCES must compile its native extension, verify that the Conda
compiler is available.

After creating or updating the environment, it may help to reactivate it:

```bash
    conda deactivate
    conda activate dmas-ai-slop
```

On Linux, inspect the available compiler with:

```bash
    ls $CONDA_PREFIX/bin | grep -E 'gcc|cc$'
```

### Fish shell

If the `CC` environment variable is empty, set it explicitly to the
compiler available inside the Conda environment.

For example:

```bash
    set -gx CC $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc
```

Then verify:

```bash
    $CC --version
```

If that exact compiler name does not exist, use the one returned by:

```bash
    ls $CONDA_PREFIX/bin | grep -E 'gcc|cc$'
```

### Bash / Zsh

If necessary, set the compiler with:

```bash
    export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
```

Then install the local DOCES fork:

```bash
    pip install -e ../doces-slop
```

Verify the installation:

```bash
    python -c "import doces; print('DOCES import works')"
```

Expected output:

```bash
    DOCES import works
```

Finally, check the full research environment:

```bash
    python scripts/check_setup.py
```

### Troubleshooting: DOCES build fails with `No such file or directory: 'gcc'`

DOCES includes a compiled C extension. If:

```bash
    pip install -e ../doces-slop
```

fails with:

```bash
    error: [Errno 2] No such file or directory: 'gcc'
```

first reactivate the Conda environment:

```bash
    conda deactivate
    conda activate dmas-ai-slop
```

Then inspect the installed compiler:

```bash
    ls $CONDA_PREFIX/bin | grep -E 'gcc|cc$'
```

For fish, set the compiler explicitly if necessary:

```bash
    set -gx CC $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc
```

Then retry:

```bash
    pip install -e ../doces-slop
```


Do not fix this by upgrading Python or NumPy independently. The current
DOCES version depends on `numpy<2.0`, and this project uses Python 3.12
for compatibility.

## 4. Check the research environment

```bash
python scripts/check_setup.py
```

## 5. Run the Oliveira baseline experiment

```bash
python experiments/reproduce_oliveira.py \
    --config configs/oliveira_qualitative.yaml
```

Outputs are written to:

```text
results/oliveira_qualitative/
figures/oliveira_qualitative/
```

The experiment runs the same generated network and initial opinion vector twice, changing only `mu`.

---

# Repository layout

```text
dmas-ai-slop/
├── README.md
├── environment.yml
├── .gitignore
├── configs/
│   └── oliveira_qualitative.yaml
├── experiments/
│   └── reproduce_oliveira.py
├── scripts/
│   └── check_setup.py
├── analysis/
│   └── README.md
├── figures/
│   └── .gitkeep
└── results/
    └── .gitkeep
```

---

# Experimental principles

We want the project to remain reproducible and interpretable.

1. **Do not modify DOCES before establishing the baseline.**
2. Use the **same network, initial opinions, and seed** when comparing model conditions.
3. Change one mechanism at a time.
4. Store experiment parameters with the results.
5. Do not hard-code publication figures into analysis scripts.
6. Separate simulator changes (`doces-slop`) from experiments (`dmas-ai-slop`).
7. Do not call a result an exact replication unless every relevant parameter matches the source.
8. Treat a "tipping point" or "phase transition" as an empirical finding to test for, not an assumption.

---

# Planned progression

- [x] Build and smoke-test unmodified DOCES
- [ ] Reproduce Oliveira et al. qualitatively
- [ ] Resolve exact Figure 2 parameterisation from Supporting Information
- [ ] Freeze/tag baseline
- [ ] Add producer type to DOCES
- [ ] Add heterogeneous activation weights
- [ ] Add content provenance
- [ ] Compare synthetic production share vs. exposure share
- [ ] Add adaptive synthetic producers
- [ ] Test interaction with recommendation strength
- [ ] Run ablations
- [ ] Test interventions

---

# Notes on the Oliveira baseline

The qualitative experiment intentionally keeps network structure and initial opinions identical between conditions.

The main paper reports that:

- the network has `N = 1000`,
- mean in-degree is `z = 10.21`,
- initial opinions are sampled uniformly from `[-1, 1]`,
- `mu = 1` corresponds to the original model,
- changing only `mu` to `0.1` can produce a polarised two-community state.

The paper also describes trajectories sampled every `10N` model iterations in a later characterisation experiment.

The exact Figure 2 parameterisation is not fully stated in the main article text available to us. Until that is resolved, use this repository to validate the experiment pipeline and qualitative behaviour rather than to claim numerical replication.

---

# Results policy

Generated results and figures are ignored by Git by default, except placeholder `.gitkeep` files.

For publication-quality experiments, every run should eventually record:

- Git commit
- configuration
- random seed
- timestamp
- simulator version
- summary statistics

This metadata layer will be added once the baseline experiment is stable.
