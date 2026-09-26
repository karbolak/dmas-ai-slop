#!/usr/bin/env fish

set CONFIG configs/volume_sweep_big.yaml
set RUNS results/volume_sweep_big/runs.csv

mkdir -p logs
mkdir -p results/volume_sweep_big
mkdir -p figures/volume_sweep_big

for N in 2000 5000 10000
    echo ""
    echo "============================================================"
    echo "Starting checkpoint: $N worlds"
    date
    echo "============================================================"

    python -u experiments/alpha_volume_sweep.py \
        --config $CONFIG \
        --replicates $N

    or begin
        echo "Simulation failed at checkpoint $N"
        date
        exit 1
    end

    python -u analysis/analyze_alpha_volume_sweep.py \
        --input $RUNS \
        --summary results/volume_sweep_big/summary_$N.csv \
        --paired results/volume_sweep_big/paired_activity_effects_$N.csv \
        --output-dir figures/volume_sweep_big/$N

    or begin
        echo "Analysis failed at checkpoint $N"
        date
        exit 1
    end

    echo "Finished checkpoint: $N worlds"
    date
end

echo ""
echo "============================================================"
echo "BIG SWEEP COMPLETE"
date
echo "============================================================"

date > results/volume_sweep_big/COMPLETE.txt