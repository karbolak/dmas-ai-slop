# Report versions

`DMAS-6-adaptive.tex` is the extended report, with the implemented receipt-responsive producer policy and completed 200-world, 7,200-run experiment. Its bibliography is `bibliography-adaptive.bib`. It includes `adaptive_discussion.tex`, the three `adaptive_*_table.tex` files, `feed_capacity_table.tex`, and images under `imgs/`.

`DMAS-5-final.tex`, its PDF and `bibliography.bib` are the validated activity-only backup. That version is preserved on `backup/dmas-5-validated` in both repositories. Its full baseline diagnostic plots remain available; the extended report retains the primary baseline curves and contrasts to focus on the new experiment.

Build the extended report from this directory:

```sh
tectonic --keep-logs DMAS-6-adaptive.tex
```

For an ordinary toolchain, run XeLaTeX, BibTeX, then XeLaTeX twice. The supplied `llncs.cls` and `splncs04.bst` preserve the Springer template.

In Overleaf, upload the source package with its folders intact, set `DMAS-6-adaptive.tex` as the main document, and select XeLaTeX. The adaptive bibliography and filenames are separate from the backup.

The original 10,000-world volume results are preserved and freshly reanalysed. The separate activity-only capacity experiment uses 200 worlds and 3,600 runs. The adaptive experiment uses another 200 worlds and 7,200 runs; these samples are not pooled. All adaptive policy contrasts use matched within-world controls and the report explicitly states the scope and uncertainty of the findings.
