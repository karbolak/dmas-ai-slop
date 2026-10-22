Editable final revision of the supplied DMAS-5 beta report, imported from DMAS.zip.

Main file: `DMAS-5-final.tex`; bibliography: `bibliography.bib`.
The bundled `llncs.cls` and `splncs04.bst` preserve the supplied Springer template.

Build with Tectonic:

```sh
tectonic --keep-logs DMAS-5-final.tex
```

Or use the ordinary XeLaTeX/BibTeX sequence (XeLaTeX, BibTeX, XeLaTeX twice). Upload this directory's source, class, bibliography, style, and image files together to Overleaf. Set `DMAS-5-final.tex` as the main document.

The 10,000-world main results are retained and freshly reanalysed. The feed-capacity sensitivity experiment is separate, using 200 newly generated paired worlds. The document explicitly distinguishes the two sample sizes, exploratory analysis, and the limits of the synthetic-producer label.
