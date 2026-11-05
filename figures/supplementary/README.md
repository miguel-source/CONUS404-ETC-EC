# Supplemental figure records

We include the captions, plotted CSV data, metadata, offline-reproduction checks, and small previews for Figures S1–S6 here. The publication-resolution PNG/TIFF/PDF/SVG exports are distributed in the separate supplemental-figure archive, so our source repository stays compact. The retained manifest records that complete export set; some high-resolution filenames in it therefore refer to the separate archive rather than files included in this folder.

We can reproduce all six full-resolution figures without downloading the native precipitation archives:

```bash
python scripts/create_supplementary_figures.py --input-bundle data/figure_inputs --output results/supplementary
```

We use Times New Roman for the paper. On a computer without that font, we can explicitly select an installed serif font, for example `--font "Liberation Serif"`; the numerical fields, masks, and color scales are unchanged. Our compact offline bundle contains only the saved arrays needed for these figures, not the full observational inputs required to rerun collocation inference.
