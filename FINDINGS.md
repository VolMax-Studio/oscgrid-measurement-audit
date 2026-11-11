# Findings and limitations

Pinned original labeled.csv SHA-256: **fa3c891163c27880296c240b77c1d23d2ce41be36858ca233f774be9e86f1854**.

## Source overlap

The published validation split assigns contiguous (file_name, label)-constant blocks rather than holding out entire original COMTRADE recordings. In the author's seed-443 split, **229 of 451 source recording IDs** represented in eligible blocks appear in **both training and test**; **232,210 of 361,183** test rows come from training-seen source IDs. Source ID means the first 32 lowercase hex characters of the original file_name.

This establishes nonindependence at the source-recording ID level. It **does not measure** numerical F1 inflation or establish the number of independent physical fault episodes.

## Class-3 sample concentration

After requiring availability of the author's five selected channels, the five largest source recordings contribute **164,706 of 192,506** class-3 rows (**85.5589%**; in eligible blocks, 85.7795%). Recording **8f49afcdd88bfa4e880c3ac108b24f07** contributes **4,458 of 5,716** class-3 test windows and has class-3 blocks in training. The other 14 test recordings with class 3 contain 1,258 windows, including 1,037 from **a066a9c201a217aed8ab79a9c7a8e512**.

The independent three-seed MLP reimplementation (original scripts under research) gives class-3 recall **1.0** for the sole seen-same-class source and **0.245 / 0.245 / 0.243** equal-record mean for the 14 unseen-same-class sources. The two groups differ in recording composition; this is **descriptive, not causal**.

## Channel-coverage observation

For source recording **a3164c73c305b027d2ba382e7840bbd2**, named Bus 1 _event N1, **47,999** rows carry the class-3 label. Across **1,494 of 1,499** nonoverlapping 32-row chunks, each of the baseline's five selected channels has a within-chunk standard deviation below **0.02 in CSV units**.

| Model input | Full labeled-row std | Additional channel | Full labeled-row std |
|:--|--:|:--|--:|
| IA | 0.006311 | UA CL | 0.381395 |
| IC | 0.006277 | UB CL | 0.525679 |
| UA BB | 0.014356 | UC CL | **57.757404** |
| UB BB | 0.014905 | — | — |
| UC BB | 0.016738 | — | — |

This is evidence of **selected-input channel coverage**, not proof of an erroneous fault label, device misconfiguration, protection-system failure, or physical topology. BB/CL are dataset channel names, not an independently audited single-line diagram.

## Unresolved Table 2 reproduction

The article's statement about nonoverlapping one-cycle windows appears in the **autoencoder** section; it does not necessarily describe subsequent supervised results. The public validation/train.py defaults to frame size 32 and stride 10. The exact Table 2 training/window settings require clarification.

Locally the exploratory three-seed MLP produced class-3 F1 **0.6969 / 0.7138 / 0.6882**, compared with approximately **0.79** in the paper's five-run table. These are **not** an exact reproduction of Table 2, nor an estimate of inflation.

In any operational case, correct measurement points, channel definitions, calibration, topology and timestamp context must be verified before connecting a calculated result to a physical decision.
