# OscGrid measurement & evaluation audit

Independent, **read-only reproduction** of structural findings from public [OscGrid v6](https://doi.org/10.6084/m9.figshare.28465427.v6) and the authors' [pinned validation code](https://github.com/AIRI-Institute/oscgrid/blob/4cbdf81f9a05f646b5e4ee508e273a7b8e92b056/validation/train.py).

Not affiliated with the dataset authors. Not a production fault classifier, field certification, or evidence of inflated published scores. Original dataset bytes and model checkpoints are **not** distributed here.

## What the structural audit computes

| Check | Pinned dataset result |
|---|---:|
| Original labeled rows | 1,554,809 |
| Rows with all 5 baseline inputs | 1,206,916 |
| Eligible source recordings | 451 |
| Train / validation / test rows | 722,721 / 120,316 / 361,183 |
| Original recordings present in both train and test | **229** |
| Test rows from recordings represented in train | **232,210 / 361,183 (64.29%)** |
| Class-3 test windows from one source recording | **4,458 / 5,716** |
| Share of class-3 rows from five largest recordings | **85.56%** |
| Near-flat 32-sample chunks on five model inputs in a316… | **1,494 / 1,499 (99.67%)** |

Read [FINDINGS.md](FINDINGS.md) for method, channel statistics and limitations. The structural audit does **not** train a model.

## Run the structural audit

Download **version 6** of the dataset from the Figshare DOI above. Extract its labeled-data archive and locate the original labeled.csv file. No source recordings are hosted in this repository.

Use Python 3.10+ and install the pinned audit dependencies:

~~~bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-audit.txt
python audit.py /path/to/labeled.csv --output local_audit.json
python scripts/check_output.py local_audit.json
~~~

The input must match SHA-256 **fa3c891163c27880296c240b77c1d23d2ce41be36858ca233f774be9e86f1854**. The independent script reproduces the author's contiguous (file_name, class)-block split with seed 443, frame 32, stride 10, split ratios 60:10:30. Expected structured output: [reference_results/structural_audit_expected.json](reference_results/structural_audit_expected.json). The checker compares **all JSON fields**, not merely the headline counts.

A 32-hex filename prefix is treated as a source-recording ID. It is **not evidence** that each ID represents an independent physical event or station.

## Optional: reproduce three MLP runs

The public issue also describes an exploratory three-seed MLP experiment (A3). Its **exact originally executed scripts and protocol** are in the research folder, with three source-level JSON results and a manifest under reference_results. The actual author model is not vendored, and rerunning the experiment requires PyTorch and additional computation.

~~~bash
python -m pip install -r requirements-mlp.txt
git clone https://github.com/AIRI-Institute/oscgrid.git upstream-oscgrid
git -C upstream-oscgrid checkout --detach 4cbdf81f9a05f646b5e4ee508e273a7b8e92b056
cp upstream-oscgrid/models/models.py research/author_models.py
cp upstream-oscgrid/validation/train.py research/author_train.py
sha256sum research/author_models.py research/author_train.py
~~~

Expected upstream SHA-256:
- author_models.py: **3fe4e391f8a32947a18967a0a79d9b606f77b0c4869ffed4ce025b85cd72665c**
- author_train.py: **6e623b800cd8205c0ea7bbf9f0c1a864b6c0f68e583660e572425201d2b417c2**

The original A3 script expects the source CSV at unpacked/labeled/labeled.csv relative to the repository root:

~~~bash
mkdir -p unpacked/labeled
ln -s "$(realpath /path/to/labeled.csv)" unpacked/labeled/labeled.csv
python research/a3_seen_vs_unseen.py
~~~

It uses the published author MLP class but **reimplements the training loop**, rather than invoking the author's complete train.py. It trains separately for seeds 443/444/445 and saves local checkpoints and full per-source results in the ignored research/a3_seen_vs_unseen_results directory. CPU/GPU and software differences may change checkpoint bytes, epochs, and last decimal places.

For class 3: one training-seen source recording has test recall **1.00** in all three runs; the other 14 source recordings have **equal-record mean** recall **0.245 / 0.245 / 0.243**. The groups are highly unequal and not randomized. The 1/6/32 source-window sensitivity cutoffs were fixed before these A3 training runs **after earlier exploratory analysis**, so this was not a blind preregistration.

The prior A3 source/protocol hashes and outputs are provided for reproducibility, **not** as a claim of exact replication of published five-run Table 2 F1.

## Interpretation and licensing

**Verified for the pinned input:** data partition overlap, class concentration, input-channel coverage. **Not established:** inflated published F1 by any amount, independent physical-event generalization, wrong expert annotations, or field reliability. The authors present their experiment as technical validation of annotations.

Sources: [Dataset](https://doi.org/10.6084/m9.figshare.28465427.v6) (CC BY 4.0), [upstream MIT code](https://github.com/AIRI-Institute/oscgrid), [Scientific Data paper](https://doi.org/10.1038/s41597-026-06587-8). Correspondence: [OscGrid issue #15](https://github.com/AIRI-Institute/oscgrid/issues/15).

The original independent audit scripts and documentation here are under [MIT](LICENSE), copyright 2026 Ivan Nestorov / VolMax Studio Lab. Reference JSON files contain **derived statistics**, not original waveforms.
