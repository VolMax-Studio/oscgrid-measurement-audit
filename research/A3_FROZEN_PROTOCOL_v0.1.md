# OscGrid A3 same-recording/same-class seen-vs-unseen check — preregistration v0.1

Frozen BEFORE A3 training, 2026-11-11. This protocol is distinct from R20.

Question: Within exactly the published author's block split, does per-recording test recall differ between pairs (original recording ID, class) that ARE present in the train partition and pairs that ARE NOT? Does the difference persist across network seeds 443, 444, 445?

All data and authors' model unchanged. Official Figshare v6 labeled.csv SHA256 fa3c891163c27880296c240b77c1d23d2ce41be36858ca233f774be9e86f1854. Upstream source commit 4cbdf81f9a05f646b5e4ee508e273a7b8e92b056. Model checkpoint hash for each run is to be recorded.

- Reproduce original block split (author split seed 443, train 722721, val 120316, test 361183 valid rows). Source recording ID first 32 hexadecimal characters of file_name; target from ML columns; contiguous block length >=32, window 32, stride 10. No changes to partitions or data.
- Same author MLP architecture, train-only StandardScaler, cross-entropy weights from train rows, AdamW lr .001 wd .0001, batch 128, dropout .3, gradient clipping 10, best validation accuracy checkpoint, early stop patience 10, 100 epochs cap. drop_last=False (author setting, no singleton on the author split).
- Model seeds 443, 444, 445; no tuning, no retraining if poor outcome; training process may fail and must be marked INCOMPLETE. Only test-time grouping differs from the previous paired-MLP experiment.
- Primary evaluation: for each class k in 0..3, for each source recording R with >0 test windows truly labelled k, define same_class_seen = (R,k) exists in training; compute recall = TP_k(R)/count_true_k(R); summarize unweighted mean per recording for seen vs unseen. Each recording may appear for multiple classes but contributes at most once in each k.
- Report raw per-record CSV: full IDs, number of true labelled class-k test windows, count correct, class-k recall; group membership; checkpoint digests; confusion matrices and ordinary pooled per-class F1 (secondary).
- Predefined small-recording sensitivity minimum labelled windows n >= 1, 6, 32 (inclusive); n in windows, NOT individual samples. Report subset sizes including if zero, do not claim stable estimate from singleton groups.
- Do not claim observed conditional difference estimates causal effect of split leakage. Seen and unseen groups are different recordings, not randomized to seen status, and a single seen class-3 recording cannot support statistical generalization.
- Class-3 paper result reproduction remains INCONCLUSIVE, published mean F1 ~0.79 vs prior local author-like F1 ~0.70.
- No edits or pushes to production repos. No issue publication; draft only after examining exact results.
