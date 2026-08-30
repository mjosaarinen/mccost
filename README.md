# Bit Operation Cost of "Holdout" Key-Recovery Attacks Against Classic McEliece — Artifact

This repository accompanies:

> Markku-Juhani O. Saarinen.<br>
> **Bit Operation Cost of "Holdout" Key-Recovery Attacks Against Classic McEliece.**<br>
> IACR Cryptology ePrint Archive, Report 2026/1786, 2026.<br>
> <https://eprint.iacr.org/2026/1786>

This is a living costing paper. Its dated revisions may update the estimate and scope, but the published title remains fixed. The files here correspond to the living revision dated 30 August 2026.

## Contents

| Path | Purpose |
| --- | --- |
| [`mccost.pdf`](mccost.pdf) | Rendered paper. |
| [`mccost.py`](mccost.py) | Self-contained verifier for the paper's current tallies. |

The Python artifact contains only the best attack route currently reported in the paper: one-chart direct locator recovery, evaluated for every standardized Classic McEliece parameter set. It does not retain earlier one-coordinate parameter searches or speculative experiments.

## Reproduction

Python 3.10 or later is sufficient; there are no third-party dependencies. From the artifact root, run:

```sh
python3 mccost.py
```

The command recomputes and checks the human-readable estimates:

```text
parameter set       category/reference  work      live state     solver field  solver failure margin
mceliece348864      1 / 2^143           2^126.77  2^51.33 bits   GF(2^240)     <2^-135.17      16.23
mceliece460896      3 / 2^207           2^145.22  2^59.10 bits   GF(2^260)     <2^-138.19      61.78
mceliece6688128     5 / 2^272           2^137.48  2^55.18 bits   GF(2^247)     <2^-132.51     134.52
mceliece6960119     5 / 2^272           2^136.65  2^54.86 bits   GF(2^247)     <2^-133.25     135.35
mceliece8192128     5 / 2^272           2^137.48  2^55.18 bits   GF(2^247)     <2^-132.51     134.52
internal_checks: PASS
```

The program checks exact dimensions and integer arithmetic before printing the estimates. It is deterministic, reads no challenge or secret key, and writes no files.

## Scope

The tally uses one shortened coordinate set of the direct locator-recovery route of Ghoshal, Ishai, Jain, and Sun for each parameter set. It prices reliable iterative scalar Lanczos solves over finite fields `GF(2^240)`, `GF(2^260)`, or `GF(2^247)`, exact linear maps back to the original fields followed by substitution into the original equations, public selection of `mt+1` compatible labels, and finite key-completion and verification estimates. Matrix entries that use only binary polynomial-derivative data are costed separately from arbitrary field entries.

The results remain conditional on four explicit source premises: the canonical rank-two relation branch exists; the preliminary three-holdout test returns only the common Frobenius-conjugate branches; the four-holdout test returns one globally compatible projective label per coordinate; and the required rank inequalities hold for the full relation matrices and their coordinate blocks. The third premise is the possible exponential obstruction: without singleton label sets, the direct fallback may have to reconcile a product of coordinate-wise choices. No probability is assigned to these structural premises. Under them, all five work estimates lie below the [NIST classical-gate reference level for their claimed category](https://csrc.nist.gov/projects/post-quantum-cryptography/post-quantum-cryptography-standardization/evaluation-criteria/security-(evaluation-criteria)). The bit-operation and simultaneously stored state tallies are not hardware-time estimates. No Classic McEliece target key has been recovered.

## Citation

```bibtex
@misc{cryptoeprint:2026/1786,
      author = {Markku-Juhani O. Saarinen},
      title = {Bit Operation Cost of ``Holdout'' Key-Recovery
               Attacks Against Classic McEliece},
      howpublished = {Cryptology {ePrint} Archive, Paper 2026/1786},
      year = {2026},
      url = {https://eprint.iacr.org/2026/1786}
}
```

## License

The artifact software is released under the [MIT License](LICENSE). The paper PDF is distributed as the accompanying preprint.
