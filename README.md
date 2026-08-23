# Bit Operation Costing for Classic McEliece Key-Recovery Attacks — Artifact

This repository accompanies the paper:

> Markku-Juhani O. Saarinen.  
> **Bit Operation Cost of ``Holdout'' Key-Recovery
Attacks Against Classic McEliece**.  
> IACR Cryptology ePrint Archive, Report 2026/XXXX, 2026.  
> <https://eprint.iacr.org/2026/XXXX>

Local preprint copy: [mccost.pdf](mccost.pdf).

```bibtex
@misc{cryptoeprint:2026/XXXX,
      author = {Markku-Juhani O. Saarinen},
      title = {Bit Operation Cost of ``Holdout'' Key-Recovery
Attacks Against Classic McEliece},
      howpublished = {Cryptology {ePrint} Archive, Paper 2026/XXXX},
      year = {2026},
      url = {https://eprint.iacr.org/2026/XXXX}
}
```

## What is included

| Path | Purpose |
| --- | --- |
| [`mccost.pdf`](mccost.pdf) | Rendered paper PDF shipped with the artifact. |
| [`mccost.py`](mccost.py) | Self-contained arithmetic, parameter-search, and bit-cost model that reproduces the paper's numerical calculations. |

The Python program independently repeats the exact integer cell arithmetic,
sparse-supplier work and state tallies, declared downstream cost models, and
the bounded parameter searches used in the paper. It is an auditable costing
model, not an implementation of the proposed key-recovery attack. In
particular, it does not prove or implement reliable small-field Krylov yield,
higher-flag recovery from the priced truncated relation block, global binary
reconstruction, cross-anchor independence, phase pairing, or the target-scale
sparse-solver backend.

## Requirements

- Python 3.10 or later.
- No third-party Python packages.
- No SageMath, compiler, challenge input, network access, or external data
  files are required.

Run commands from the artifact root. The script is deterministic, reads no
secret or challenge key, and does not write files itself.

## Quick checks

Run the complete internal regression suite and print the human-readable report:

```sh
python3 mccost.py
```

Every invocation runs the internal checks before performing the requested
action. A failed arithmetic identity, parameter boundary, or pinned headline
value terminates with a nonzero status. The default report spells out the
selected-cell Hermite inequality, dimensions, rank and kernel bounds, sparse
nonzero count, Krylov application count, relation batches, work components,
and retained-state components before reproducing the paper's comparison
tables.

## Parameter searches

Repeat the exhaustive width-64 singleton supplier search for a Classic
McEliece parameter set, here allowing Holdout degree at most 24:

```sh
python3 mccost.py \
  --scan mceliece348864 \
  --maximum-degree 24
```

The accepted target names are:

- `mceliece348864`
- `mceliece460896`
- `mceliece6688128`
- `mceliece6960119`
- `mceliece8192128`

Use `python3 mccost.py --help` for the complete command-line interface.

## Cost conventions and scope

- Work is reported primarily as base-two logarithms of classical Boolean
  gates.
- Memory is reported throughout as the base-two logarithm of the retained
  number of bits; for example, `53` means `2^53` bits.
- One 64-bit XOR word operation is charged as 64 Boolean gates.
- A multiplication in `GF(2^m)` uses the paper's optimistic `m^2` AND-gate
  proxy; multiplier XORs are omitted.
- Retained vector and safe-rank-bound block-Wiedemann sequence state is
  counted. Relation-block storage is reported separately; streaming, memory
  access, communication, replication, storage penalties, and recursive solver
  scratch are not charged.
- Supplier work uses one nominal balanced Coppersmith attempt for each required
  relation batch on the augmented matrix. The model exposes but does not bound
  the small-field failure probability or expected retry count.
- Values below a NIST category comparator are conditional cost-model outputs,
  not claims that a Classic McEliece key has been recovered.

## Artifact status

- The artifact is intentionally small: the paper and one standard-library
  Python costing program.
- All reported searches are deterministic and require only public parameter
  tuples.
- No binary key material, challenge input, experimental scratch data, or large
  generated files are shipped.
- The paper is a living costing document; the `XXXX` ePrint placeholder should
  be replaced when the report number is assigned.

## License

The artifact software is released under the [MIT License](LICENSE). The paper
PDF is distributed as the accompanying preprint.
