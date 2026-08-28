# Bit Operation Cost of "Holdout" Key-Recovery Attacks Against Classic McEliece -- Artifact

This repository accompanies the paper:

> Markku-Juhani O. Saarinen.  
> **Bit Operation Cost of "Holdout" Key-Recovery
Attacks Against Classic McEliece**.  
> IACR Cryptology ePrint Archive, Report 2026/1786, 2026.  
> <https://eprint.iacr.org/2026/1786>

Local preprint copy: [mccost.pdf](mccost.pdf).

This is a living costing paper: estimates and assumption boundaries may evolve
through dated revisions, while the published paper title remains fixed.

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

## What is included

| Path | Purpose |
| --- | --- |
| [`mccost.pdf`](mccost.pdf) | Rendered paper PDF shipped with the artifact. |
| [`mccost.py`](mccost.py) | Self-contained arithmetic, parameter-search, and bit-cost model that reproduces the paper's numerical calculations and the distinct published conditional rank-two baseline discussed below. |

The Python program independently repeats the exact integer cell arithmetic,
point-block rank and kernel-floor calculations, sparse relation-generation
work, active retained-state accounting, declared downstream cost models, and
the bounded parameter searches used in the paper. It also reports joint cell
optima, relation-block-size sensitivities, Apon's correct-leaf bounds, and a
separately priced continuation envelope. It also independently reconstructs
the five published GhIsJa+26 rank-two baseline rows from their homogeneous-jet
dimensions, predicted rank margins, sparse locator systems, shortening covers,
and block-Lanczos cost formula. The default report keeps the two routes and
their assumptions separate.

This is an auditable costing model, not an implementation of either
key-recovery route. For the singleton hold-out route, it does not prove reliable
binary Krylov yield, sufficient relation-block richness or higher-order
derivative recovery at target scale, generic wrong-leaf rank, success of the
public-column list problem, compatible affine normalization, common-kernel
richness, binary reconstruction, the public pure cross-pairing needed for phase
synchronization, or a target-scale sparse-solver backend. Reproducing the
rank-two arithmetic does not establish its canonical-form and rank-one-rigidity
conjectures, predicted global and projected ranks, reliable sparse-solver model,
or a functional implementation. The unitemized optimized estimate announced
in GhIsJa+26 Remark 7.3 is not included.

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
selected-cell Hermite inequality, dimensions, point-block rank and kernel
bounds, sparse nonzero count, padded or fallback augmented solver shape,
Krylov application count, relation batches, work components, and active-state
components. It then reproduces the paper's selected-cell and jointly optimized
tables, relation-block-size sensitivity, Apon bounds, continuation envelope,
and other declared model sensitivities. A distinct conditional-route tally
compares those singleton subtotals with the independently reconstructed
GhIsJa+26 rank-two baseline and explicitly refuses an unqualified best-attack
claim.

## Parameter searches

Repeat the exhaustive width-64 singleton relation-generation search for a
Classic McEliece parameter set, here allowing hold-out degree at most 24:

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

The default `--objective relation` minimizes relation generation alone. Use
`--objective affine-dense` or `--objective pgl-nested` to jointly optimize the
corresponding accounted subtotal. Adding `--wide` selects the smallest
multiple-of-64 block width that requests the required relations in one batch.

Use `python3 mccost.py --help` for the complete command-line interface.

## Cost conventions and scope

- Work is reported primarily as base-two logarithms of elementary bit
  operations.
- Memory is reported throughout as the base-two logarithm of the retained
  number of bits; for example, `53` means `2^53` bits.
- A sparse block application charges `ceil(b/64)` packed 64-bit words per
  incidence, with each word operation counted as 64 bit operations.
- A multiplication in `GF(2^m)` uses the paper's optimistic `m^2`
  binary-operation proxy; multiplier XORs are omitted.
- The report separates vector-panel, block-Wiedemann sequence, combined solver,
  and retained-relation state. Its headline active peak is the larger of the
  final relation block and the solver state plus previously certified relation
  batches. Memory access, addressing, communication, replication, storage
  penalties, generator-basis temporaries, and recursive solver scratch are not
  charged.
- Relation generation uses one nominal equal-block Coppersmith attempt for
  each required relation batch. When `R_M <= N`, the solver uses the injectively
  padded square operator `B = E M`, which has the same right kernel and rank
  bound as `M` and requires one sparse traversal. The augmented operator is
  only a fallback when padding is unavailable. The model exposes but does not
  bound the small-field failure probability or expected retry count.
- Every primary singleton-route subtotal stops at candidate interpolation. The
  program reports one fail-closed list-problem continuation attempt separately;
  its target-scale success and generic rejection of wrong leaves remain
  heuristic.
- The distinct rank-two rows reproduce a published conditional recovery model,
  not a measured or independently implemented attack. Their conjectures and
  predicted-rank conditions are not shared with the singleton subtotals.
- Values below a NIST category comparator are conditional cost-model outputs,
  not claims that a Classic McEliece key has been recovered.

## Artifact status

- The artifact is intentionally small: the paper and one standard-library
  Python costing program.
- All reported searches are deterministic and require only public parameter
  tuples.
- No binary key material, challenge input, experimental scratch data, or large
  generated files are shipped.

## License

The artifact software is released under the [MIT License](LICENSE). The paper
PDF is distributed as the accompanying preprint.
