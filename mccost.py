#!/usr/bin/env python3
"""Self-contained arithmetic and bit-cost model for Classic McEliece.

This file is deliberately self-contained: it imports only Python's standard
library, opens no challenge input, and writes no files.  It repeats the exact
integer cell arithmetic, the sparse relation-generation bit-operation tally,
and the downstream cost models used in the manuscript.  The selector uses a
four-anchor lower bound with the strict-profile condition count; indexed
filtration counts are exposed but never used to reduce that bound.
The report also reoptimizes Category 1 under hypothetical larger relation
blocks because the first-order certificate does not determine how many
relations the systems used for higher-order derivative recovery need.
``Price.total_log2`` includes replay and all-anchor certification and denotes an
accounted subtotal through candidate interpolation. A correct binary-Goppa
leaf then faces the full-code list problem identified by Apon; the script
models it as public-column continuation and reports both Apon's correct-leaf
dimension and further-correct-label lower bounds, plus a deliberately coarse
one-leaf work envelope. Continuation success and generic rejection of wrong
leaves remain separate, clearly labelled heuristics. This is not an implementation of the
attack.  In particular, it does not prove reliable binary Krylov yield,
Vedenev's conjecture underlying higher-order derivative recovery for the
priced relation block, binary reconstruction, cross-anchor independence,
common-kernel richness, or the public pure cross-pairing needed for
Frobenius-phase synchronization.

Typical use::

    python3 mccost.py                         # report and internal checks
    python3 mccost.py --scan mceliece348864   # exhaustive one-position hold-out scan

Cost units follow the manuscript.  A block application uses
``ceil(block_width/64)`` packed 64-bit words per sparse incidence, and a
GF(2^m) multiplication is represented by the optimistic m^2 binary-operation
proxy; XORs in the multiplier are omitted.  The model does not charge random
access, addressing, communication, or storage.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from math import ceil, comb, log2


NIST_BIT_OPERATION_THRESHOLDS = {
    1: 143.0, 2: 146.0, 3: 207.0, 4: 210.0, 5: 272.0,
}
PIB = 1 << 50
EIB = 1 << 60
BRANCH_EXTRACTION_TRIALS = 64


@dataclass(frozen=True)
class Target:
    name: str
    n: int
    m: int
    t: int
    category: int
    selected_k: int
    selected_d: int


TARGETS = {
    target.name: target
    for target in (
        Target("mceliece348864", 3488, 12, 64, 1, 208, 8),
        Target("mceliece460896", 4608, 13, 96, 3, 269, 8),
        Target("mceliece6688128", 6688, 13, 128, 5, 305, 8),
        Target("mceliece6960119", 6960, 13, 119, 5, 295, 8),
        Target("mceliece8192128", 8192, 13, 128, 5, 305, 8),
    )
}

@dataclass(frozen=True)
class Schedule:
    high_order: int
    high_columns: int
    low_order: int
    low_columns: int

    @property
    def order_total(self) -> int:
        return self.high_order * self.high_columns + self.low_order * self.low_columns

    @property
    def text(self) -> str:
        return f"{self.high_columns}@{self.high_order}+{self.low_columns}@{self.low_order}"


@dataclass(frozen=True)
class Cell:
    target: str
    n: int
    m: int
    t: int
    k: int
    d: int
    h: int
    n_l: int
    D_l: int
    constrained_columns: int
    order_target: int
    hermite_margin: int
    schedule: Schedule
    high_levels: tuple[int, ...]
    low_levels: tuple[int, ...]
    modeled_column_weight: int
    weight_threshold: int
    N: int
    operator_rows: int
    row_rank_bound: int
    kernel_floor: int
    nonzeros: int
    anchors: int
    relation_minimum: int

    @property
    def padding_available(self) -> bool:
        """Whether the row space embeds injectively into an ``N``-square box."""
        return self.operator_rows <= self.N


@dataclass(frozen=True)
class Price:
    cell: Cell
    block_width: int
    operator_kind: str
    organization: str
    normalization: str
    solve: str
    phase_alignment: bool
    relation_generation_log2: float
    guess_log2: float
    per_guess_field_ops_log2: float
    per_guess_bitops_log2: float
    downstream_log2: float
    replay_log2: float
    certification_log2: float
    derivative_recovery_log2: float
    derivative_recovery_retries: int
    alignment_relation_generation_log2: float | None
    alignment_replay_log2: float | None
    alignment_cost_charged: bool
    pairing_construction_assumed: bool
    total_log2: float
    vector_state_bytes: int
    sequence_state_bytes: int
    solver_state_bytes: int
    relation_state_bytes: int
    active_peak_state_bytes: int
    state_bytes: int
    conditional_subtotal: bool

    @property
    def accounted_log2(self) -> float:
        """Alias emphasizing that ``total_log2`` stops at candidate interpolation."""
        return self.total_log2


@dataclass(frozen=True)
class RelationBlockSensitivity:
    """One reoptimized relation-block sensitivity for higher-order derivative recovery."""

    model: str
    cell: Cell
    relation_count: int
    relation_generation_log2: float
    total_log2: float
    state_bytes: int

def _require_positive(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _binom0(n: int, r: int) -> int:
    """Binomial coefficient with the manuscript's out-of-range-is-zero rule."""
    return comb(n, r) if n >= 0 and 0 <= r <= n else 0


@lru_cache(maxsize=None)
def level_modules(d: int, j: int) -> frozenset[int]:
    """Harmonic modules seen at derivative level ``j`` (Lucas criterion)."""
    _require_positive("d", d)
    if isinstance(j, bool) or not isinstance(j, int) or not 0 <= j <= d:
        raise ValueError("j must be an integer in [0,d]")
    return frozenset(i for i in range(j + 1) if ((j - i) & ~(d - i)) == 0)


@lru_cache(maxsize=None)
def lucas_levels(d: int, s: int) -> tuple[int, ...]:
    """Lexicographically first minimum set covering modules 0,...,s-1."""
    _require_positive("d", d)
    _require_positive("s", s)
    if s > d:
        raise ValueError("s must not exceed d")
    wanted = frozenset(range(s))
    for size in range(1, s + 1):
        for levels in combinations(range(s), size):
            seen = frozenset().union(*(level_modules(d, j) for j in levels))
            if wanted <= seen:
                return levels
    raise AssertionError("the full level stack always covers the target")


@lru_cache(maxsize=None)
def point_block_rank(k: int, w: int, d: int, s: int) -> int:
    """Exact binary rank of one order-``s`` point block."""
    total = 0
    for b in range(s):
        inner, rest = d - b, s - 1 - b
        if 0 <= inner <= w:
            total += _binom0(k - w, b) * _binom0(w, min(rest, inner, w - inner))
    return total


@lru_cache(maxsize=None)
def point_block_nnz(k: int, w: int, d: int, levels: tuple[int, ...]) -> int:
    """Exact nonzeros in the retained Hasse levels at a weight-``w`` point."""
    return sum(
        _binom0(w, a) * _binom0(k - w, j - a) * _binom0(w - a, d - j)
        for j in levels
        for a in range(j + 1)
    )


def balanced_schedule(
    columns: int, order_target: int, d: int, hermite_margin: int = 1,
) -> Schedule | None:
    """Cheapest adjacent-order schedule with the requested Hermite margin."""
    _require_positive("hermite_margin", hermite_margin)
    total = order_target + hermite_margin
    high = ceil(total / columns)
    if not 2 <= high <= d - 1:
        return None
    high_columns = total - columns * (high - 1)
    return Schedule(high, high_columns, high - 1, columns - high_columns)


def strict_profile_condition_count(k: int) -> int:
    """Unindexed strict-complete-filtration conditions at one anchor."""
    if k < 2:
        raise ValueError("strict-profile count requires k >= 2")
    return comb(k, 2)


def indexed_filtration_condition_count(
    k: int, dimensions: tuple[int, ...],
) -> int:
    """Exact indexed count ``sum_j (k-dim V^[j])`` for a supplied profile.

    The costing model never uses an observed value larger than the strict-profile
    count to reduce the anchor count: that would require a structured wrong-guess
    rank theorem.
    """
    if k < 2 or len(dimensions) < k - 1:
        raise ValueError("indexed profile requires at least k-1 dimensions")
    if any(
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or not 1 <= dimension <= min(index + 1, k - 1)
        for index, dimension in enumerate(dimensions)
    ):
        raise ValueError(
            "indexed dimensions must satisfy 1 <= dim V^[j] <= min(j+1,k-1)"
        )
    if any(left > right for left, right in zip(dimensions, dimensions[1:])):
        raise ValueError("indexed dimensions must be nondecreasing")
    if any(right > left + 1 for left, right in zip(dimensions, dimensions[1:])):
        raise ValueError("indexed dimension can increase by at most one per order")
    return sum(k - dimension for dimension in dimensions)


def anchor_count(k: int, D_l: int) -> int:
    """Wrong-guess anchor count under the strict-profile substitution.

    Vedenev's indexed filtration can impose a measured count ``E_i >= B``.
    The artifact deliberately substitutes only the certified strict-profile
    value ``B=binom(k,2)`` because exploiting ``E_i>B`` would require a new
    structured wrong-guess rank statement.
    """
    if k < 3 or D_l < 0:
        raise ValueError("anchor count requires k >= 3 and D_l >= 0")
    unknowns = k * (D_l + 1)
    conditions_per_anchor = strict_profile_condition_count(k)
    return max(4, ceil(Fraction(unknowns, conditions_per_anchor)))


def holdout_cell(
    target: Target, k: int, d: int, h: int = 1, hermite_margin: int = 1,
) -> Cell | None:
    """Return one dimension-guaranteed modeled-weight cell, or ``None``.

    The row-rank bound is a sum of point-block ranks and may exceed the true
    global rank.  Hence ``N-row_rank_bound`` is a rigorous kernel lower bound.
    The nonzero count is exact only for the modeled non-pivot weight ``k//2``.
    """
    mt = target.m * target.t
    if not d + 2 <= k <= target.n - mt or d < 3 or h < 1:
        return None
    n_l = k + mt
    D_l = n_l - 2 * target.t - 1
    columns = mt - h
    order_target = k + d * (mt - 2 * target.t - 1)
    schedule = balanced_schedule(columns, order_target, d, hermite_margin)
    if D_l < 1 or order_target < 0 or schedule is None:
        return None
    w = k // 2
    threshold = d + schedule.high_order - 1
    if w < threshold:
        return None
    N = comb(k, d)
    high_rank = point_block_rank(k, w, d, schedule.high_order)
    low_rank = point_block_rank(k, w, d, schedule.low_order)
    row_rank_bound = schedule.high_columns * high_rank + schedule.low_columns * low_rank
    if row_rank_bound >= N:
        return None
    high_levels = lucas_levels(d, schedule.high_order)
    low_levels = lucas_levels(d, schedule.low_order) if schedule.low_columns else ()
    nonzeros = schedule.high_columns * point_block_nnz(k, w, d, high_levels)
    operator_rows = schedule.high_columns * sum(comb(k, j) for j in high_levels)
    if schedule.low_columns:
        nonzeros += schedule.low_columns * point_block_nnz(k, w, d, low_levels)
        operator_rows += schedule.low_columns * sum(comb(k, j) for j in low_levels)
    anchors = anchor_count(k, D_l)
    relation_minimum = k - (target.m + 1) + comb(target.m, 2)
    return Cell(
        target.name, target.n, target.m, target.t, k, d, h, n_l, D_l,
        columns, order_target, hermite_margin, schedule, high_levels, low_levels, w, threshold,
        N, operator_rows, row_rank_bound, N - row_rank_bound, nonzeros, anchors,
        relation_minimum,
    )


def _logsum2(*exponents: float) -> float:
    peak = max(exponents)
    return peak + log2(sum(2.0 ** (value - peak) for value in exponents))


def resolve_operator_kind(cell: Cell, operator_kind: str = "auto") -> str:
    """Select the exact square operator used by the Krylov model.

    If ``R_M <= N``, ``padded`` means ``B=EM`` for any injective
    ``E:F_2^R_M -> F_2^N``.  Then ``ker(B)=ker(M)`` and one application uses
    one traversal of ``M``.  The augmented fallback is required when the
    emitted row count exceeds ``N``.
    """
    if operator_kind not in {"auto", "padded", "augmented"}:
        raise ValueError("operator_kind must be auto, padded, or augmented")
    if operator_kind == "auto":
        return "padded" if cell.padding_available else "augmented"
    if operator_kind == "padded" and not cell.padding_available:
        raise ValueError("padded operator requires R_M <= N")
    return operator_kind


def packed_lane_bits(block_width: int) -> int:
    """Bit-operation charge for one packed block XOR per incidence."""
    _require_positive("block_width", block_width)
    return 64 * ceil(block_width / 64)


def operator_dimension(cell: Cell, operator_kind: str = "auto") -> int:
    """Dimension of the exact square matrix passed to Coppersmith."""
    kind = resolve_operator_kind(cell, operator_kind)
    return cell.N if kind == "padded" else cell.N + cell.operator_rows


def operator_rank_bound(cell: Cell, operator_kind: str = "auto") -> int:
    """Safe rank bound for the exact square solver operator."""
    kind = resolve_operator_kind(cell, operator_kind)
    return cell.row_rank_bound if kind == "padded" else 2 * cell.row_rank_bound


def operator_traversals(operator_kind: str) -> int:
    """Sparse ``M``/``M^T`` traversals in one square-operator application."""
    if operator_kind == "padded":
        return 1
    if operator_kind == "augmented":
        return 2
    raise ValueError("operator_kind must be resolved before traversal pricing")


def coppersmith_applications(
    cell: Cell, block_width: int, operator_kind: str = "auto",
) -> int:
    """Nominal applications in one balanced equal-block attempt."""
    _require_positive("block_width", block_width)
    return 3 * ceil(operator_rank_bound(cell, operator_kind) / block_width)


def sequence_length(
    cell: Cell, block_width: int, operator_kind: str = "auto",
) -> int:
    """Safe rank-bound precision for the retained ``b``-by-``b`` sequence."""
    return 2 * ceil(operator_rank_bound(cell, operator_kind) / block_width) + block_width


def vector_state_bytes(
    cell: Cell, block_width: int, operator_kind: str = "auto",
) -> int:
    """Packed vector-panel storage for one solver block."""
    return ceil(operator_dimension(cell, operator_kind) * block_width / 8)


def sequence_state_bytes(
    cell: Cell, block_width: int, operator_kind: str = "auto",
) -> int:
    """Retained scalar-sequence floor; generator-basis temporaries are extra."""
    return ceil(sequence_length(cell, block_width, operator_kind) * block_width**2 / 8)


def solver_state_bytes(
    cell: Cell, block_width: int, operator_kind: str = "auto",
) -> tuple[int, int, int]:
    """Return vector-panel, retained-sequence, and summed solver state."""
    vector = vector_state_bytes(cell, block_width, operator_kind)
    sequence = sequence_state_bytes(cell, block_width, operator_kind)
    return vector, sequence, vector + sequence


def relation_state_bytes(cell: Cell, relation_count: int | None = None) -> int:
    """Packed coefficient storage for one active certified relation block."""
    count = cell.relation_minimum if relation_count is None else _require_positive(
        "relation_count", relation_count,
    )
    return ceil(count * cell.N / 8)


def retained_relation_capacity(cell: Cell, block_width: int) -> int:
    """Maximum useful independent output retained from all required batches."""
    batches = ceil(cell.relation_minimum / block_width)
    return min(cell.kernel_floor, batches * block_width)


def active_peak_state_bytes(
    cell: Cell, block_width: int, operator_kind: str = "auto",
) -> tuple[int, int, int]:
    """Return solver, retained-relation, and sequential active-peak storage.

    Earlier full batches must remain available while the final batch is
    reduced against them.  The returned peak excludes generator-basis
    temporaries and permits the final solver panel to be reused for its output.
    """
    _, _, solver = solver_state_bytes(cell, block_width, operator_kind)
    batches = ceil(cell.relation_minimum / block_width)
    capacity = retained_relation_capacity(cell, block_width)
    relations = relation_state_bytes(cell, capacity)
    prior = relation_state_bytes(cell, (batches - 1) * block_width) if batches > 1 else 0
    return solver, relations, max(solver + prior, relations)


def derivative_recovery_field_ops_upper(cell: Cell, retries: int = 1) -> int:
    """Loose dense upper bound for higher-order derivative recovery.

    This is arithmetic accounting, not a correctness theorem.  It assumes that
    Vedenev's sequential systems are consistent and that the priced truncated
    relation block is rich enough at every order.  For auditability we charge a
    naive rescan of all coefficients and dense elimination at every branch and
    higher order; a real implementation should be substantially cheaper.
    """
    retries = _require_positive("retries", retries)
    r = cell.relation_minimum
    per_branch = r * cell.N * cell.d * cell.k**3
    per_branch += max(cell.k - 2, 0) * (r * cell.k**2 + cell.k**3)
    return retries * cell.anchors * cell.m * per_branch


def derivative_recovery_bitops_log2(cell: Cell, retries: int = 1) -> float:
    """Convert the dense bound for higher-order derivative recovery to bit operations."""
    return log2(derivative_recovery_field_ops_upper(cell, retries) * cell.m**2)


def relation_generation_price(
    cell: Cell,
    *,
    block_width: int = 64,
    organization: str = "singleton",
    solver_attempts: int = 1,
    operator_kind: str = "auto",
) -> tuple[float, int]:
    """Return nominal sequential packed-block Coppersmith work and peak state.

    One width-``b`` recovery yields at most ``b`` relations, so the relation
    block needs ``ceil(r_min/b)`` sequential recoveries.  Each sparse incidence
    is charged ``ceil(b/64)`` full 64-bit words.  ``solver_attempts`` is the
    attempts-per-batch multiplier.  No theorem bounds its required value for
    this structured matrix over F_2; displayed rows use one
    verified-or-refused attempt for every required batch.
    """
    if organization not in {"singleton", "cover", "common"}:
        raise ValueError("organization must be singleton, cover, or common")
    _require_positive("block_width", block_width)
    solver_attempts = _require_positive("solver_attempts", solver_attempts)
    kind = resolve_operator_kind(cell, operator_kind)
    kernels = {
        "singleton": cell.anchors,
        "cover": ceil(cell.anchors / cell.h),
        "common": 1,
    }[organization]
    batches = ceil(cell.relation_minimum / block_width)
    bitops_integer = (
        coppersmith_applications(cell, block_width, kind)
        * operator_traversals(kind) * packed_lane_bits(block_width) * cell.nonzeros
        * kernels * batches * solver_attempts
    )
    bitops = log2(bitops_integer)
    _, _, peak = active_peak_state_bytes(cell, block_width, kind)
    return bitops, peak


def pair_alignment_relation_generation_price(
    target: Target,
    k: int,
    d: int,
    anchors: int,
    *,
    block_width: int = 64,
    relation_count: int | None = None,
    solver_attempts: int = 1,
    hermite_margin: int = 1,
    operator_kind: str = "auto",
) -> tuple[float, int, Cell] | None:
    """Price ``anchors-1`` two-position relation-generation computations.

    This prices the proposed source of cross-anchor data; it does not construct
    a map from the two local jet spaces to a nonzero pure cross-pairing.  Thus a
    caller may charge this work without discharging the synchronization
    hypothesis.
    """
    if anchors < 2:
        raise ValueError("anchors must be at least two")
    pair_cell = holdout_cell(target, k, d, 2, hermite_margin)
    if pair_cell is None:
        return None
    kind = resolve_operator_kind(pair_cell, operator_kind)
    required = pair_cell.relation_minimum if relation_count is None else _require_positive(
        "relation_count", relation_count,
    )
    batches = ceil(required / block_width)
    bitops_integer = (
        coppersmith_applications(pair_cell, block_width, kind)
        * operator_traversals(kind) * packed_lane_bits(block_width) * pair_cell.nonzeros
        * batches * (anchors - 1) * _require_positive("solver_attempts", solver_attempts)
    )
    bitops = log2(bitops_integer)
    _, _, peak = active_peak_state_bytes(pair_cell, block_width, kind)
    return bitops, peak, pair_cell


def relation_replay_bit_upper(cell: Cell, relations: int, kernels: int) -> int:
    """Fresh-annihilation plus naive-independence bit bound."""
    relations = _require_positive("relations", relations)
    kernels = _require_positive("kernels", kernels)
    full_word_batches = ceil(relations / 64)
    return kernels * (
        64 * full_word_batches * cell.nonzeros + relations**2 * cell.N
    )


def replay_bit_upper(
    cell: Cell, organization: str = "singleton", block_width: int = 64,
) -> int:
    """Main relation block's fresh-annihilation and independence bit bound."""
    kernels = {
        "singleton": cell.anchors,
        "cover": ceil(cell.anchors / cell.h),
        "common": 1,
    }[organization]
    return relation_replay_bit_upper(
        cell, retained_relation_capacity(cell, block_width), kernels,
    )


def branch_extraction_bitops_envelope(
    m: int, trials: int = BRANCH_EXTRACTION_TRIALS,
) -> int:
    """Modeled fail-closed multiplication-matrix decomposition envelope.

    The routine row-reduces the quadratic image, constructs its degree-two
    quotient multiplication maps, and tries public bounded lists of uniformly
    generated denominator and separator forms.  For every separator it scans
    all ``q`` possible eigenvalues by dense row reduction.  The constants are a
    reproducible arithmetic envelope, not a proved optimal operation count.
    The final factor ``m^2`` is the manuscript's optimistic bit-operation proxy for one
    GF(2^m) operation.
    """
    _require_positive("m", m)
    trials = _require_positive("trials", trials)
    quadratic_dimension = comb(m + 1, 2)
    field_size = 1 << m
    field_operations = (
        8 * quadratic_dimension**3
        + 32 * trials * field_size * m**3
    )
    return field_operations * m**2


def certification_per_anchor_bit_envelope(
    cell: Cell, branch_trials: int = BRANCH_EXTRACTION_TRIALS,
) -> int:
    """Modeled map-building and branch-extraction envelope at one anchor."""
    r = cell.relation_minimum
    return (
        r * cell.N * (1 + cell.d + comb(cell.d, 2))
        + r * cell.k**2
        + r * comb(cell.m, 2) ** 2
        + branch_extraction_bitops_envelope(cell.m, branch_trials)
    )


def certification_bit_envelope(
    cell: Cell, branch_trials: int = BRANCH_EXTRACTION_TRIALS,
) -> int:
    """Modeled certification envelope for all anchors in one configuration."""
    return cell.anchors * certification_per_anchor_bit_envelope(cell, branch_trials)


def keycheck_leading_bit_count(target: Target) -> int:
    """Leading parity-check/public-generator multiplication tally.

    This is the explicit ``2*m*t*n*k`` term.  It omits the lower-order rank
    check, so it is deliberately not named or reported as a strict upper bound.
    """
    k = target.n - target.m * target.t
    return 2 * target.m * target.t * target.n * k


def keycheck_bit_upper(target: Target) -> int:
    """Loose bit bound for public-code multiplication and parity-check rank."""
    mt = target.m * target.t
    k = target.n - mt
    return 2 * mt * target.n * k + 2 * mt**2 * target.n


def dense_row_echelon_multiplies(rows: int, unknowns: int) -> int:
    """Multiply count for ordinary elimination below successive pivots."""
    if rows < unknowns:
        raise ValueError("dense row-echelon sensitivity requires rows >= unknowns")
    return (
        (rows - unknowns) * unknowns * (unknowns - 1) // 2
        + unknowns * (unknowns - 1) * (2 * unknowns - 1) // 6
    )


def per_guess_field_ops(cell: Cell, solve: str, omega: float = 3.0) -> float:
    """Base-two logarithm of GF(2^m) operations for one interpolation solve."""
    U = cell.k * (cell.D_l + 1)
    B = comb(cell.k, 2)
    if solve == "dense":
        return 3 * log2(U)
    if solve == "nested":
        hot = U - (cell.anchors - 1) * B
        if hot < 2:
            raise ValueError("nonpositive nested hot dimension")
        return log2(B) + 2 * log2(hot)
    if solve == "order_basis":
        return (omega - 1) * log2(cell.k) + log2(cell.anchors * B)
    raise ValueError("solve must be dense, nested, or order_basis")


def guess_count(cell: Cell, normalization: str, phase_alignment: bool) -> float:
    fixed = {"affine": 2, "projective": 3}[normalization]
    exponent = cell.m * (cell.anchors - fixed)
    if not phase_alignment:
        exponent += (cell.anchors - 1) * log2(cell.m)
    return exponent


def wrong_leaf_surplus(cell: Cell) -> int:
    """Strict-profile row surplus ``c*B-U`` used only as a sensitivity."""
    unknowns = cell.k * (cell.D_l + 1)
    return cell.anchors * strict_profile_condition_count(cell.k) - unknowns


def wrong_leaf_random_row_log2_upper(
    cell: Cell, normalization: str = "affine", phase_alignment: bool = False,
) -> float:
    """Independent-uniform-row benchmark for all enumerated guesses.

    This is ``log2(G*q^(-Delta_wr))`` with ``G`` in place of the no-larger
    semantic wrong-leaf count.  It is not a theorem for the structured filtration
    rows and is never folded into an attack price.
    """
    return (
        guess_count(cell, normalization, phase_alignment)
        - cell.m * wrong_leaf_surplus(cell)
    )


def continuation_field_ops_upper(cell: Cell) -> int:
    """Deliberately loose field-operation envelope for one continued leaf.

    The terms cover, respectively, all public-column rank-maximization scans
    and basis updates, complete shortened-code matching, a fail-closed scan of
    projective chart poles followed by affine restoration through Vedenev's
    system (22), a final global multiplier solve, and Hemmert's
    support-to-Goppa completion using naive polynomial and dense linear
    algebra.  The chart loop makes every known shortened support finite and
    rejects unless each local restoration nullspace and the final global
    nullspace are one-dimensional with all-nonzero generators.  This is a
    coarse deterministic envelope, not a measured or sharp implementation
    cost.  Its repeated ``2*unknowns**3`` basis-update allowance also
    dominates one fresh dense solve used to materialize an exceptional correct
    prefix.
    """
    q_projective = (1 << cell.m) + 1
    unknowns = cell.k * (cell.D_l + 1)
    rank = min(cell.k - 1, unknowns)
    candidate_rank_test = (
        2 * cell.D_l * cell.k * unknowns
        + 4 * (cell.k - 1) * unknowns * rank
    )
    unassigned_shortened = max(cell.n_l - cell.anchors, 0)
    rank_maximization = unassigned_shortened * (
        q_projective * candidate_rank_test + 2 * unknowns**3
    )
    complete_matching = (
        2 * cell.D_l * cell.k * q_projective
        + 3 * cell.n_l * cell.k * q_projective
    )
    shortened_positions = cell.n - cell.n_l
    restore_shortening = (
        2 * q_projective * (shortened_positions * q_projective + 1) * cell.n**4
    )
    goppa_completion = 2 * q_projective * cell.n**4
    return (
        rank_maximization
        + complete_matching
        + restore_shortening
        + goppa_completion
    )


def continuation_bitops_upper(cell: Cell) -> int:
    """Coarse bit-operation envelope for one fail-closed continuation."""
    target = TARGETS[cell.target]
    projective_candidates = (1 << cell.m) + 1
    return (
        cell.m**2 * continuation_field_ops_upper(cell)
        + projective_candidates * keycheck_bit_upper(target)
    )


def correct_leaf_dimension_lower(cell: Cell) -> int:
    """Apon lower bound for a nondegenerate correct binary-Goppa leaf.

    The theorem gives ``dim(W) >= 2*t+4-c`` for ``c <= 2*t+2`` correctly
    labelled anchors.  Beyond that range this helper returns only the trivial
    lower bound supplied by the true coefficient curve.  The strict first-order
    profile assumed by the manuscript implies the theorem's
    nondegeneracy condition.  This bound neither changes the guessed-anchor
    count nor controls wrong leaves or the full-space ranks of public
    value-line updates.
    """
    target = TARGETS[cell.target]
    if cell.anchors <= 2 * target.t + 2:
        return 2 * target.t + 4 - cell.anchors
    return 1


def correct_continuation_labels_lower(cell: Cell) -> int:
    """Further distinct correct labels required by the Ap26 protected family.

    Lemma 7 of Ap26 applies already at derivative order zero.  Consequently,
    each correct public value-line restriction removes at most one dimension
    from the explicit true-curve subfamily, even though it may remove many
    other directions from the full candidate space.  The result is a lower
    bound only: it neither proves that this many labels suffice nor controls
    wrong-label ranks.
    """
    return correct_leaf_dimension_lower(cell) - 1


def accounted_plus_one_continuation_log2(priced: Price) -> float:
    """Arithmetic subtotal plus one fail-closed continuation attempt.

    Interpreting this as an end-to-end attack price additionally assumes that
    the public-column rank-maximization continuation succeeds on a correct
    leaf.  The function does not charge continuation at every guessed leaf.
    """
    return _logsum2(
        priced.total_log2,
        log2(continuation_bitops_upper(priced.cell)),
    )


def price(
    target: Target,
    k: int,
    d: int,
    *,
    block_width: int = 64,
    organization: str = "singleton",
    held_per_kernel: int | None = None,
    normalization: str = "affine",
    solve: str = "dense",
    phase_alignment: bool = False,
    charge_pair_alignment: bool = False,
    pair_relation_count: int | None = None,
    hermite_margin: int = 1,
    solver_attempts: int = 1,
    derivative_recovery_retries: int = 1,
    operator_kind: str = "auto",
) -> Price | None:
    """Price one modeled configuration.

    ``charge_pair_alignment`` adds ``c-1`` two-position hold-out relation
    computations and their fresh-annihilation/independence replay.  A
    displayed pair row sets ``pair_relation_count`` to the exact certified
    block size assumed by CrossPair (by default r_min).  Charging that work does
    not make phase alignment unconditional: the public pure cross-pairing map
    is still unconstructed.
    """
    if charge_pair_alignment and not phase_alignment:
        raise ValueError("charge_pair_alignment requires phase_alignment")
    if charge_pair_alignment and organization != "singleton":
        raise ValueError("pair alignment is defined between singleton spaces")
    if organization == "cover" and held_per_kernel is None:
        raise ValueError("cover organization requires held_per_kernel")
    if organization != "cover" and held_per_kernel is not None:
        raise ValueError("held_per_kernel is only valid for cover organization")
    probe = holdout_cell(target, k, d, 1, hermite_margin)
    if probe is None:
        return None
    held = {
        "singleton": 1,
        "cover": held_per_kernel,
        "common": probe.anchors,
    }[organization]
    assert held is not None
    cell = holdout_cell(target, k, d, held, hermite_margin)
    if cell is None:
        return None
    kind = resolve_operator_kind(cell, operator_kind)
    if solve == "nested":
        unknowns = cell.k * (cell.D_l + 1)
        conditions_per_anchor = comb(cell.k, 2)
        if unknowns - (cell.anchors - 1) * conditions_per_anchor < 2:
            return None
    relation_generation, state = relation_generation_price(
        cell,
        block_width=block_width,
        organization=organization,
        solver_attempts=solver_attempts,
        operator_kind=kind,
    )
    vector_state, sequence_state, solver_state = solver_state_bytes(
        cell, block_width, kind,
    )
    solver_state_check, relation_state, active_peak = active_peak_state_bytes(
        cell, block_width, kind,
    )
    assert solver_state_check == solver_state and state == active_peak
    state = active_peak
    replay = log2(replay_bit_upper(cell, organization, block_width))
    certification = log2(certification_bit_envelope(cell))
    guesses = guess_count(cell, normalization, phase_alignment)
    field_ops = per_guess_field_ops(cell, solve)
    per_guess_bitops = field_ops + log2(cell.m * cell.m)
    downstream = guesses + per_guess_bitops
    derivative_recovery = derivative_recovery_bitops_log2(
        cell, derivative_recovery_retries,
    )
    alignment_relation_generation = None
    alignment_replay = None
    if charge_pair_alignment:
        alignment = pair_alignment_relation_generation_price(
            target, k, d, cell.anchors,
            block_width=block_width,
            relation_count=pair_relation_count,
            solver_attempts=solver_attempts,
            hermite_margin=hermite_margin,
            operator_kind=operator_kind,
        )
        if alignment is None:
            return None
        alignment_relation_generation, alignment_state, pair_cell = alignment
        pair_relations = (
            pair_cell.relation_minimum
            if pair_relation_count is None
            else pair_relation_count
        )
        alignment_replay = log2(relation_replay_bit_upper(
            pair_cell, pair_relations, cell.anchors - 1,
        ))
        if alignment_state > state:
            state = alignment_state
            pair_kind = resolve_operator_kind(pair_cell, operator_kind)
            vector_state, sequence_state, solver_state = solver_state_bytes(
                pair_cell, block_width, pair_kind,
            )
            _, relation_state, active_peak = active_peak_state_bytes(
                pair_cell, block_width, pair_kind,
            )
    terms = [
        relation_generation,
        replay,
        certification,
        downstream,
        derivative_recovery,
    ]
    if alignment_relation_generation is not None:
        terms.append(alignment_relation_generation)
    if alignment_replay is not None:
        terms.append(alignment_replay)
    total = _logsum2(*terms)
    return Price(
        cell=cell,
        block_width=block_width,
        operator_kind=kind,
        organization=organization,
        normalization=normalization,
        solve=solve,
        phase_alignment=phase_alignment,
        relation_generation_log2=relation_generation,
        guess_log2=guesses,
        per_guess_field_ops_log2=field_ops,
        per_guess_bitops_log2=per_guess_bitops,
        downstream_log2=downstream,
        replay_log2=replay,
        certification_log2=certification,
        derivative_recovery_log2=derivative_recovery,
        derivative_recovery_retries=derivative_recovery_retries,
        alignment_relation_generation_log2=alignment_relation_generation,
        alignment_replay_log2=alignment_replay,
        alignment_cost_charged=charge_pair_alignment,
        pairing_construction_assumed=phase_alignment,
        total_log2=total,
        vector_state_bytes=vector_state,
        sequence_state_bytes=sequence_state,
        solver_state_bytes=solver_state,
        relation_state_bytes=relation_state,
        active_peak_state_bytes=active_peak,
        state_bytes=state,
        # Every row is conditional on binary solver yield and sufficient
        # relation richness for higher-order derivative recovery; the following
        # options add further explicit assumptions.
        conditional_subtotal=True,
    )


def dense_elimination_sensitivity_log2(
    priced: Price, multiplier_bitops: int,
) -> float:
    """Accounted dense subtotal under an assumed multiplier cost.

    The row-echelon multiply/add count is explicit.  The multiplier circuit,
    field basis, and caller-supplied bit-operation count remain an external sensitivity
    assumption.  This helper is not applicable to the multiply-shaped nested
    or order-basis monomials.
    """
    if priced.solve != "dense":
        raise ValueError("dense sensitivity requires a dense Price")
    multiplier_bitops = _require_positive("multiplier_bitops", multiplier_bitops)
    cell = priced.cell
    rows = cell.anchors * strict_profile_condition_count(cell.k)
    unknowns = cell.k * (cell.D_l + 1)
    multiplies = dense_row_echelon_multiplies(rows, unknowns)
    resolved_downstream = priced.guess_log2 + log2(
        multiplies * (multiplier_bitops + cell.m)
    )
    terms = [
        priced.relation_generation_log2,
        priced.replay_log2,
        priced.certification_log2,
        priced.derivative_recovery_log2,
        resolved_downstream,
    ]
    if priced.alignment_relation_generation_log2 is not None:
        terms.append(priced.alignment_relation_generation_log2)
    if priced.alignment_replay_log2 is not None:
        terms.append(priced.alignment_replay_log2)
    return _logsum2(*terms)


def continuation_no_success_upper_log2(priced: Price) -> float:
    """Coarse deterministic ``log2(G*C_ext)`` no-success traversal ceiling."""
    return priced.guess_log2 + log2(continuation_bitops_upper(priced.cell))


LADDER = ((9, 208, 8), (8, 215, 8), (7, 258, 9), (6, 322, 9), (5, 429, 10))


@lru_cache(maxsize=None)
def one_position_relation_generation_floor(
    target: Target, maximum_degree: int = 24, block_width: int = 64,
    operator_kind: str = "auto",
) -> tuple[int, Cell, float]:
    """Count admitted singleton cells and minimize relation-generation work."""
    admitted = 0
    best_cell: Cell | None = None
    best_log = float("inf")
    for d in range(3, maximum_degree + 1):
        for k in range(d + 2, target.n - target.m * target.t + 1):
            cell = holdout_cell(target, k, d, 1)
            if cell is None:
                continue
            admitted += 1
            work, _ = relation_generation_price(
                cell, block_width=block_width, organization="singleton",
                operator_kind=operator_kind,
            )
            if (work, k, d) < (best_log, best_cell.k if best_cell else 0, best_cell.d if best_cell else 0):
                best_cell, best_log = cell, work
    if best_cell is None:
        raise AssertionError("relation-generation scan found no cell")
    return admitted, best_cell, best_log


def one_batch_word_width(cell: Cell) -> int:
    """Smallest multiple of 64 able to request ``r_min`` relations at once."""
    return 64 * ceil(cell.relation_minimum / 64)


@lru_cache(maxsize=None)
def practical_wide_relation_generation_floor(
    target: Target, maximum_degree: int = 24,
) -> tuple[int, Cell, int, float]:
    """Minimize work using one practical multiword batch at each cell."""
    admitted = 0
    best: tuple[float, int, int, int, Cell] | None = None
    for d in range(3, maximum_degree + 1):
        for k in range(d + 2, target.n - target.m * target.t + 1):
            cell = holdout_cell(target, k, d, 1)
            if cell is None:
                continue
            admitted += 1
            width = one_batch_word_width(cell)
            work, _ = relation_generation_price(
                cell, block_width=width, organization="singleton",
            )
            candidate = (work, k, d, width, cell)
            if best is None or candidate[:4] < best[:4]:
                best = candidate
    if best is None:
        raise AssertionError("wide relation-generation scan found no cell")
    work, _, _, width, cell = best
    return admitted, cell, width, work


@lru_cache(maxsize=None)
def jointly_optimized_price(
    target: Target,
    maximum_degree: int = 24,
    normalization: str = "affine",
    solve: str = "dense",
    wide: bool = False,
) -> tuple[int, Price]:
    """Minimize one accounted modeled subtotal over admitted singleton cells.

    ``wide=False`` fixes ``b=64``.  ``wide=True`` uses the smallest multiple of
    64 that requests the minimum relation block in one recovery.  Both scans
    use the exact padded operator whenever ``R_M<=N`` and the augmented
    fallback otherwise.
    """
    admitted = 0
    best: Price | None = None
    for d in range(3, maximum_degree + 1):
        for k in range(d + 2, target.n - target.m * target.t + 1):
            cell = holdout_cell(target, k, d, 1)
            if cell is None:
                continue
            admitted += 1
            width = one_batch_word_width(cell) if wide else 64
            got = price(
                target, k, d, block_width=width,
                normalization=normalization, solve=solve,
            )
            if got is None:
                continue
            if best is None or (
                got.total_log2, got.state_bytes, got.cell.k, got.cell.d, got.block_width
            ) < (
                best.total_log2, best.state_bytes, best.cell.k, best.cell.d,
                best.block_width,
            ):
                best = got
    if best is None:
        raise AssertionError("joint subtotal scan found no cell")
    return admitted, best


RELATION_BLOCK_MODELS = (
    "first-order",
    "2k",
    "k^2",
    "6.9k^2",
    "162k^2",
)


def modeled_relation_block_size(cell: Cell, model: str) -> int:
    """Return the assumed block size for a higher-order derivative recovery sensitivity.

    The first-order model is the block justified by the two certified local
    rank tests.  The larger laws are stress tests, not claims about how many
    relations actually suffice for higher-order derivative recovery.
    """
    if model == "first-order":
        return cell.relation_minimum
    if model == "2k":
        return 2 * cell.k
    if model == "k^2":
        return cell.k**2
    if model == "6.9k^2":
        return ceil(Fraction(69, 10) * cell.k**2)
    if model == "162k^2":
        return 162 * cell.k**2
    raise ValueError(f"unknown relation-block model: {model}")


def reprice_relation_block(
    priced: Price, relation_count: int, model: str = "fixed",
) -> RelationBlockSensitivity | None:
    """Reprice one width-64 singleton Affine+dense cell at a larger block.

    The cloned cell substitutes ``relation_count`` for the first-order block
    minimum in relation generation, replay, certification, the arithmetic
    envelope for higher-order derivative recovery, and retained relation state.
    The downstream enumeration and interpolation term is unchanged. A request larger than
    the rigorous kernel floor is refused.
    """
    relation_count = _require_positive("relation_count", relation_count)
    if (
        priced.block_width != 64
        or priced.organization != "singleton"
        or priced.normalization != "affine"
        or priced.solve != "dense"
    ):
        raise ValueError(
            "relation-block sensitivity requires width-64 singleton Affine+dense"
        )
    if relation_count > priced.cell.kernel_floor:
        return None
    cell = replace(priced.cell, relation_minimum=relation_count)
    relation_generation, state = relation_generation_price(cell)
    replay = log2(replay_bit_upper(cell))
    certification = log2(certification_bit_envelope(cell))
    derivative_recovery = derivative_recovery_bitops_log2(
        cell, priced.derivative_recovery_retries,
    )
    total = _logsum2(
        relation_generation,
        replay,
        certification,
        derivative_recovery,
        priced.downstream_log2,
    )
    return RelationBlockSensitivity(
        model=model,
        cell=priced.cell,
        relation_count=relation_count,
        relation_generation_log2=relation_generation,
        total_log2=total,
        state_bytes=state,
    )


@lru_cache(maxsize=None)
def jointly_optimized_relation_block_sensitivity(
    target: Target,
    model: str,
    maximum_degree: int = 24,
) -> tuple[int, RelationBlockSensitivity]:
    """Re-optimize the Category-style subtotal under one block-size law."""
    if model not in RELATION_BLOCK_MODELS:
        raise ValueError(f"unknown relation-block model: {model}")
    admitted = 0
    best: RelationBlockSensitivity | None = None
    for d in range(3, maximum_degree + 1):
        for k in range(d + 2, target.n - target.m * target.t + 1):
            priced = price(target, k, d)
            if priced is None:
                continue
            admitted += 1
            candidate = reprice_relation_block(
                priced,
                modeled_relation_block_size(priced.cell, model),
                model,
            )
            if candidate is None:
                continue
            if best is None or (
                candidate.total_log2,
                candidate.state_bytes,
                candidate.cell.k,
                candidate.cell.d,
                candidate.relation_count,
            ) < (
                best.total_log2,
                best.state_bytes,
                best.cell.k,
                best.cell.d,
                best.relation_count,
            ):
                best = candidate
    if best is None:
        raise AssertionError("relation-block sensitivity scan found no cell")
    return admitted, best


def nonzeros_at_weight(cell: Cell, weight: int) -> int:
    """Re-evaluate one fixed cell's direct-traversal nonzeros at a new weight."""
    if not 0 <= weight <= cell.k:
        raise ValueError("weight must lie in [0,k]")
    result = cell.schedule.high_columns * point_block_nnz(
        cell.k, weight, cell.d, cell.high_levels,
    )
    if cell.schedule.low_columns:
        result += cell.schedule.low_columns * point_block_nnz(
            cell.k, weight, cell.d, cell.low_levels,
        )
    return result


def selected_prices() -> dict[str, dict[str, Price]]:
    result: dict[str, dict[str, Price]] = {}
    for target in TARGETS.values():
        dense = price(target, target.selected_k, target.selected_d)
        nested = price(
            target, target.selected_k, target.selected_d,
            normalization="projective", solve="nested",
        )
        assert dense is not None and nested is not None
        result[target.name] = {"dense": dense, "nested": nested}
    return result


@lru_cache(maxsize=None)
def memory_cap_rows(maximum_degree: int = 24) -> tuple[tuple[str, Price | None], ...]:
    """Exhaustively minimize the PGL+nested variant under four state caps."""
    target = TARGETS["mceliece348864"]
    candidates: list[Price] = []
    for d in range(3, maximum_degree + 1):
        for k in range(d + 2, target.n - target.m * target.t + 1):
            probe = holdout_cell(target, k, d, 1)
            if probe is None:
                continue
            widths = tuple(sorted({64, 32, 16, 8, 4, 2, 1, one_batch_word_width(probe)}))
            for width in widths:
                for organization in ("singleton", "common"):
                    got = price(
                        target, k, d, block_width=width, organization=organization,
                        normalization="projective", solve="nested",
                    )
                    if got is not None:
                        candidates.append(got)
    rows = []
    for label, cap in (("53.00", PIB), ("56.32", 10 * PIB), ("59.64", 100 * PIB), ("63.00", EIB)):
        fitting = [item for item in candidates if item.state_bytes <= cap]
        best = min(
            fitting,
            key=lambda item: (
                item.total_log2, item.state_bytes, item.cell.k, item.cell.d,
                item.block_width, item.organization,
            ),
        ) if fitting else None
        rows.append((label, best))
    return tuple(rows)


def _state_bits(value: int) -> str:
    """Return log2 of the retained number of bits, as used in tables."""
    return f"{log2(8 * value):.2f}"


def _power_of_two_bits(value: int) -> str:
    """Format retained state in the paper's directly comparable bit unit."""
    return f"2^{log2(8 * value):.2f} bits"


def _print_table(title: str, headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    """Print a compact plain-text table without third-party dependencies."""
    rendered = [[str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered:
        for column, value in enumerate(row):
            widths[column] = max(widths[column], len(value))
    print(f"\n{title}")
    print("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rendered:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def internal_checks() -> None:
    """Pin key independent values so silent formula drift fails closed."""
    assert NIST_BIT_OPERATION_THRESHOLDS == {
        1: 143.0, 2: 146.0, 3: 207.0, 4: 210.0, 5: 272.0,
    }
    for k in range(3, 32):
        assert strict_profile_condition_count(k) == comb(k, 2)
        assert indexed_filtration_condition_count(k, tuple(range(1, k))) == comb(k, 2)
    for invalid_profile in (
        (5,), (1, 3, 3, 4), (1, 2, 4, 4), (1, 2, 3, 5), (1, 1, 3, 4),
    ):
        try:
            indexed_filtration_condition_count(5, invalid_profile)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid indexed profile accepted: {invalid_profile}")
    assert lucas_levels(8, 7) == (0, 4, 6)
    assert lucas_levels(8, 6) == (0, 4, 5)
    assert lucas_levels(9, 8) == (1, 5, 7)
    assert lucas_levels(9, 7) == (1, 5, 6)
    # Every binary point weight, including zero and the low-weight boundary,
    # is a valid input to the exact rank and nonzero formulas.
    for k in range(3, 11):
        for d in range(1, k + 1):
            for s in range(1, d + 1):
                levels = lucas_levels(d, s)
                for weight in range(k + 1):
                    assert point_block_rank(k, weight, d, s) >= 0
                    assert point_block_nnz(k, weight, d, levels) >= 0
    expected = {
        "mceliece348864": (112.35, 172.13, 146.29),
        "mceliece460896": (119.49, 200.03, 176.77),
        "mceliece6688128": (123.11, 235.09, 207.83),
        "mceliece6960119": (122.10, 217.96, 194.35),
        "mceliece8192128": (123.11, 235.09, 207.83),
    }
    for name, models in selected_prices().items():
        got = (
            models["dense"].relation_generation_log2,
            models["dense"].total_log2,
            models["nested"].total_log2,
        )
        for observed, wanted in zip(got, expected[name]):
            assert abs(observed - wanted) < 0.006, (name, observed, wanted)
        cell = models["dense"].cell
        uniform_hermite_order = ceil((cell.order_target + 1) / cell.constrained_columns)
        uniform_dimension_order = max(
            order for order in range(1, cell.d)
            if cell.constrained_columns * point_block_rank(
                cell.k, cell.modeled_column_weight, cell.d, order,
            ) < cell.N
        )
        assert uniform_dimension_order + 1 == uniform_hermite_order
    target = TARGETS["mceliece348864"]
    floor_edge = holdout_cell(target, 1283, 17)
    assert floor_edge is not None and floor_edge.anchors == 4
    floor_edge_U = floor_edge.k * (floor_edge.D_l + 1)
    assert anchor_count(floor_edge.k, floor_edge.D_l) == 4
    assert floor_edge_U - (floor_edge.anchors - 1) * comb(floor_edge.k, 2) <= 0
    floor_edge_dense = price(target, 1283, 17)
    floor_edge_nested = price(
        target, 1283, 17, normalization="projective", solve="nested",
    )
    assert floor_edge_dense is not None and floor_edge_dense.cell.anchors == 4
    assert floor_edge_nested is None
    c7 = price(target, 258, 9)
    assert c7 is not None
    assert c7.cell.anchors == 7 and abs(c7.total_log2 - 142.1451313207443) < 1e-12
    assert wrong_leaf_surplus(c7.cell) == 387
    assert abs(wrong_leaf_random_row_log2_upper(c7.cell) + 4562.490224995673) < 1e-12
    assert dense_row_echelon_multiplies(
        c7.cell.anchors * strict_profile_condition_count(c7.cell.k),
        c7.cell.k * (c7.cell.D_l + 1),
    ) == 4_155_763_813_455_136
    assert abs(dense_elimination_sensitivity_log2(c7, 250) - 141.4272900732) < 1e-10
    assert continuation_field_ops_upper(c7.cell) == 12_233_673_568_707_332_013_064_944
    assert continuation_bitops_upper(c7.cell) == 1_761_648_993_893_932_371_328_214_784
    assert abs(log2(continuation_bitops_upper(c7.cell)) - 90.50898505987522) < 1e-12
    assert correct_leaf_dimension_lower(c7.cell) == 125
    assert correct_continuation_labels_lower(c7.cell) == 124
    assert abs(accounted_plus_one_continuation_log2(c7) - c7.total_log2) < 1e-12
    assert abs(continuation_no_success_upper_log2(c7) - 172.01876006420216) < 1e-12
    for name, k, d, wanted_dimension, wanted_labels in (
        ("mceliece348864", 322, 9, 126, 125),
        ("mceliece460896", 424, 9, 189, 188),
        ("mceliece6688128", 565, 9, 253, 252),
        ("mceliece6960119", 525, 9, 235, 234),
        ("mceliece8192128", 565, 9, 253, 252),
    ):
        joint_cell = price(TARGETS[name], k, d)
        assert joint_cell is not None
        assert correct_leaf_dimension_lower(joint_cell.cell) == wanted_dimension
        assert correct_continuation_labels_lower(joint_cell.cell) == wanted_labels
    relation_sensitivity_expected = {
        "first-order": ((322, 9, 6), 375, 132.14560837435326, 132.215688254905),
        "2k": ((322, 9, 6), 644, 133.0200774922694, 133.0587242353749),
        "k^2": ((322, 9, 6), 103_684, 140.22331424914964, 140.22358004920324),
        "6.9k^2": ((258, 9, 7), 459_292, 137.2865697722559, 142.1939880854629),
        "162k^2": ((258, 9, 7), 10_783_368, 141.8397165727746, 143.00047172026922),
    }
    for model, (wanted_cell, wanted_count, wanted_relation, wanted_total) in (
        relation_sensitivity_expected.items()
    ):
        sensitivity_admitted, sensitivity = jointly_optimized_relation_block_sensitivity(
            target, model,
        )
        assert sensitivity_admitted == 19_338
        assert (
            sensitivity.cell.k, sensitivity.cell.d, sensitivity.cell.anchors,
        ) == wanted_cell
        assert sensitivity.relation_count == wanted_count
        assert abs(sensitivity.relation_generation_log2 - wanted_relation) < 1e-12
        assert abs(sensitivity.total_log2 - wanted_total) < 1e-12
    c6 = price(target, 322, 9)
    assert c6 is not None
    c6_threshold = reprice_relation_block(c6, 711_000)
    assert c6_threshold is not None
    assert abs(c6_threshold.total_log2 - 143.00025587683027) < 1e-12
    assert c7.cell.schedule.order_total == c7.cell.order_target + 1
    # The padded square operator has the exact right kernel, half the rank
    # bound, and one sparse traversal per application.
    assert c7.cell.padding_available
    assert resolve_operator_kind(c7.cell) == "padded"
    assert operator_dimension(c7.cell) == c7.cell.N
    assert operator_rank_bound(c7.cell) == c7.cell.row_rank_bound
    assert coppersmith_applications(c7.cell, 64) == 3 * ceil(
        c7.cell.row_rank_bound / 64,
    )
    augmented_work, augmented_active = relation_generation_price(
        c7.cell, block_width=64, operator_kind="augmented",
    )
    _, _, augmented_solver = solver_state_bytes(c7.cell, 64, "augmented")
    assert abs(c7.relation_generation_log2 - augmented_work + 2) < 1e-9
    assert round(log2(8 * augmented_solver), 3) == 61.660
    assert round(log2(8 * c7.solver_state_bytes), 3) == 60.738
    assert round(log2(8 * augmented_active), 3) == 62.549
    assert round(log2(8 * c7.state_bytes), 3) == 62.124
    c9 = price(target, 208, 8, block_width=32)
    c8 = price(target, 215, 8, block_width=32)
    assert c9 is not None and c8 is not None
    assert c9.state_bytes == 3_332_402_297_816_696
    assert c8.state_bytes == 4_317_043_579_322_568
    for cell in (c9.cell, c8.cell, c7.cell):
        largest_j = cell.d - 2
        assert (cell.k - 2 * largest_j) ** 2 >= cell.k + 2
        projected_relations = 64 * ceil(cell.relation_minimum / 64)
        assert projected_relations < cell.kernel_floor
    assert max(
        k for k in range(10, target.n - target.m * target.t + 1)
        if holdout_cell(target, k, 8, 1) is not None
    ) == 256
    cap_rows = memory_cap_rows()
    cap_expected = (
        ("53.00", None),
        ("56.32", (246, 8, 2, "singleton", 132.70515526776964, 11_228_185_523_421_754)),
        ("59.64", (256, 8, 320, "singleton", 126.87394767903615, 38_999_720_556_665_600)),
        ("63.00", (258, 9, 64, "common", 124.31739332242026, 637_306_146_948_865_536)),
    )
    for (label, got), expected_cap in zip(cap_rows, cap_expected):
        wanted_label, values = expected_cap
        assert label == wanted_label
        if values is None:
            assert got is None
            continue
        assert got is not None
        k, d, width, organization, total, state = values
        assert (got.cell.k, got.cell.d, got.block_width, got.organization) == (
            k, d, width, organization,
        )
        assert abs(got.total_log2 - total) < 1e-12
        assert got.state_bytes == state
    admitted, floor_cell, floor_log = one_position_relation_generation_floor(target)
    assert admitted == 19_338
    assert (floor_cell.k, floor_cell.d, floor_cell.anchors) == (208, 8, 9)
    assert abs(floor_log - 112.35333862656522) < 1e-12
    wide_admitted, wide_cell, wide_width, wide_log = practical_wide_relation_generation_floor(
        target,
    )
    assert wide_admitted == admitted
    assert (wide_cell.k, wide_cell.d, wide_cell.anchors, wide_width) == (208, 8, 9, 320)
    assert abs(wide_log - 110.03141053167785) < 1e-12
    margin42 = holdout_cell(target, 215, 8, 1, 42)
    assert margin42 is not None
    assert margin42.schedule.text == "767@7+0@6"
    assert margin42.kernel_floor == 1_187_277_582_725
    assert holdout_cell(target, 215, 8, 1, 43) is None
    cover4 = price(target, 215, 8, block_width=32, organization="cover", held_per_kernel=4)
    assert cover4 is not None and cover4.cell.kernel_floor == 4_055_009_590_230
    assert abs(
        cover4.relation_generation_log2
        - c8.relation_generation_log2
        + 1.97119
    ) < 0.001
    gauge = holdout_cell(target, 215, 8, 1)
    assert gauge is not None
    assert abs(log2(nonzeros_at_weight(gauge, 60) / gauge.nonzeros) + 2.149284) < 1e-6
    charged_expected = {
        (215, 8, 64): (113.9219, 0.7887, 70.71352101999118),
        (215, 8, 32): (115.6608, 0.8735, 70.71352101999118),
        (215, 8, 16): (117.5460, 0.9011, 70.71352101999118),
        (258, 9, 32): (129.6980, 0.8986, 78.02915763435273),
    }
    for (k, d, width), (
        wanted_total, wanted_increment, wanted_replay,
    ) in charged_expected.items():
        omitted = price(
            target, k, d, block_width=width, normalization="projective",
            solve="nested", phase_alignment=True,
        )
        charged = price(
            target, k, d, block_width=width, normalization="projective",
            solve="nested", phase_alignment=True, charge_pair_alignment=True,
        )
        assert omitted is not None and charged is not None
        assert charged.alignment_cost_charged and charged.pairing_construction_assumed
        assert charged.alignment_replay_log2 is not None
        assert charged.conditional_subtotal
        assert abs(charged.total_log2 - wanted_total) < 0.001
        assert abs(charged.total_log2 - omitted.total_log2 - wanted_increment) < 0.001
        assert abs(charged.alignment_replay_log2 - wanted_replay) < 1e-12


def print_report() -> None:
    selected = selected_prices()
    print("MCCOST — self-contained Classic McEliece cost model")
    print("All exponents are base two. Work is in elementary bit operations.")
    print("Sparse work charges ceil(b/64) packed words per incidence; GF(2^m)")
    print("multiplication uses the optimistic m^2 binary-operation proxy and omits XORs.")
    print("Memory is active retained state: the larger of the final relation block")
    print("and solver state plus previously certified relation batches.")
    print("In a schedule a@s, a columns are assigned vanishing order s.")
    print("Relation generation uses one nominal equal-block Coppersmith attempt per")
    print("required relation batch on B=E*M when R_M<=N, with the augmented operator")
    print("only as a fallback; retries, access,")
    print("and temporary solver memory are not charged.")
    print("Every displayed subtotal stops at candidate interpolation.")
    print("Apon proves that the selected anchors leave the correct leaf nonunique")
    print("and identifies full-code candidate testing as a nontrivial list problem.")
    print("The strict first-order profile supplies its nondegeneracy condition.")
    print("The manuscript models that step as public-column continuation; one")
    print("attempt is lower order, but target-scale success remains heuristic.")
    print("Generic wrong-leaf rank is a separate manuscript assumption.")
    print("The anchor count uses max(4,ceil(U/B)) with strict B=binom(k_l,2).")
    print("Indexed E_i>=B is represented but is not used to reduce c without a rank theorem.")
    print("The r_min block is justified only for first-order certification; larger")
    print("block-size laws for higher-order derivative recovery are reoptimized as sensitivities.")

    target_rows: list[tuple[object, ...]] = []
    for target in TARGETS.values():
        dense = selected[target.name]["dense"]
        nested = selected[target.name]["nested"]
        cell = dense.cell
        target_rows.append((
            target.name,
            target.category,
            f"({cell.k},{cell.d},1)",
            cell.schedule.text,
            cell.anchors,
            f"{log2(cell.N):.2f}",
            f"{dense.relation_generation_log2:.2f}",
            f"{dense.total_log2:.2f}",
            f"{nested.total_log2:.2f}",
            _power_of_two_bits(dense.state_bytes),
        ))
    _print_table(
        "Selected cells and accounted candidate-interpolation subtotals",
        (
            "target", "cat", "(k_l,d,h)", "orders", "c", "log N",
            "relation generation", "Affine+dense subtotal", "PGL+nested subtotal", "state",
        ),
        target_rows,
    )

    joint_rows: list[tuple[object, ...]] = []
    joint_fixed: dict[str, Price] = {}
    for target in TARGETS.values():
        _, fixed = jointly_optimized_price(target, normalization="affine", solve="dense")
        joint_fixed[target.name] = fixed
        _, wide = jointly_optimized_price(
            target, normalization="affine", solve="dense", wide=True,
        )
        joint_rows.append((
            target.name,
            f"({fixed.cell.k},{fixed.cell.d},{fixed.cell.anchors},64)",
            f"{fixed.total_log2:.2f}", _power_of_two_bits(fixed.state_bytes),
            f"({wide.cell.k},{wide.cell.d},{wide.cell.anchors},{wide.block_width})",
            f"{wide.total_log2:.2f}", _power_of_two_bits(wide.state_bytes),
        ))
    _print_table(
        "Jointly optimized Affine+dense subtotals",
        (
            "target", "fixed-width cell", "b=64 subtotal", "active state",
            "wide cell", "wide subtotal", "active state",
        ),
        joint_rows,
    )

    richness_rows: list[tuple[object, ...]] = []
    category_one = TARGETS["mceliece348864"]
    for model in RELATION_BLOCK_MODELS:
        _, sensitivity = jointly_optimized_relation_block_sensitivity(
            category_one, model,
        )
        richness_rows.append((
            model,
            f"({sensitivity.cell.k},{sensitivity.cell.d},{sensitivity.cell.anchors})",
            f"{sensitivity.relation_count:,}",
            f"{sensitivity.relation_generation_log2:.2f}",
            f"{sensitivity.total_log2:.2f}",
            _power_of_two_bits(sensitivity.state_bytes),
        ))
    _print_table(
        "Category-1 relation-block-size sensitivity for higher-order derivative recovery",
        ("block law", "cell", "relations", "C_rel", "subtotal", "active state"),
        richness_rows,
    )

    continuation_rows: list[tuple[object, ...]] = []
    for target in TARGETS.values():
        fixed = joint_fixed[target.name]
        continuation_rows.append((
            target.name,
            fixed.cell.anchors,
            correct_leaf_dimension_lower(fixed.cell),
            correct_continuation_labels_lower(fixed.cell),
            f"{log2(continuation_bitops_upper(fixed.cell)):.2f}",
            f"{accounted_plus_one_continuation_log2(fixed):.2f}",
        ))
    _print_table(
        "Apon correct-leaf bound and priced list-problem continuation",
        (
            "target", "c", "correct dim >=", "further correct labels >=",
            "log2(C_ext)",
            "subtotal + one C_ext",
        ),
        continuation_rows,
    )

    print("\nExact selected-cell arithmetic")
    for target in TARGETS.values():
        got = selected[target.name]["dense"]
        cell = got.cell
        hermite_left = cell.k * (cell.d - 1) + cell.schedule.order_total
        hermite_right = cell.d * cell.D_l
        batches = ceil(cell.relation_minimum / 64)
        applications = coppersmith_applications(
            cell, got.block_width, got.operator_kind,
        )
        print(f"\n  {target.name}")
        print(
            f"    shortened shape: n_l={cell.n_l}, k_l={cell.k}, "
            f"D_l={cell.D_l}; d={cell.d}, h={cell.h}"
        )
        print(
            f"    Hermite: {cell.k}*({cell.d}-1)+{cell.schedule.order_total}"
            f" = {hermite_left} > {hermite_right} = {cell.d}*{cell.D_l}"
        )
        print(
            f"    schedule: {cell.schedule.text}; Lucas levels "
            f"{cell.high_levels}/{cell.low_levels}; modeled non-pivot weight "
            f"{cell.modeled_column_weight}"
        )
        print(
            f"    dimensions: N={cell.N:,}; rows={cell.operator_rows:,}; "
            f"rank(M)<={cell.row_rank_bound:,}; dim ker(M)>={cell.kernel_floor:,}"
        )
        print(
            f"    sparse operator: nnz={cell.nonzeros:,}; kind={got.operator_kind}; "
            f"rank(solver)<={operator_rank_bound(cell, got.operator_kind):,}; "
            f"applications={applications:,}"
        )
        print(
            f"    relation block: r_min={cell.relation_minimum}, batches={batches}; "
            f"anchors={cell.anchors}; relation generation="
            f"2^{got.relation_generation_log2:.4f} bit operations"
        )
        print(
            f"    retained state: vector={_power_of_two_bits(got.vector_state_bytes)}, "
            f"sequence={_power_of_two_bits(got.sequence_state_bytes)}, "
            f"solver={_power_of_two_bits(got.solver_state_bytes)}; retained relations="
            f"{_power_of_two_bits(got.relation_state_bytes)}; active peak="
            f"{_power_of_two_bits(got.state_bytes)}"
        )

    target = TARGETS["mceliece348864"]
    c7_padding = price(target, 258, 9)
    assert c7_padding is not None
    augmented_work, augmented_active = relation_generation_price(
        c7_padding.cell, block_width=64, operator_kind="augmented",
    )
    _, _, augmented_solver = solver_state_bytes(c7_padding.cell, 64, "augmented")
    print("\nmceliece348864 c=7 padding comparison at b=64")
    print(
        f"  relation generation: augmented 2^{augmented_work:.4f}, "
        f"padded 2^{c7_padding.relation_generation_log2:.4f}, change "
        f"{c7_padding.relation_generation_log2 - augmented_work:+.4f} bits"
    )
    print(
        f"  solver state: augmented {_power_of_two_bits(augmented_solver)}, "
        f"padded {_power_of_two_bits(c7_padding.solver_state_bytes)}; active peak: "
        f"augmented {_power_of_two_bits(augmented_active)}, "
        f"padded {_power_of_two_bits(c7_padding.state_bytes)}"
    )

    ladder_rows: list[tuple[object, ...]] = []
    for anchors, k, d in LADDER:
        dense = price(target, k, d)
        nested = price(target, k, d, normalization="projective", solve="nested")
        assert dense is not None and nested is not None and dense.cell.anchors == anchors
        ladder_rows.append((
            anchors, f"({k},{d})", f"{dense.relation_generation_log2:.2f}",
            f"{dense.guess_log2:.2f}", f"{wrong_leaf_surplus(dense.cell):,}",
            f"{wrong_leaf_random_row_log2_upper(dense.cell):.2f}",
            f"{dense.total_log2:.2f}",
            f"{nested.total_log2:.2f}", _power_of_two_bits(dense.state_bytes),
        ))
    _print_table(
        "mceliece348864 anchor ladder",
        (
            "c", "(k_l,d)", "relation generation", "affine guesses", "Delta_wr",
            "affine random-row log upper", "Affine+dense", "PGL+nested", "state",
        ),
        ladder_rows,
    )

    cap_rows = []
    for label, got in memory_cap_rows():
        if got is None:
            cap_rows.append((label, "--", "--", "--", "--", "no feasible cell"))
        else:
            cap_rows.append((
                label,
                f"({got.cell.k},{got.cell.d},{got.cell.anchors})",
                got.block_width,
                got.organization,
                f"{got.total_log2:.2f}",
                _power_of_two_bits(got.state_bytes),
            ))
    _print_table(
        "Exhaustive mceliece348864 PGL+nested optima under state caps",
        ("cap", "(k_l,d,c)", "b", "organization", "subtotal", "state"),
        cap_rows,
    )

    admitted, floor_cell, floor_log = one_position_relation_generation_floor(target)
    _, floor_state = relation_generation_price(floor_cell, block_width=64)
    print(
        f"\nOne-position relation-generation floor: scanned {admitted:,} admitted h=1 cells "
        f"with 3<=d<=24; optimum (k_l,d,c,b)=({floor_cell.k},{floor_cell.d},"
        f"{floor_cell.anchors},64), 2^{floor_log:.4f} bit operations, "
        f"{_power_of_two_bits(floor_state)} retained state."
    )
    _, wide_floor_cell, wide_floor_width, wide_floor_log = (
        practical_wide_relation_generation_floor(target)
    )
    _, wide_floor_state = relation_generation_price(
        wide_floor_cell, block_width=wide_floor_width,
    )
    print(
        f"Practical one-batch floor: (k_l,d,c,b)=({wide_floor_cell.k},"
        f"{wide_floor_cell.d},{wide_floor_cell.anchors},{wide_floor_width}), "
        f"2^{wide_floor_log:.4f} bit operations, "
        f"{_power_of_two_bits(wide_floor_state)} active state."
    )
    phase_rows: list[tuple[object, ...]] = []
    for anchors, k, d, width, solve in (
        (8, 215, 8, 8, "nested"),
        (9, 208, 8, 16, "order_basis"),
        (8, 215, 8, 32, "nested"),
        (9, 208, 8, 32, "order_basis"),
        (8, 215, 8, 64, "dense"),
        (8, 215, 8, 64, "nested"),
        (9, 208, 8, 64, "order_basis"),
        (7, 258, 9, 32, "nested"),
    ):
        plain = price(
            target, k, d, block_width=width, normalization="projective",
            solve=solve,
        )
        omitted = price(
            target, k, d, block_width=width, normalization="projective",
            solve=solve, phase_alignment=True,
        )
        charged = price(
            target, k, d, block_width=width, normalization="projective",
            solve=solve, phase_alignment=True, charge_pair_alignment=True,
        )
        assert plain is not None and omitted is not None
        assert plain.cell.anchors == anchors
        pair = "--"
        charged_total = "--"
        state = omitted.state_bytes
        if charged is not None:
            assert charged.alignment_relation_generation_log2 is not None
            pair = f"{charged.alignment_relation_generation_log2:.2f}"
            charged_total = f"{charged.total_log2:.2f}"
            state = charged.state_bytes
        phase_rows.append((
            f"({k},{d},{anchors})", width, solve,
            f"{plain.total_log2:.2f}", f"{omitted.total_log2:.2f}", pair,
            charged_total, _power_of_two_bits(state),
        ))
    _print_table(
        "Frobenius-phase sensitivity (cross-pairing is not constructed)",
        (
            "(k_l,d,c)", "b", "solve", "no sync", "uncharged",
            "pair generation", "charged", "state",
        ),
        phase_rows,
    )
    print("  order_basis rows are illustrative leading-monomial models.")

    lever_rows: list[tuple[object, ...]] = []
    base = price(
        target, 215, 8, block_width=32, organization="singleton",
        normalization="projective", solve="nested",
    )
    assert base is not None
    for held in range(1, 7):
        organization = "singleton" if held == 1 else "cover"
        got = price(
            target, 215, 8, block_width=32, organization=organization,
            held_per_kernel=None if held == 1 else held,
            normalization="projective", solve="nested",
        )
        assert got is not None
        lever_rows.append((
            held, ceil(got.cell.anchors / held), f"{got.cell.kernel_floor:,}",
            f"{got.relation_generation_log2:.2f}",
            f"{got.relation_generation_log2 - base.relation_generation_log2:+.2f}",
            _power_of_two_bits(got.state_bytes),
        ))
    _print_table(
        "Exact multi-position hold-out cover arithmetic "
        "(richness for higher-order derivative recovery remains assumed)",
        ("held/kernel", "kernels", "kernel floor", "relation generation", "delta", "state"),
        lever_rows,
    )

    margin_rows: list[tuple[object, ...]] = []
    margin_base = holdout_cell(target, 215, 8, 1, 1)
    assert margin_base is not None
    margin_base_work, _ = relation_generation_price(margin_base)
    for slack in (1, 42):
        cell = holdout_cell(target, 215, 8, 1, slack)
        assert cell is not None
        work, _ = relation_generation_price(cell)
        margin_rows.append((
            slack, cell.schedule.text, f"{cell.kernel_floor:,}",
            f"{work:.2f}", f"{work - margin_base_work:+.3f}",
        ))
    _print_table(
        "Hermite-margin sensitivity at (k_l,d)=(215,8)",
        ("slack", "orders", "kernel floor", "relation generation", "delta"),
        margin_rows,
    )

    gauge_cell = holdout_cell(target, 215, 8, 1)
    assert gauge_cell is not None
    weight_rows = []
    for weight in (60, 75, 90, 107, 120, 140):
        nonzeros = nonzeros_at_weight(gauge_cell, weight)
        weight_rows.append((
            weight, f"{nonzeros:,}", f"{log2(nonzeros / gauge_cell.nonzeros):+.2f}",
        ))
    _print_table(
        "Modeled non-pivot-weight sensitivity at (k_l,d)=(215,8)",
        ("weight", "nonzeros", "relation-generation delta"),
        weight_rows,
    )

    local_rows = []
    for name, models in selected.items():
        cell = models["dense"].cell
        local_rows.append((
            name,
            f"{log2(replay_bit_upper(cell)):.2f}",
            f"{log2(certification_bit_envelope(cell)):.2f}",
            f"{log2(keycheck_leading_bit_count(TARGETS[name])):.2f}",
        ))
    _print_table(
        "Replay/certification included; key check shown separately",
        ("target", "replay bound", "certify envelope", "key-check leading"),
        local_rows,
    )

    c7 = price(target, 258, 9)
    assert c7 is not None
    print("\nSensitivity and continuation bound")
    print(
        "  c=7 dense row-echelon subtotal under the assumed 250--300-bit-operation "
        f"multiplier range: 2^{dense_elimination_sensitivity_log2(c7, 250):.2f}"
        f"--2^{dense_elimination_sensitivity_log2(c7, 300):.2f} bit operations."
    )
    print("  The multiply/add count is resolved; no basis or multiplier circuit is supplied.")
    print(
        "  C_cont <= G*C_ext in a no-success traversal.  C_ext covers one "
        "fail-closed public-column extension, full matching, shortening "
        "restoration, Goppa completion, and exact verification."
    )
    print(
        f"  For affine c=7 the deliberately coarse deterministic ceiling is "
        f"log2(C_ext)<={log2(continuation_bitops_upper(c7.cell)):.2f} and "
        f"log2(G*C_ext)<={continuation_no_success_upper_log2(c7):.2f}."
    )
    print("  This finite ceiling is not a sharp survivor-dependent attack estimate.")

    print("\nSearch interface")
    print("  --scan TARGET          repeat an exhaustive singleton search")
    print("  --objective relation|affine-dense|pgl-nested")
    print("  --wide                 use one practical multiword relation batch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan", choices=tuple(TARGETS),
        help="exhaustively scan one target",
    )
    parser.add_argument(
        "--objective", choices=("relation", "affine-dense", "pgl-nested"),
        default="relation", help="quantity minimized by --scan",
    )
    parser.add_argument(
        "--wide", action="store_true",
        help="use the smallest multiple-of-64 width giving one relation batch",
    )
    parser.add_argument("--maximum-degree", type=int, default=24)
    args = parser.parse_args()
    internal_checks()
    if args.scan:
        target = TARGETS[args.scan]
        if args.objective == "relation":
            if args.wide:
                admitted, best, width, work = practical_wide_relation_generation_floor(
                    target, args.maximum_degree,
                )
            else:
                admitted, best, work = one_position_relation_generation_floor(
                    target, args.maximum_degree,
                )
                width = 64
            _, state = relation_generation_price(
                best, block_width=width, organization="singleton",
            )
            total = work
        else:
            normalization, solve = {
                "affine-dense": ("affine", "dense"),
                "pgl-nested": ("projective", "nested"),
            }[args.objective]
            admitted, priced = jointly_optimized_price(
                target, args.maximum_degree, normalization, solve, args.wide,
            )
            best = priced.cell
            width = priced.block_width
            work = priced.relation_generation_log2
            total = priced.total_log2
            state = priced.state_bytes
        print(f"target: {target.name}")
        print(f"admitted cells: {admitted}")
        print(
            f"cell: k_l={best.k}, d={best.d}, h={best.h}, c={best.anchors}, "
            f"b={width}"
        )
        print(f"schedule: {best.schedule.text}")
        print(f"ambient: {best.N}")
        print(f"row-rank bound: {best.row_rank_bound}")
        print(f"kernel floor: {best.kernel_floor}")
        print(f"relation generation: 2^{work:.6f} bit operations")
        if args.objective != "relation":
            print(f"accounted subtotal: 2^{total:.6f} bit operations")
        print(f"active retained state: 2^{log2(8 * state):.4f} bits")
    else:
        print_report()


if __name__ == "__main__":
    main()
