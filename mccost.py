#!/usr/bin/env python3
"""Self-contained arithmetic and bit-cost model for Classic McEliece.

This file is deliberately self-contained: it imports only Python's standard
library, opens no challenge input, and writes no files.  It repeats the exact
integer cell arithmetic, the sparse-supplier gate tally, and the declared
downstream cost models used in the manuscript.  It is not an implementation of
the attack.  In particular it does not prove reliable binary Krylov yield,
Vedenev's higher-flag conjecture for the priced truncated relation block, the
binary reconstruction, cross-anchor independence, common-kernel richness, PGL
branch bookkeeping, or the public pure cross-pairing needed for
Frobenius-phase synchronization.

Typical use::

    python3 mccost.py                         # report and internal checks
    python3 mccost.py --scan mceliece348864   # exhaustive singleton scan

Cost units follow the manuscript.  A 64-bit XOR word operation is 64 Boolean
gates, and a GF(2^m) multiplication is represented by the optimistic m^2
AND-gate proxy; XORs in the multiplier are omitted.  The model does not charge
random access, addressing, communication, or storage.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from math import ceil, comb, log2


NIST_GATES = {1: 143.0, 2: 146.0, 3: 207.0, 4: 210.0, 5: 272.0}
PIB = 1 << 50
EIB = 1 << 60


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
    projection_weight: int
    weight_threshold: int
    N: int
    operator_rows: int
    row_rank_bound: int
    kernel_floor: int
    nonzeros: int
    anchors: int
    relation_minimum: int

    @property
    def augmented_vector_width64_bytes(self) -> int:
        """Panel-only width-64 bytes; use ``solver_state_bytes`` for totals."""
        return 8 * (self.N + self.operator_rows)

    @property
    def augmented_one_batch_word_operations(self) -> Fraction:
        """One nominal width-64 Coppersmith attempt, in word XORs.

        The matrix passed to the solver is the augmented matrix, whose rank is
        bounded by twice the row-rank bound of M.  A balanced equal-block
        attempt is charged three block applications per rank/block unit, and
        every augmented application traverses both M and M^T.
        """
        calls = 3 * ceil(2 * self.row_rank_bound / 64)
        return Fraction(calls * 2 * 64 * self.nonzeros, 64)


@dataclass(frozen=True)
class Price:
    cell: Cell
    block_width: int
    organization: str
    normalization: str
    solve: str
    phase_alignment: bool
    supplier_log2: float
    guess_log2: float
    per_guess_field_ops_log2: float
    per_guess_gates_log2: float
    downstream_log2: float
    higher_flag_log2: float
    higher_flag_retries: int
    alignment_supplier_log2: float | None
    alignment_cost_charged: bool
    pairing_construction_assumed: bool
    total_log2: float
    vector_state_bytes: int
    sequence_state_bytes: int
    relation_state_bytes: int
    state_bytes: int
    conditional_subtotal: bool

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


def holdout_cell(
    target: Target, k: int, d: int, h: int = 1, hermite_margin: int = 1,
) -> Cell | None:
    """Return one dimension-guaranteed projected-weight cell, or ``None``.

    The row-rank bound is a sum of point-block ranks and may exceed the true
    global rank.  Hence ``N-row_rank_bound`` is a rigorous kernel lower bound.
    The nonzero count is exact only for the declared projection weight k//2.
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
    U = k * (D_l + 1)
    anchors = ceil((U - 1) / comb(k, 2))
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


def vector_state_bytes(cell: Cell, block_width: int) -> int:
    """One augmented-domain block over both sides of the operator."""
    return (cell.N + cell.operator_rows) * block_width // 8


def augmented_rank_bound(cell: Cell) -> int:
    """Rank bound for the actual matrix passed to the Krylov solver."""
    return 2 * cell.row_rank_bound


def coppersmith_applications(cell: Cell, block_width: int) -> int:
    """Nominal augmented applications in one balanced equal-block attempt."""
    _require_positive("block_width", block_width)
    return 3 * ceil(augmented_rank_bound(cell) / block_width)


def sequence_length(cell: Cell, block_width: int) -> int:
    """Safe rank-bound precision for the augmented b-by-b sequence."""
    return 2 * ceil(augmented_rank_bound(cell) / block_width) + block_width


def sequence_state_bytes(cell: Cell, block_width: int) -> int:
    """Retained scalar-sequence floor; recursive PM-Basis scratch is extra."""
    return sequence_length(cell, block_width) * block_width * block_width // 8


def solver_state_bytes(cell: Cell, block_width: int) -> tuple[int, int, int]:
    """Return vector-panel, retained-sequence, and summed state bytes."""
    vector = vector_state_bytes(cell, block_width)
    sequence = sequence_state_bytes(cell, block_width)
    return vector, sequence, vector + sequence


def relation_state_bytes(cell: Cell, relation_count: int | None = None) -> int:
    """Packed coefficient storage for one active certified relation block."""
    count = cell.relation_minimum if relation_count is None else _require_positive(
        "relation_count", relation_count,
    )
    return ceil(count * cell.N / 8)


def higher_flag_field_ops_upper(cell: Cell, retries: int = 1) -> int:
    """Loose dense upper bound for all sequential higher-flag systems.

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


def higher_flag_gates_log2(cell: Cell, retries: int = 1) -> float:
    """Convert the dense higher-flag field-operation bound to gate units."""
    return log2(higher_flag_field_ops_upper(cell, retries) * cell.m**2)


def supplier_price(
    cell: Cell,
    *,
    block_width: int = 64,
    organization: str = "singleton",
    solver_attempts: int = 1,
) -> tuple[float, int]:
    """Return nominal sequential full-word Coppersmith work and state.

    One width-``b`` recovery yields at most ``b`` relations, so the relation
    block needs ``ceil(r_min/b)`` sequential recoveries.  Each recovery retains
    width-``b`` state but charges every sparse incidence as one full 64-bit
    word.  ``solver_attempts`` is the attempts-per-batch multiplier.  No theorem
    bounds its required value for this structured matrix over F_2; displayed
    rows use one verified-or-refused attempt for every required batch.
    """
    if organization not in {"singleton", "cover", "common"}:
        raise ValueError("organization must be singleton, cover, or common")
    if block_width not in {64, 32, 16, 8, 4, 2, 1}:
        raise ValueError("block_width must divide 64")
    solver_attempts = _require_positive("solver_attempts", solver_attempts)
    kernels = {
        "singleton": cell.anchors,
        "cover": ceil(cell.anchors / cell.h),
        "common": 1,
    }[organization]
    batches = ceil(cell.relation_minimum / block_width)
    gates_integer = (
        coppersmith_applications(cell, block_width)
        * 2 * 64 * cell.nonzeros
        * kernels * batches * solver_attempts
    )
    gates = log2(gates_integer)
    _, _, state = solver_state_bytes(cell, block_width)
    return gates, state


def pair_alignment_supplier_price(
    target: Target,
    k: int,
    d: int,
    anchors: int,
    *,
    block_width: int = 64,
    relation_count: int | None = None,
    solver_attempts: int = 1,
    hermite_margin: int = 1,
) -> tuple[float, int, Cell] | None:
    """Price ``anchors-1`` two-position suppliers on a spanning tree.

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
    required = pair_cell.relation_minimum if relation_count is None else _require_positive(
        "relation_count", relation_count,
    )
    batches = ceil(required / block_width)
    gates_integer = (
        coppersmith_applications(pair_cell, block_width)
        * 2 * 64 * pair_cell.nonzeros
        * batches * (anchors - 1) * _require_positive("solver_attempts", solver_attempts)
    )
    gates = log2(gates_integer)
    _, _, state = solver_state_bytes(pair_cell, block_width)
    return gates, state, pair_cell


def replay_bit_upper(cell: Cell, organization: str = "singleton") -> int:
    """Fresh-annihilation plus naive-independence bit bound."""
    kernels = {
        "singleton": cell.anchors,
        "cover": ceil(cell.anchors / cell.h),
        "common": 1,
    }[organization]
    batches = ceil(cell.relation_minimum / 64)
    return kernels * (64 * batches * cell.nonzeros + cell.relation_minimum**2 * cell.N)


def certification_bit_upper(cell: Cell) -> int:
    """Loose full-coefficient value/gradient/quadratic certification bound."""
    r = cell.relation_minimum
    return (
        r * cell.N * (1 + cell.d + comb(cell.d, 2))
        + r * cell.k**2
        + r * comb(cell.m, 2) ** 2
    )


def keycheck_leading_bit_count(target: Target) -> int:
    """Leading parity-check/public-generator multiplication tally.

    This is the explicit ``2*m*t*n*k`` term.  It omits the lower-order rank
    check, so it is deliberately not named or reported as a strict upper bound.
    """
    k = target.n - target.m * target.t
    return 2 * target.m * target.t * target.n * k


def per_guess_field_ops(cell: Cell, solve: str, omega: float = 3.0) -> float:
    """Base-two logarithm of GF(2^m) operations for one Step-3 solve."""
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
    higher_flag_retries: int = 1,
) -> Price | None:
    """Price one declared model.

    ``charge_pair_alignment`` adds ``c-1`` two-position Holdout suppliers.  A
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
    supplier, state = supplier_price(
        cell,
        block_width=block_width,
        organization=organization,
        solver_attempts=solver_attempts,
    )
    vector_state, sequence_state, _ = solver_state_bytes(cell, block_width)
    relation_state = relation_state_bytes(cell)
    guesses = guess_count(cell, normalization, phase_alignment)
    field_ops = per_guess_field_ops(cell, solve)
    per_guess_gates = field_ops + log2(cell.m * cell.m)
    downstream = guesses + per_guess_gates
    higher_flag = higher_flag_gates_log2(cell, higher_flag_retries)
    alignment_supplier = None
    if charge_pair_alignment:
        alignment = pair_alignment_supplier_price(
            target, k, d, cell.anchors,
            block_width=block_width,
            relation_count=pair_relation_count,
            solver_attempts=solver_attempts,
            hermite_margin=hermite_margin,
        )
        if alignment is None:
            return None
        alignment_supplier, alignment_state, _ = alignment
        if alignment_state > state:
            state = alignment_state
            pair_cell = alignment[2]
            vector_state, sequence_state, _ = solver_state_bytes(pair_cell, block_width)
            required = pair_cell.relation_minimum if pair_relation_count is None else pair_relation_count
            relation_state = relation_state_bytes(pair_cell, required)
    terms = [supplier, downstream, higher_flag]
    if alignment_supplier is not None:
        terms.append(alignment_supplier)
    total = _logsum2(*terms)
    return Price(
        cell=cell,
        block_width=block_width,
        organization=organization,
        normalization=normalization,
        solve=solve,
        phase_alignment=phase_alignment,
        supplier_log2=supplier,
        guess_log2=guesses,
        per_guess_field_ops_log2=field_ops,
        per_guess_gates_log2=per_guess_gates,
        downstream_log2=downstream,
        higher_flag_log2=higher_flag,
        higher_flag_retries=higher_flag_retries,
        alignment_supplier_log2=alignment_supplier,
        alignment_cost_charged=charge_pair_alignment,
        pairing_construction_assumed=phase_alignment,
        total_log2=total,
        vector_state_bytes=vector_state,
        sequence_state_bytes=sequence_state,
        relation_state_bytes=relation_state,
        state_bytes=state,
        # Every row is conditional on binary solver yield and higher-flag
        # richness; the following options add further explicit assumptions.
        conditional_subtotal=True,
    )


LADDER = ((9, 208, 8), (8, 215, 8), (7, 258, 9), (6, 322, 9), (5, 429, 10))


@lru_cache(maxsize=None)
def singleton_supplier_floor(
    target: Target, maximum_degree: int = 24,
) -> tuple[int, Cell, float]:
    """Count admitted h=1 cells and minimize their width-64 supplier work."""
    admitted = 0
    best_cell: Cell | None = None
    best_log = float("inf")
    for d in range(3, maximum_degree + 1):
        for k in range(d + 2, target.n - target.m * target.t + 1):
            cell = holdout_cell(target, k, d, 1)
            if cell is None:
                continue
            admitted += 1
            work, _ = supplier_price(cell, block_width=64, organization="singleton")
            if (work, k, d) < (best_log, best_cell.k if best_cell else 0, best_cell.d if best_cell else 0):
                best_cell, best_log = cell, work
    if best_cell is None:
        raise AssertionError("declared supplier-floor scan found no cell")
    return admitted, best_cell, best_log


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
def memory_cap_rows(maximum_degree: int = 24) -> tuple[tuple[str, Price], ...]:
    """Exhaustively minimize the nested/projective lane under four state caps."""
    target = TARGETS["mceliece348864"]
    candidates: list[Price] = []
    for d in range(3, maximum_degree + 1):
        for k in range(d + 2, target.n - target.m * target.t + 1):
            for width in (64, 32, 16, 8, 4, 2, 1):
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
        rows.append((
            label,
            min(
                fitting,
                key=lambda item: (
                    item.total_log2, item.state_bytes, item.cell.k, item.cell.d,
                    item.block_width, item.organization,
                ),
            ),
        ))
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
    assert NIST_GATES == {1: 143.0, 2: 146.0, 3: 207.0, 4: 210.0, 5: 272.0}
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
        "mceliece348864": (114.35, 172.13, 146.29),
        "mceliece460896": (121.49, 200.03, 176.77),
        "mceliece6688128": (125.11, 235.09, 207.83),
        "mceliece6960119": (124.10, 217.96, 194.35),
        "mceliece8192128": (125.11, 235.09, 207.83),
    }
    for name, models in selected_prices().items():
        got = (models["dense"].supplier_log2, models["dense"].total_log2, models["nested"].total_log2)
        for observed, wanted in zip(got, expected[name]):
            assert abs(observed - wanted) < 0.006, (name, observed, wanted)
        cell = models["dense"].cell
        uniform_hermite_order = ceil((cell.order_target + 1) / cell.constrained_columns)
        uniform_dimension_order = max(
            order for order in range(1, cell.d)
            if cell.constrained_columns * point_block_rank(
                cell.k, cell.projection_weight, cell.d, order,
            ) < cell.N
        )
        assert uniform_dimension_order + 1 == uniform_hermite_order
    target = TARGETS["mceliece348864"]
    c7 = price(target, 258, 9)
    assert c7 is not None
    assert c7.cell.anchors == 7 and abs(c7.total_log2 - 142.14523524919355) < 1e-12
    assert c7.cell.schedule.order_total == c7.cell.order_target + 1
    # Exact augmented identity is algebraic; here we pin one Coppersmith attempt.
    one_batch_gates = c7.cell.augmented_one_batch_word_operations * 64
    assert one_batch_gates == (
        coppersmith_applications(c7.cell, 64) * 2 * 64 * c7.cell.nonzeros
    )
    c9 = price(target, 208, 8, block_width=32)
    c8 = price(target, 215, 8, block_width=32)
    assert c9 is not None and c8 is not None
    assert c9.state_bytes == 1_810_356_471_721_316
    assert c8.state_bytes == 2_256_823_629_599_764
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
        ("53.00", 256, 8, 4, "singleton", 127.6957124229015, 911_560_528_788_311),
        ("56.32", 256, 8, 32, "singleton", 126.8913699711271, 7_292_484_230_310_524),
        ("59.64", 256, 8, 64, "singleton", 126.8781571063955, 14_584_968_460_645_624),
        ("63.00", 258, 9, 64, "common", 126.12982587262806, 478_189_145_871_160_720),
    )
    for (label, got), expected_cap in zip(cap_rows, cap_expected):
        wanted_label, k, d, width, organization, total, state = expected_cap
        assert label == wanted_label
        assert (got.cell.k, got.cell.d, got.block_width, got.organization) == (
            k, d, width, organization,
        )
        assert abs(got.total_log2 - total) < 1e-12
        assert got.state_bytes == state
    admitted, floor_cell, floor_log = singleton_supplier_floor(target)
    assert admitted == 19_338
    assert (floor_cell.k, floor_cell.d, floor_cell.anchors) == (208, 8, 9)
    assert abs(floor_log - 114.35333862656461) < 1e-12
    margin42 = holdout_cell(target, 215, 8, 1, 42)
    assert margin42 is not None
    assert margin42.schedule.text == "767@7+0@6"
    assert margin42.kernel_floor == 1_187_277_582_725
    assert holdout_cell(target, 215, 8, 1, 43) is None
    cover4 = price(target, 215, 8, block_width=32, organization="cover", held_per_kernel=4)
    assert cover4 is not None and cover4.cell.kernel_floor == 4_055_009_590_230
    assert abs(cover4.supplier_log2 - c8.supplier_log2 + 1.97119) < 0.001
    gauge = holdout_cell(target, 215, 8, 1)
    assert gauge is not None
    assert abs(log2(nonzeros_at_weight(gauge, 60) / gauge.nonzeros) + 2.149284) < 1e-6
    charged_expected = {
        (215, 8, 64): (115.8085, 0.8772),
        (215, 8, 32): (117.6278, 0.9016),
        (215, 8, 16): (119.5371, 0.9088),
        (258, 9, 32): (131.6980, 0.8986),
    }
    for (k, d, width), (wanted_total, wanted_increment) in charged_expected.items():
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
        assert charged.conditional_subtotal
        assert abs(charged.total_log2 - wanted_total) < 0.001
        assert abs(charged.total_log2 - omitted.total_log2 - wanted_increment) < 0.001


def print_report() -> None:
    selected = selected_prices()
    print("MCCOST — self-contained Classic McEliece cost model")
    print("All exponents are base two. Work is in classical Boolean gates.")
    print("Sparse work charges 64 gates per 64-bit XOR word; GF(2^m)")
    print("multiplication uses the optimistic m^2 AND-gate proxy and omits XORs.")
    print("Memory is reported as log2 bits; an entry 53 means 2^53 bits.")
    print("The supplier uses one nominal equal-block Coppersmith attempt per")
    print("required relation batch on A=[[0,M^T],[M,0]]; retries, access,")
    print("recursive basis scratch are not charged. Every total is conditional.")

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
            f"{dense.supplier_log2:.2f}",
            f"{dense.total_log2:.2f}",
            f"{nested.total_log2:.2f}",
            _power_of_two_bits(dense.state_bytes),
        ))
    _print_table(
        "Selected cells and paper headline totals",
        (
            "target", "cat", "(k_l,d,h)", "orders", "c", "log N",
            "supplier", "affine/dense", "PGL/nested", "state",
        ),
        target_rows,
    )

    print("\nExact selected-cell arithmetic")
    for target in TARGETS.values():
        got = selected[target.name]["dense"]
        cell = got.cell
        hermite_left = cell.k * (cell.d - 1) + cell.schedule.order_total
        hermite_right = cell.d * cell.D_l
        batches = ceil(cell.relation_minimum / 64)
        applications = coppersmith_applications(cell, got.block_width)
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
            f"{cell.high_levels}/{cell.low_levels}; projection weight {cell.projection_weight}"
        )
        print(
            f"    dimensions: N={cell.N:,}; rows={cell.operator_rows:,}; "
            f"rank(M)<={cell.row_rank_bound:,}; dim ker(M)>={cell.kernel_floor:,}"
        )
        print(
            f"    sparse operator: nnz={cell.nonzeros:,}; rank(A)<="
            f"{augmented_rank_bound(cell):,}; applications={applications:,}"
        )
        print(
            f"    relation block: r_min={cell.relation_minimum}, batches={batches}; "
            f"kernels={cell.anchors}; supplier=2^{got.supplier_log2:.4f} gates"
        )
        print(
            f"    retained state: vector={_power_of_two_bits(got.vector_state_bytes)}, "
            f"sequence={_power_of_two_bits(got.sequence_state_bytes)}, "
            f"sum={_power_of_two_bits(got.state_bytes)}; active relations="
            f"{_power_of_two_bits(got.relation_state_bytes)}"
        )

    target = TARGETS["mceliece348864"]
    ladder_rows: list[tuple[object, ...]] = []
    for anchors, k, d in LADDER:
        dense = price(target, k, d)
        nested = price(target, k, d, normalization="projective", solve="nested")
        assert dense is not None and nested is not None and dense.cell.anchors == anchors
        ladder_rows.append((
            anchors, f"({k},{d})", f"{dense.supplier_log2:.2f}",
            f"{dense.guess_log2:.2f}", f"{dense.total_log2:.2f}",
            f"{nested.total_log2:.2f}", _power_of_two_bits(dense.state_bytes),
        ))
    _print_table(
        "mceliece348864 anchor ladder",
        ("c", "(k_l,d)", "supplier", "affine guesses", "dense total", "nested total", "state"),
        ladder_rows,
    )

    cap_rows = [
        (
            label,
            f"({got.cell.anchors},{got.cell.k},{got.cell.d})",
            got.block_width,
            got.organization,
            f"{got.total_log2:.2f}",
            _power_of_two_bits(got.state_bytes),
        )
        for label, got in memory_cap_rows()
    ]
    _print_table(
        "Exhaustive mceliece348864 nested/projective optima under state caps",
        ("cap", "(c,k_l,d)", "b", "organization", "total", "state"),
        cap_rows,
    )

    admitted, floor_cell, floor_log = singleton_supplier_floor(target)
    _, _, floor_state = solver_state_bytes(floor_cell, 64)
    print(
        f"\nSingleton supplier floor: scanned {admitted:,} admitted h=1 cells "
        f"with 3<=d<=24; optimum (k_l,d,c,b)=({floor_cell.k},{floor_cell.d},"
        f"{floor_cell.anchors},64), 2^{floor_log:.4f} gates, "
        f"{_power_of_two_bits(floor_state)} retained state."
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
    ):
        got = price(
            target, k, d, block_width=width, normalization="projective",
            solve=solve, phase_alignment=True,
        )
        assert got is not None and got.cell.anchors == anchors
        phase_rows.append((
            anchors, k, width, solve, f"{got.guess_log2:.2f}",
            f"{got.downstream_log2:.2f}", f"{got.supplier_log2:.2f}",
            f"{got.total_log2:.2f}", _power_of_two_bits(got.state_bytes),
        ))
    _print_table(
        "Conditional phase-aligned subtotals (cross-pairing is not constructed)",
        ("c", "k_l", "b", "solve", "guesses", "downstream", "supplier", "subtotal", "state"),
        phase_rows,
    )
    print("  order_basis rows are illustrative leading-monomial models.")

    phase_charge_rows: list[tuple[object, ...]] = []
    for anchors, k, d, width in (
        (8, 215, 8, 64), (8, 215, 8, 32), (8, 215, 8, 16), (7, 258, 9, 32),
    ):
        plain = price(
            target, k, d, block_width=width, normalization="projective", solve="nested",
        )
        omitted = price(
            target, k, d, block_width=width, normalization="projective", solve="nested",
            phase_alignment=True,
        )
        pair_cell = holdout_cell(target, k, d, 2)
        assert pair_cell is not None
        charged = price(
            target, k, d, block_width=width, normalization="projective", solve="nested",
            phase_alignment=True, charge_pair_alignment=True,
            pair_relation_count=pair_cell.relation_minimum,
        )
        assert plain is not None and omitted is not None and charged is not None
        assert charged.alignment_supplier_log2 is not None
        phase_charge_rows.append((
            anchors, k, width, f"{plain.total_log2:.2f}", f"{omitted.total_log2:.2f}",
            f"{charged.alignment_supplier_log2:.2f}", f"{charged.total_log2:.2f}",
            f"{charged.total_log2 - omitted.total_log2:.2f}",
            _power_of_two_bits(charged.state_bytes),
        ))
    _print_table(
        "Conditional cost of proposed pair suppliers",
        ("c", "k_l", "b", "no align", "align omitted", "pair supply", "charged", "increment", "state"),
        phase_charge_rows,
    )

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
            f"{got.supplier_log2:.2f}", f"{got.supplier_log2 - base.supplier_log2:+.2f}",
            _power_of_two_bits(got.state_bytes),
        ))
    _print_table(
        "Exact multi-holdout cover arithmetic (flag richness remains assumed)",
        ("held/kernel", "kernels", "kernel floor", "supplier", "delta", "state"),
        lever_rows,
    )

    margin_rows: list[tuple[object, ...]] = []
    margin_base = holdout_cell(target, 215, 8, 1, 1)
    assert margin_base is not None
    margin_base_work, _ = supplier_price(margin_base)
    for slack in (1, 42):
        cell = holdout_cell(target, 215, 8, 1, slack)
        assert cell is not None
        work, _ = supplier_price(cell)
        margin_rows.append((
            slack, cell.schedule.text, f"{cell.kernel_floor:,}",
            f"{work:.2f}", f"{work - margin_base_work:+.3f}",
        ))
    _print_table(
        "Hermite-margin sensitivity at (k_l,d)=(215,8)",
        ("slack", "orders", "kernel floor", "supplier", "delta"),
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
        "Projection-weight sensitivity at (k_l,d)=(215,8)",
        ("weight", "nonzeros", "supplier delta"),
        weight_rows,
    )

    local_rows = []
    for name, models in selected.items():
        cell = models["dense"].cell
        local_rows.append((
            name,
            f"{log2(replay_bit_upper(cell)):.2f}",
            f"{log2(certification_bit_upper(cell)):.2f}",
            f"{log2(keycheck_leading_bit_count(TARGETS[name])):.2f}",
        ))
    _print_table(
        "Lower-order bit-operation tallies at selected cells",
        ("target", "replay upper", "certify upper", "key-check leading"),
        local_rows,
    )

    print("\nSearch interface")
    print("  --scan TARGET          repeat the exhaustive width-64 singleton search")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan", choices=tuple(TARGETS),
        help="exhaustively minimize the complete width-64 singleton supplier cost",
    )
    parser.add_argument("--maximum-degree", type=int, default=24)
    args = parser.parse_args()
    internal_checks()
    if args.scan:
        target = TARGETS[args.scan]
        admitted, best, work = singleton_supplier_floor(
            target, args.maximum_degree,
        )
        _, state = supplier_price(
            best, block_width=64, organization="singleton",
        )
        print(f"target: {target.name}")
        print(f"admitted cells: {admitted}")
        print(f"cell: k_l={best.k}, d={best.d}, h={best.h}, c={best.anchors}")
        print(f"schedule: {best.schedule.text}")
        print(f"ambient: {best.N}")
        print(f"row-rank bound: {best.row_rank_bound}")
        print(f"kernel floor: {best.kernel_floor}")
        print(f"supplier gates: 2^{work:.6f}")
        print(f"retained state: 2^{log2(8 * state):.4f} bits")
    else:
        print_report()


if __name__ == "__main__":
    main()
