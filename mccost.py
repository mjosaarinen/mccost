#!/usr/bin/env python3
"""Reproduce the current Classic McEliece bit-operation estimates.

This is the publication artifact for

    Bit Operation Cost of ``Holdout'' Key-Recovery Attacks Against
    Classic McEliece

It implements one route only: the one-chart direct locator-recovery method
reported in the living revision dated 30 August 2026. It evaluates that route
for every standardized Classic McEliece parameter set and contains neither
superseded cost models nor exploratory parameter searches.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, comb, log2
import sys


HOLDOUT_MULTIPLICITY = 5
HOLDOUTS = 4


@dataclass(frozen=True)
class Cell:
    name: str
    extension_degree: int
    public_length: int
    goppa_degree: int
    public_dimension: int
    shortening: int
    relation_degree: int
    multiplicity_profile: tuple[tuple[int, int], ...]
    claimed_category: int
    category_bit_threshold: int

    @property
    def field_order(self) -> int:
        return 1 << self.extension_degree

    @property
    def shortened_length(self) -> int:
        return self.public_length - self.shortening

    @property
    def shortened_dimension(self) -> int:
        return self.public_dimension - self.shortening


CELLS = (
    Cell(
        "mceliece348864", 12, 3488, 64, 2720, 2569, 7,
        ((5, 42), (6, 877)), 1, 143,
    ),
    Cell(
        "mceliece460896", 13, 4608, 96, 3360, 3159, 8,
        ((5, 4), (6, 177), (7, 1268)), 3, 207,
    ),
    Cell(
        "mceliece6688128", 13, 6688, 128, 5024, 4802, 7,
        ((5, 35), (6, 1851)), 5, 272,
    ),
    Cell(
        "mceliece6960119", 13, 6960, 119, 5413, 5198, 7,
        ((5, 24), (6, 1738)), 5, 272,
    ),
    Cell(
        "mceliece8192128", 13, 8192, 128, 6528, 6306, 7,
        ((5, 35), (6, 1851)), 5, 272,
    ),
)


@dataclass(frozen=True)
class Shape:
    cell: Cell
    jet_columns_per_copy: int
    monomial_rows_per_copy: int
    holdout_block_length: int
    largest_locator_block: int
    unknowns: int
    equations: int
    binary_jet_nonzeros: int
    generic_auxiliary_nonzeros: int
    candidate_systems: int
    selected_parity_rank: int
    required_label_count: int


@dataclass(frozen=True)
class Estimate:
    shape: Shape
    solver_field_degree: int
    field_multiply_bit_estimate: int
    field_multiply_add_bit_estimate: int
    normal_product_bit_estimate: int
    per_system_bit_estimate: int
    all_systems_bit_estimate: int
    label_selection_bit_estimate: int
    finisher_bit_estimate: int
    total_bit_estimate: int
    total_log2_bit_estimate: float
    aggregate_solver_failure_log2: float
    active_state_bits_estimate: int
    active_state_log2_bits_estimate: float
    margin_below_category_threshold_bits: float


def field_multiply_bit_estimate(degree: int) -> int:
    """Schoolbook polynomial-basis multiplication, in elementary gates."""

    if degree <= 0:
        raise ValueError("field degree must be positive")
    return 3 * degree * degree - 3 * degree + 1


def field_multiply_add_bit_estimate(degree: int) -> int:
    """One multiplication followed by one field addition."""

    return field_multiply_bit_estimate(degree) + degree


def locator_block_size(k_l: int, multiplicity: int) -> int:
    """Number of coordinates used by one locator test block."""

    first_order = multiplicity - HOLDOUT_MULTIPLICITY + 1
    return sum(
        comb(k_l + order - 1, order)
        for order in range(max(0, first_order), multiplicity)
    )


def system_shape(cell: Cell) -> Shape:
    """Compute the fixed direct locator-recovery system for one parameter set."""

    n_l = cell.shortened_length
    k_l = cell.shortened_dimension
    if HOLDOUT_MULTIPLICITY <= 2:
        raise ValueError("holdout multiplicity must exceed two")
    locator_power = HOLDOUT_MULTIPLICITY - 1
    if locator_power & (locator_power - 1):
        raise ValueError("holdout multiplicity minus one must be a power of two")
    if cell.relation_degree <= HOLDOUT_MULTIPLICITY:
        raise ValueError("relation degree must exceed the holdout multiplicity")
    if sum(
        count
        for multiplicity, count in cell.multiplicity_profile
        if multiplicity == HOLDOUT_MULTIPLICITY
    ) < HOLDOUTS:
        raise ValueError("multiplicity profile has too few holdout coordinates")
    if cell.public_length > cell.field_order:
        raise ValueError("binary Goppa support is larger than the base field")
    if sum(count for _multiplicity, count in cell.multiplicity_profile) != n_l:
        raise ValueError("multiplicity profile length differs from the chart")
    ambient_degree = n_l - 2 * cell.goppa_degree - 1
    required_multiplicity = (
        cell.relation_degree * ambient_degree
        + 1
        - cell.goppa_degree
        + HOLDOUT_MULTIPLICITY
    )
    if sum(
        multiplicity * count
        for multiplicity, count in cell.multiplicity_profile
    ) != required_multiplicity:
        raise ValueError("multiplicity profile violates the binary-Goppa equation")

    jet_columns = sum(
        count * comb(k_l + multiplicity - 1, multiplicity - 1)
        for multiplicity, count in cell.multiplicity_profile
    )
    monomial_rows = comb(k_l + cell.relation_degree - 1, cell.relation_degree)
    holdout_block = comb(
        k_l + HOLDOUT_MULTIPLICITY - 1, HOLDOUT_MULTIPLICITY - 1
    )
    locator_block = max(
        locator_block_size(k_l, multiplicity)
        for multiplicity, _count in cell.multiplicity_profile
    )
    auxiliary_rows = HOLDOUTS * (holdout_block - 1) + 1 + locator_block
    jet_weight = sum(
        count
        * sum(comb(cell.relation_degree, order) for order in range(multiplicity))
        for multiplicity, count in cell.multiplicity_profile
    )
    parity_rank = cell.public_length - cell.public_dimension
    if parity_rank != cell.extension_degree * cell.goppa_degree:
        raise ArithmeticError("Classic parity rank differs from m*t")
    required_labels = parity_rank + 1
    label_surplus = n_l - required_labels
    if label_surplus < 0:
        raise ArithmeticError("one shortened chart has too few labels")
    return Shape(
        cell=cell,
        jet_columns_per_copy=jet_columns,
        monomial_rows_per_copy=monomial_rows,
        holdout_block_length=holdout_block,
        largest_locator_block=locator_block,
        unknowns=2 * jet_columns,
        equations=2 * monomial_rows + auxiliary_rows,
        binary_jet_nonzeros=2 * monomial_rows * jet_weight,
        generic_auxiliary_nonzeros=2 * auxiliary_rows,
        candidate_systems=(n_l + 1) * (cell.field_order + 1),
        selected_parity_rank=parity_rank,
        required_label_count=required_labels,
    )


def finisher_bit_estimate(shape: Shape) -> int:
    """Estimate known-point completion and projective recharting."""

    cell = shape.cell
    labels = shape.required_label_count
    missing_poles = cell.field_order + 1 - labels
    field_operations = missing_poles * (
        cell.field_order * labels**6
        + cell.field_order * cell.public_length * shape.selected_parity_rank**4
    )
    return field_operations * (
        2
        * cell.extension_degree
        * field_multiply_bit_estimate(cell.extension_degree)
    )


def price(cell: Cell, field_degree: int) -> Estimate:
    """Price one complete scan at a solver degree containing the base field."""

    shape = system_shape(cell)
    delta = int(field_degree)
    if delta <= 0 or delta % cell.extension_degree:
        raise ValueError(
            "solver field degree must be a positive multiple of the base degree"
        )

    multiply = field_multiply_bit_estimate(delta)
    multiply_add = field_multiply_add_bit_estimate(delta)
    normal_product = (
        2 * shape.binary_jet_nonzeros * delta
        + (
            2 * shape.generic_auxiliary_nonzeros
            + 2 * shape.unknowns
            + shape.equations
        )
        * multiply_add
    )
    recurrence = (17 * shape.unknowns + 4 * delta + 3) * multiply_add
    initialization = (20 * shape.unknowns + 4 * delta + 3) * multiply_add
    base_multiply_add = field_multiply_add_bit_estimate(cell.extension_degree)
    replay = (
        shape.binary_jet_nonzeros * cell.extension_degree
        + shape.generic_auxiliary_nonzeros * base_multiply_add
        + shape.equations * cell.extension_degree
    )
    source_and_retraction = (
        2 * shape.unknowns
        + shape.equations
        + shape.unknowns * cell.extension_degree
    ) * delta
    rank_bound = shape.equations
    per_system = (
        (rank_bound + 3) * normal_product
        + rank_bound * recurrence
        + initialization
        + replay
        + source_and_retraction
    )
    all_systems = shape.candidate_systems * per_system
    label_selection = cell.shortened_length * (
        shape.selected_parity_rank**2 + shape.selected_parity_rank
    )
    finisher = finisher_bit_estimate(shape)
    total = all_systems + label_selection + finisher

    solver_field_order = 1 << delta
    precondition_failure = (
        11 * shape.unknowns**2 - shape.unknowns
    ) / (2.0 * (solver_field_order - 1))
    lanczos_failure = rank_bound * (rank_bound + 1) / solver_field_order
    aggregate_failure = shape.candidate_systems * (
        precondition_failure + lanczos_failure
    )
    state = (
        (8 * shape.unknowns + 2 * shape.equations) * delta
        + shape.generic_auxiliary_nonzeros * delta
    )
    total_log2 = log2(total)
    return Estimate(
        shape=shape,
        solver_field_degree=delta,
        field_multiply_bit_estimate=multiply,
        field_multiply_add_bit_estimate=multiply_add,
        normal_product_bit_estimate=normal_product,
        per_system_bit_estimate=per_system,
        all_systems_bit_estimate=all_systems,
        label_selection_bit_estimate=label_selection,
        finisher_bit_estimate=finisher,
        total_bit_estimate=total,
        total_log2_bit_estimate=total_log2,
        aggregate_solver_failure_log2=log2(aggregate_failure),
        active_state_bits_estimate=state,
        active_state_log2_bits_estimate=log2(state),
        margin_below_category_threshold_bits=(
            cell.category_bit_threshold - total_log2
        ),
    )


def optimized_estimate(cell: Cell) -> Estimate:
    """Choose the least-cost solver degree with failure below 2^-128."""

    admitted = []
    for delta in range(cell.extension_degree, 781, cell.extension_degree):
        value = price(cell, delta)
        if value.aggregate_solver_failure_log2 < -128:
            admitted.append(value)
    if not admitted:
        raise ArithmeticError("no reliable solver field through degree 780")
    return min(
        admitted,
        key=lambda value: (value.total_bit_estimate, value.solver_field_degree),
    )


def current_estimates() -> tuple[Estimate, ...]:
    """Return one current direct-route estimate for every Classic set."""

    return tuple(optimized_estimate(cell) for cell in CELLS)


def internal_checks() -> None:
    expected = (
        (
            "mceliece348864", 815_496_128_413, 1_267_729_188_264,
            3_769_240, 769, 240, 126.77, 51.33, -135.18,
        ),
        (
            "mceliece460896", 151_750_260_798_283, 258_607_050_562_756,
            11_879_850, 1249, 260, 145.22, 59.10, -138.20,
        ),
        (
            "mceliece6688128", 11_586_519_351_014, 17_795_316_080_640,
            15_460_191, 1665, 247, 137.48, 55.18, -132.52,
        ),
        (
            "mceliece6960119", 9_287_018_469_789, 14_264_993_356_992,
            14_444_259, 1548, 247, 136.65, 54.86, -133.25,
        ),
        (
            "mceliece8192128", 11_586_519_351_014, 17_795_316_080_640,
            15_460_191, 1665, 247, 137.48, 55.18, -132.52,
        ),
    )
    values = current_estimates()
    for value, wanted in zip(values, expected):
        shape = value.shape
        (
            name,
            equations,
            unknowns,
            systems,
            labels,
            solver_degree,
            work,
            state,
            failure,
        ) = wanted
        if shape.cell.name != name:
            raise AssertionError("parameter-set ordering changed")
        exact = (
            shape.equations,
            shape.unknowns,
            shape.candidate_systems,
            shape.required_label_count,
            value.solver_field_degree,
        )
        if exact != (equations, unknowns, systems, labels, solver_degree):
            raise AssertionError(f"{name}: exact dimensions or solver degree changed")
        rounded = (
            round(value.total_log2_bit_estimate, 2),
            round(value.active_state_log2_bits_estimate, 2),
            round(value.aggregate_solver_failure_log2, 2),
        )
        if rounded != (work, state, failure):
            raise AssertionError(f"{name}: reported tally changed")
        predecessor = price(shape.cell, solver_degree - shape.cell.extension_degree)
        if predecessor.aggregate_solver_failure_log2 < -128:
            raise AssertionError(f"{name}: predecessor field unexpectedly qualifies")
        if value.margin_below_category_threshold_bits <= 0:
            raise AssertionError(f"{name}: estimate no longer crosses its category level")

    first = values[0]
    exact_category_one = {
        "field_multiply_bit_estimate": 172_081,
        "field_multiply_add_bit_estimate": 172_321,
        "normal_product_bit_estimate": 43_357_797_573_481_659_425,
        "per_system_bit_estimate": 38_386_671_446_795_679_851_939_170_117_672,
        "all_systems_bit_estimate": 144_688_577_484_120_148_325_123_197_574_334_009_280,
        "finisher_bit_estimate": 27_017_420_898_816_121_210_159_497_216,
        "total_bit_estimate": 144_688_577_511_137_569_223_939_318_785_036_260_544,
        "active_state_bits_estimate": 2_825_868_988_017_600,
    }
    for name, wanted in exact_category_one.items():
        if getattr(first, name) != wanted:
            raise AssertionError(f"mceliece348864 {name} changed")


def report(values: tuple[Estimate, ...]) -> None:
    print(
        f"{'parameter set':<19} {'category/reference':<19} {'work':<9} "
        f"{'live state':<14} {'solver field':<13} {'solver failure':<14} margin"
    )
    for value in values:
        cell = value.shape.cell
        category = f"{cell.claimed_category} / 2^{cell.category_bit_threshold}"
        work = f"2^{value.total_log2_bit_estimate:.2f}"
        state = f"2^{value.active_state_log2_bits_estimate:.2f} bits"
        field = f"GF(2^{value.solver_field_degree})"
        failure_exponent = ceil(100 * value.aggregate_solver_failure_log2) / 100
        failure = f"<2^{failure_exponent:.2f}"
        print(
            f"{cell.name:<19} {category:<19} {work:<9} {state:<14} "
            f"{field:<13} {failure:<14} "
            f"{value.margin_below_category_threshold_bits:6.2f}"
        )


def main() -> None:
    if len(sys.argv) != 1:
        raise SystemExit("usage: python3 mccost.py")
    internal_checks()
    report(current_estimates())
    print("internal_checks: PASS")


if __name__ == "__main__":
    main()
