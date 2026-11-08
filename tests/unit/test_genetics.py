"""Layer 1 tests for deterministic discrete genetic operators."""

from __future__ import annotations

import numpy as np
import pytest

from steel_frame_optimization.optimization.genetics import (
    DiscreteGeneticAlgorithm,
    GeneticAlgorithmPolicy,
    Genome,
    restore_pcg64,
    serialize_rng_state,
)

pytestmark = pytest.mark.unit


def _policy() -> GeneticAlgorithmPolicy:
    """Return one compact operator policy shared by unit fixtures."""

    return GeneticAlgorithmPolicy(
        selection="tournament",
        tournament_size=3,
        crossover="one_point",
        crossover_probability=0.9,
        mutation="ordered_neighborhood",
        mutation_probability_per_gene=0.4,
        mutation_window_fraction=0.25,
        elite_count=1,
    )


def _ga(seed: int) -> DiscreteGeneticAlgorithm:
    """Build a small deterministic GA fixture."""

    return DiscreteGeneticAlgorithm(
        domain_sizes=(4, 7, 5),
        policy=_policy(),
        rng=np.random.Generator(np.random.PCG64(seed)),
    )


def test_population_initialization_is_seed_reproducible_and_anchor_first() -> None:
    """The same PCG64 seed produces the same bounded anchor-first population."""

    anchor = Genome(alleles=(1, 2, 3))
    first = _ga(17).initialize_population(8, anchors=(anchor,))
    second = _ga(17).initialize_population(8, anchors=(anchor,))
    assert first == second
    assert first[0] == anchor
    for genome in first:
        _ga(0).validate_genome(genome)


def test_operator_sequence_is_reproducible_and_preserves_domain_bounds() -> None:
    """Selection, crossover, mutation and elitism repeat exactly for a fixed seed."""

    population = _ga(9).initialize_population(10)
    rank_keys = tuple(
        (index % 3, float(index), genome.alleles) for index, genome in enumerate(population)
    )
    first_ga = _ga(42)
    second_ga = _ga(42)
    first = first_ga.next_population(population=population, rank_keys=rank_keys)
    second = second_ga.next_population(population=population, rank_keys=rank_keys)
    assert first == second
    expected_elite = min(
        range(len(population)), key=lambda index: (rank_keys[index], population[index].alleles)
    )
    assert first[0] == population[expected_elite]
    for genome in first:
        first_ga.validate_genome(genome)


def test_ordered_mutation_never_leaves_configured_neighborhood() -> None:
    """A forced mutation remains inside each explicit weight-ordered local window."""

    policy = _policy().model_copy(update={"mutation_probability_per_gene": 1.0})
    ga = DiscreteGeneticAlgorithm(
        domain_sizes=(11, 21),
        policy=policy,
        rng=np.random.Generator(np.random.PCG64(3)),
    )
    original = Genome(alleles=(5, 10))
    mutated = ga.mutate(original)
    assert mutated != original
    assert abs(mutated.alleles[0] - 5) <= 2
    assert abs(mutated.alleles[1] - 10) <= 5


def test_pcg64_checkpoint_state_restores_next_draws_exactly() -> None:
    """Serialized RNG state resumes without repeating or skipping stochastic draws."""

    rng = np.random.Generator(np.random.PCG64(1234))
    _ = rng.integers(0, 100, size=7)
    restored = restore_pcg64(serialize_rng_state(rng))
    assert np.array_equal(rng.integers(0, 100, size=20), restored.integers(0, 100, size=20))


def test_invalid_genome_is_rejected() -> None:
    """Out-of-domain alleles cannot reach crossover, mutation or evaluation."""

    with pytest.raises(ValueError, match="outside"):
        _ga(0).validate_genome(Genome(alleles=(4, 0, 0)))
