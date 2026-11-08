"""Provide deterministic discrete-genome operators for steel-section optimization."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from pydantic import Field, model_validator

from steel_frame_optimization.config.schemas import StrictConfigModel


class Genome(StrictConfigModel):
    """Represent one immutable vector of zero-based section-domain alleles."""

    alleles: tuple[int, ...] = Field(min_length=1)


class GeneticAlgorithmPolicy(StrictConfigModel):
    """Freeze all stochastic operators needed to reproduce one GA run."""

    selection: Literal["tournament"]
    tournament_size: int = Field(ge=2)
    crossover: Literal["one_point"]
    crossover_probability: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    mutation: Literal["ordered_neighborhood"]
    mutation_probability_per_gene: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    mutation_window_fraction: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    elite_count: int = Field(ge=1)

    @model_validator(mode="after")
    def require_nontrivial_variation(self) -> GeneticAlgorithmPolicy:
        """Reject policies that cannot create or retain a useful population."""

        if self.crossover_probability == 0.0 and self.mutation_probability_per_gene == 0.0:
            raise ValueError("at least one variation operator must be active")
        return self


class DiscreteGeneticAlgorithm:
    """Generate and evolve bounded integer genomes using one explicit RNG instance."""

    def __init__(
        self,
        *,
        domain_sizes: Sequence[int],
        policy: GeneticAlgorithmPolicy,
        rng: np.random.Generator,
    ) -> None:
        """Validate domain bounds and retain deterministic operator state."""

        if not domain_sizes or any(size < 2 for size in domain_sizes):
            raise ValueError("every GA domain must contain at least two alleles")
        self.domain_sizes = tuple(int(size) for size in domain_sizes)
        self.policy = policy
        self.rng = rng

    def validate_genome(self, genome: Genome) -> None:
        """Reject a genome whose length or alleles fall outside configured domains."""

        if len(genome.alleles) != len(self.domain_sizes):
            raise ValueError("genome length does not match domain count")
        for gene_index, (allele, domain_size) in enumerate(
            zip(genome.alleles, self.domain_sizes, strict=True)
        ):
            if allele < 0 or allele >= domain_size:
                raise ValueError(f"allele {gene_index} is outside its domain")

    def initialize_population(
        self,
        population_size: int,
        *,
        anchors: Sequence[Genome] = (),
    ) -> tuple[Genome, ...]:
        """Create an anchor-first population followed by uniform random genomes."""

        if population_size <= 0:
            raise ValueError("population size must be positive")
        if len(anchors) > population_size:
            raise ValueError("anchors cannot exceed population size")
        population = list(anchors)
        for anchor in population:
            self.validate_genome(anchor)
        while len(population) < population_size:
            population.append(
                Genome(
                    alleles=tuple(
                        int(self.rng.integers(0, domain_size)) for domain_size in self.domain_sizes
                    )
                )
            )
        return tuple(population)

    def tournament_index(self, rank_keys: Sequence[tuple[object, ...]]) -> int:
        """Return the stable best index from a sampled tournament with replacement."""

        if not rank_keys:
            raise ValueError("tournament requires at least one candidate")
        sampled = tuple(
            int(self.rng.integers(0, len(rank_keys))) for _ in range(self.policy.tournament_size)
        )
        return min(sampled, key=lambda index: (rank_keys[index], index))

    def crossover_pair(self, first: Genome, second: Genome) -> tuple[Genome, Genome]:
        """Apply deterministic-RNG one-point crossover while preserving legal alleles."""

        self.validate_genome(first)
        self.validate_genome(second)
        gene_count = len(self.domain_sizes)
        if gene_count < 2 or float(self.rng.random()) >= self.policy.crossover_probability:
            return first.model_copy(deep=True), second.model_copy(deep=True)
        point = int(self.rng.integers(1, gene_count))
        return (
            Genome(alleles=first.alleles[:point] + second.alleles[point:]),
            Genome(alleles=second.alleles[:point] + first.alleles[point:]),
        )

    def mutate(self, genome: Genome) -> Genome:
        """Mutate genes within their weight-ordered local neighborhoods."""

        self.validate_genome(genome)
        mutated = list(genome.alleles)
        for gene_index, (allele, domain_size) in enumerate(
            zip(genome.alleles, self.domain_sizes, strict=True)
        ):
            if float(self.rng.random()) >= self.policy.mutation_probability_per_gene:
                continue
            window = max(1, round((domain_size - 1) * self.policy.mutation_window_fraction))
            lower = max(0, allele - window)
            upper = min(domain_size - 1, allele + window)
            alternatives = tuple(
                candidate for candidate in range(lower, upper + 1) if candidate != allele
            )
            if alternatives:
                mutated[gene_index] = alternatives[int(self.rng.integers(0, len(alternatives)))]
        result = Genome(alleles=tuple(mutated))
        self.validate_genome(result)
        return result

    def next_population(
        self,
        *,
        population: Sequence[Genome],
        rank_keys: Sequence[tuple[object, ...]],
    ) -> tuple[Genome, ...]:
        """Retain stable elites and fill the remaining population with varied offspring."""

        if len(population) != len(rank_keys) or not population:
            raise ValueError("population and rank keys must have the same positive length")
        if self.policy.elite_count >= len(population):
            raise ValueError("elite count must be smaller than population size")
        for genome in population:
            self.validate_genome(genome)
        ranked_indices = sorted(
            range(len(population)), key=lambda index: (rank_keys[index], population[index].alleles)
        )
        next_generation = [
            population[index].model_copy(deep=True)
            for index in ranked_indices[: self.policy.elite_count]
        ]
        while len(next_generation) < len(population):
            first = population[self.tournament_index(rank_keys)]
            second = population[self.tournament_index(rank_keys)]
            children = self.crossover_pair(first, second)
            for child in children:
                next_generation.append(self.mutate(child))
                if len(next_generation) == len(population):
                    break
        return tuple(next_generation)


def serialize_rng_state(rng: np.random.Generator) -> str:
    """Return the PCG64 state as stable compact JSON for exact recovery."""

    return json.dumps(rng.bit_generator.state, sort_keys=True, separators=(",", ":"))


def restore_pcg64(state_json: str) -> np.random.Generator:
    """Recreate a PCG64 generator from a persisted state without drawing a value."""

    state = json.loads(state_json)
    if not isinstance(state, dict) or state.get("bit_generator") != "PCG64":
        raise ValueError("checkpoint RNG state is not PCG64")
    bit_generator = np.random.PCG64()
    bit_generator.state = cast(Any, state)
    return np.random.Generator(bit_generator)


def load_ga_policy(path: Path) -> GeneticAlgorithmPolicy:
    """Load one standalone GA operator policy for tests or tooling."""

    return GeneticAlgorithmPolicy.model_validate_json(path.read_text(encoding="utf-8"))
