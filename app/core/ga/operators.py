import random
from typing import Literal

from app.core.ga.individual import Individual


def tournament_selection(
    population: list[Individual],
    tournament_size: int = 3,
) -> Individual:
    """
    Tournament selection: pick best from random sample.

    Args:
        population (list[Individual]): List of chromosomes
        tournament_size (int): Number of individuals in tournament

    Returns:
        Individual: Selected Individual
    """
    tournament = random.sample(population, min(tournament_size, len(population)))
    return min(tournament, key=lambda c: c.fitness)


def roulette_selection(population: list[Individual]) -> Individual:
    """
    Roulette wheel selection based on fitness.

    For minimization: invert fitness to get selection probability.

    Args:
        population (list[Individual]): List of chromosomes

    Returns:
        Individual: Selected Individual
    """
    # Invert fitness for minimization (lower is better)
    max_fitness = max(c.fitness for c in population)
    inverted_fitness = [max_fitness - c.fitness + 1 for c in population]
    total_fitness = sum(inverted_fitness)

    if total_fitness == 0:
        return random.choice(population)

    # Roulette wheel
    pick = random.uniform(0, total_fitness)
    current = 0
    for i, fitness in enumerate(inverted_fitness):
        current += fitness
        if current >= pick:
            return population[i]

    return population[-1]


def order_crossover(parent1: Individual, parent2: Individual) -> Individual:
    """
    Order Crossover (OX): preserves relative order of customers.

    1. Select random segment from parent1
    2. Copy segment to child
    3. Fill remaining positions with parent2 order

    Args:
        parent1 (Individual): First parent
        parent2 (Individual): Second parent

    Returns:
        Individual: Child Individual
    """
    # Get flat customer sequences
    seq1 = parent1.get_all_customers()
    seq2 = parent2.get_all_customers()

    if len(seq1) <= 2:
        return parent1.copy()

    # Select crossover points
    size = len(seq1)
    cx_point1 = random.randint(0, size - 2)
    cx_point2 = random.randint(cx_point1 + 1, size)

    # Copy segment from parent1
    child_seq = [None] * size
    child_seq[cx_point1:cx_point2] = seq1[cx_point1:cx_point2]
    copied = set(seq1[cx_point1:cx_point2])

    # Fill remaining with parent2 order
    p2_idx = 0
    for i in range(size):
        if child_seq[i] is None:
            # Find next customer from parent2 not in copied segment
            while seq2[p2_idx] in copied:
                p2_idx += 1
            child_seq[i] = seq2[p2_idx]
            p2_idx += 1

    # Convert back to routes
    return Individual.from_giant_tour(child_seq, parent1.instance)


def partially_mapped_crossover(
    parent1: Individual,
    parent2: Individual,
) -> Individual:
    """
    Partially Mapped Crossover (PMX).

    Args:
        parent1 (Individual): First parent
        parent2 (Individual): Second parent

    Returns:
        Individual: Child Individual
    """
    seq1 = parent1.get_all_customers()
    seq2 = parent2.get_all_customers()

    if len(seq1) <= 2:
        return parent1.copy()

    size = len(seq1)
    cx_point1 = random.randint(0, size - 2)
    cx_point2 = random.randint(cx_point1 + 1, size)

    # Initialize child with parent1
    child_seq = seq1.copy()

    # Create mapping from crossover segment
    mapping = {}
    for i in range(cx_point1, cx_point2):
        mapping[seq2[i]] = seq1[i]

    # Apply mapping outside segment
    for i in range(size):
        if i < cx_point1 or i >= cx_point2:
            value = seq2[i]
            while value in mapping:
                value = mapping[value]
            child_seq[i] = value

    return Individual.from_giant_tour(child_seq, parent1.instance)


def swap_mutation(individual: Individual) -> Individual:
    """
    Swap mutation: exchange two random customers.

    Args:
        individual: Individual to mutate

    Returns:
        Individual: Mutated individual
    """
    mutated = individual.copy()
    customers = mutated.get_all_customers()

    if len(customers) < 2:
        return mutated

    # Get flat sequence and swap
    seq = mutated.get_all_customers()
    i, j = random.sample(range(len(seq)), 2)
    seq[i], seq[j] = seq[j], seq[i]

    # Rebuild routes
    return Individual.from_giant_tour(seq, individual.instance)


def inversion_mutation(individual: Individual) -> Individual:
    """
    Inversion mutation: reverse a random segment.

    Args:
        individual: Individual to mutate

    Returns:
        Individual: Mutated individual
    """
    mutated = individual.copy()
    seq = mutated.get_all_customers()

    if len(seq) < 2:
        return mutated

    # Select segment and reverse
    i = random.randint(0, len(seq) - 2)
    j = random.randint(i + 1, len(seq))
    seq[i:j] = reversed(seq[i:j])

    return Individual.from_giant_tour(seq, individual.instance)


def insert_mutation(individual: Individual) -> Individual:
    """
    Insert mutation: move a customer to a different position.

    Args:
        individual: Individual to mutate

    Returns:
        Individual: Mutated individual
    """
    mutated = individual.copy()
    seq = mutated.get_all_customers()

    if len(seq) < 2:
        return mutated

    # Remove and insert at different position
    i = random.randint(0, len(seq) - 1)
    customer = seq.pop(i)
    j = random.randint(0, len(seq))
    seq.insert(j, customer)

    return Individual.from_giant_tour(seq, individual.instance)


def two_opt_mutation(individual: Individual) -> Individual:
    """
    2-opt mutation: remove two edges and reconnect differently.

    This is applied within a single route to improve it locally.

    Args:
        Individual: Individual to mutate

    Returns:
        Mutated Individual
    """
    mutated = individual.copy()

    if not mutated.routes:
        return mutated

    # Select random route
    route_idx = random.randint(0, len(mutated.routes) - 1)
    route = mutated.routes[route_idx]

    if len(route) < 4:  # Need at least 4 customers for 2-opt
        return mutated

    # Select two edges
    i = random.randint(0, len(route) - 3)
    j = random.randint(i + 2, len(route) - 1)

    # Reverse segment between i and j
    mutated.routes[route_idx] = (
        route[: i + 1] + route[i + 1 : j + 1][::-1] + route[j + 1 :]
    )
    mutated._fitness = None  # Invalidate cached fitness

    return mutated


def apply_mutation(
    individual: Individual,
    method: Literal["swap", "insert", "inversion", "2opt"] = "swap",
) -> Individual:
    """
    Apply mutation based on method.

    Args:
        individual (Individual): Individual to mutate
        method (Literal["swap", "insert", "inversion", "2opt"]): Mutation method

    Returns:
        Individual: Mutated Individual
    """
    mutation_functions = {
        "swap": swap_mutation,
        "insert": insert_mutation,
        "inversion": inversion_mutation,
        "2opt": two_opt_mutation,
    }

    mutation_func = mutation_functions.get(method, swap_mutation)
    return mutation_func(individual)


def apply_crossover(
    parent1: Individual,
    parent2: Individual,
    method: Literal["ox", "pmx", "edge"] = "ox",
) -> Individual:
    """
    Apply crossover based on method.

    Args:
        parent1 (Individual): First parent
        parent2 (Individual): Second parent
        method (Literal["ox", "pmx", "edge"]): Crossover method

    Returns:
        Individual: Child Individual
    """
    if method == "pmx":
        return partially_mapped_crossover(parent1, parent2)
    elif method == "edge":
        # Edge crossover is complex, use OX as fallback
        return order_crossover(parent1, parent2)
    else:  # "ox"
        return order_crossover(parent1, parent2)


def apply_selection(
    population: list[Individual],
    method: Literal["tournament", "roulette"] = "tournament",
    tournament_size: int = 3,
) -> Individual:
    """
    Apply selection based on method.

    Args:
        population (list[Individual]): List of chromosomes
        method (Literal["tournament", "roulette"]): Selection method
        tournament_size (int): Tournament size for tournament selection

    Returns:
        Individual: Selected Individual
    """
    if method == "roulette":
        return roulette_selection(population)
    else:  # "tournament"
        return tournament_selection(population, tournament_size)
