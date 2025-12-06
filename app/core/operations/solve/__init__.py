import time

import torch

from app.api.routers.solve.payload_schemas import ComparisonRequest
from app.api.routers.solve.response_schemas import (
    AlgorithmResult,
    ComparisonMetrics,
    ComparisonResponse,
)
from app.config import settings
from app.config.logging import logger
from app.core.drl.actor_critic_agent import ActorCriticAgent
from app.core.drl.population_generator import generate_population_with_drl
from app.core.ga.genetic_algorithm import GeneticAlgorithm
from app.core.operations.instances import load_instance_by_id
from app.exceptions import (
    ModelLoadException,
    ModelNotFoundException,
)
from app.schemas import DRLConfig


def run_comparision(
    request: ComparisonRequest, session_id: str | None = None
) -> ComparisonResponse:
    """
    Run comparison between NeuroGen (GA + DRL) and pure GA.

    Args:
        request (ComparisonRequest): Comparison request parameters
        session_id (str | None): Optional session identifier for instance isolation

    Returns:
        ComparisonResponse: Results of the comparison

    Raises:
        ModelNotFoundException: If the specified model doesn't exist
        ModelLoadException: If the model fails to load
        InstanceParseException: If the instance cannot be loaded
    """
    instance = load_instance_by_id(request.instance_id, session_id=session_id)

    model_path = settings.CHECKPOINTS_DIR / f"{request.drl_model_id}.pth"
    if not model_path.exists():
        raise ModelNotFoundException(request.drl_model_id)

    try:
        checkpoint = torch.load(model_path, map_location="cpu")
    except Exception as e:
        raise ModelLoadException(request.drl_model_id, str(e))

    drl_config = DRLConfig(
        embedding_dim=256,
        hidden_dim=256,
        num_layers=2,
        episodes=0,  # Not training
        device="cpu",
    )
    agent = ActorCriticAgent(instance, drl_config)

    try:
        agent.network.load_state_dict(checkpoint["network"])
        agent.network.eval()
        agent.epsilon = 0.0
    except Exception as e:
        raise ModelLoadException(request.drl_model_id, f"Invalid weights: {str(e)}")

    # Generate DRL-based population (NeuroGen approach)
    logger.info(f"Generating DRL population with {request.drl_model_id}...")
    drl_gen_start = time.time()
    drl_population = generate_population_with_drl(
        instance=instance,
        agent=agent,
        population_size=request.ga_config.population_size,
        diversity=0.2,
    )
    drl_gen_time = time.time() - drl_gen_start
    logger.info(f"DRL population generated in {drl_gen_time:.2f}s")

    logger.info("Running GA with DRL population (NeuroGen)...")
    ga_neurogen = GeneticAlgorithm(
        instance=instance,
        config=request.ga_config,
    )
    ga_neurogen.initialize_population(custom_population=drl_population)
    initial_neurogen_best = ga_neurogen.get_best_individual().fitness

    neurogen_ga_start = time.time()
    solution_neurogen = ga_neurogen.run(custom_population=drl_population)
    neurogen_ga_time = time.time() - neurogen_ga_start
    neurogen_total_time = drl_gen_time + neurogen_ga_time
    logger.info(
        f"NeuroGen GA completed in {neurogen_ga_time:.2f}s (total: {neurogen_total_time:.2f}s): {solution_neurogen.total_cost:.2f}"
    )

    logger.info("Running GA with random population (pure GA)...")
    pure_gen_start = time.time()
    ga_pure = GeneticAlgorithm(
        instance=instance,
        config=request.ga_config,
    )
    ga_pure.initialize_population(use_heuristics=False)
    initial_pure_best = ga_pure.get_best_individual().fitness
    pure_gen_time = time.time() - pure_gen_start

    pure_ga_start = time.time()
    solution_pure = ga_pure.run(use_heuristics=False)
    pure_ga_time = time.time() - pure_ga_start
    pure_total_time = pure_gen_time + pure_ga_time
    logger.info(
        f"Pure GA completed in {pure_ga_time:.2f}s (total: {pure_total_time:.2f}s): {solution_pure.total_cost:.2f}"
    )

    improvement_absolute = solution_pure.total_cost - solution_neurogen.total_cost
    improvement_percentage = (improvement_absolute / solution_pure.total_cost) * 100
    time_difference = neurogen_total_time - pure_total_time
    initial_gap = (
        (initial_pure_best - initial_neurogen_best) / initial_pure_best
    ) * 100
    vehicles_diff = len(solution_neurogen.routes) - len(solution_pure.routes)

    winner = (
        "neurogen"
        if solution_neurogen.total_cost < solution_pure.total_cost
        else "ga_pure"
    )
    if abs(improvement_percentage) < 0.5:
        winner = "tie"

    neurogen_convergence = [
        {"generation": i, "fitness": fitness}
        for i, fitness in enumerate(ga_neurogen.convergence_history)
    ]
    pure_convergence = [
        {"generation": i, "fitness": fitness}
        for i, fitness in enumerate(ga_pure.convergence_history)
    ]

    return ComparisonResponse(
        neurogen=AlgorithmResult(
            algorithm_name="GA + DRL (NeuroGen)",
            initial_fitness=initial_neurogen_best,
            final_solution=solution_neurogen,
            computation_time=neurogen_total_time,
            population_generation_time=drl_gen_time,
            ga_convergence_time=neurogen_ga_time,
            convergence_history=neurogen_convergence,
        ),
        ga_pure=AlgorithmResult(
            algorithm_name="Pure GA",
            initial_fitness=initial_pure_best,
            final_solution=solution_pure,
            computation_time=pure_total_time,
            population_generation_time=pure_gen_time,
            ga_convergence_time=pure_ga_time,
            convergence_history=pure_convergence,
        ),
        metrics=ComparisonMetrics(
            improvement_absolute=improvement_absolute,
            improvement_percentage=improvement_percentage,
            time_difference=time_difference,
            initial_gap_percentage=initial_gap,
            vehicles_difference=vehicles_diff,
            winner=winner,
        ),
    )
