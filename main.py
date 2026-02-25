"""
Main Simulation Entry Point
===========================

Primary entry point for the Digital Clockwork Muse (DCM) simulation.
Orchestrates agent creation, artifact generation, social interaction,
and data logging.

The simulation explores how social interactions and individual creative
processes (based on genetic programming and novelty search) lead to
the evolution of artistic expressions over time.

Key Components:
    - Argument Parsing: CLI configuration with paper-equivalent defaults.
    - Logging: CSV and TensorBoard loggers (async, thread-safe).
    - Scheduler: Manages the agent activation loop (Algorithm 1).
    - Artifact Generator: Creates visual artifacts from expression trees.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import argparse
import concurrent.futures
import random
import time
from datetime import datetime
from typing import List

from tqdm import tqdm

import numpy as np
import torch

from framework import Agent, Artifact, ArtifactGenerator, Logger
from logger import CSVLogger, TensorBoardLogger, CompositeLogger
from scheduler import ParallelScheduler
from genart import ExpressionNode
from timing_utils import time_it, TimingStats

class ExpressionArtifactGenerator(ArtifactGenerator):
    """
    Generates artifacts via expression tree breeding (3.2.2).

    Implements Algorithm 1 step 1: for each agent, breed current
    expression with one from memory (subtree crossover), then
    mutate. If no prior expression exists, create a random tree.
    """
    def __init__(self, mutation_rate: float = 0.1):
        """
        Initializes the artifact generator.

        Args:
            mutation_rate (float): The probability of a mutation occurring in a
                                   newly generated expression tree.
        """
        self.mutation_rate = mutation_rate

    @time_it
    def _generate_for_agent(self, agent: Agent):
        """
        Generates a new expression for a single agent (3.2.2).

        If no prior expression: create random tree of depth gen_depth.
        Otherwise: breed current expression with one from memory,
        apply mutation.

        DEVIATION(paper 3.2.2): Max 5 breeding attempts with
        forced high-mutation fallback. Paper describes breeding +
        mutation without retry logic. This ensures the agent
        always produces a novel-looking expression.

        Args:
            agent (Agent): The agent for whom to generate.

        Returns:
            Artifact: New artifact with generated expression.
        """
        parent1_id, parent2_id = None, None
        
        if not agent.current_expression:
            # First step: create random tree (3.2.2)
            # Depth is agent.gen_depth (set in scheduler init)
            new_expr = ExpressionNode.create_random(depth=agent.gen_depth)
        else:
            parent1_id = agent.current_artifact_id
            
            # DEVIATION(paper 3.2.2): Retry loop with up to 5
            # breeding attempts. If all produce identical output,
            # fall back to 3x mutation rate to force variation.
            max_attempts = 5
            current_expr_str = agent.current_expression.to_string()
            
            for attempt in range(max_attempts):
                if agent.artifact_memory:
                    other_artifact_dict = random.choice(agent.artifact_memory)
                    other_expr = other_artifact_dict['expression']
                    parent2_id = other_artifact_dict['id']
                    new_expr = agent.current_expression.breed(other_expr)
                    new_expr.mutate(rate=self.mutation_rate, max_depth=agent.gen_depth)
                else:
                    new_expr = agent.current_expression._copy()
                    new_expr.mutate(rate=self.mutation_rate, max_depth=agent.gen_depth)

                # Check if different from current
                if new_expr.to_string() != current_expr_str:
                    break
            else:
                # All attempts produced same expression — force
                # high-mutation fallback (3x rate)
                new_expr = agent.current_expression._copy()
                new_expr.mutate(rate=self.mutation_rate * 3, max_depth=agent.gen_depth)

        origin_creator_id = agent.current_creator_id if agent.current_creator_id is not None else agent.unique_id
        return Artifact(
            content=new_expr,
            creator_id=origin_creator_id,
            parent1_id=parent1_id,
            parent2_id=parent2_id,
            producer_id=agent.unique_id
        )
        
    @time_it
    def generate(self, agents: List['Agent']):
        """
        Generates a new artifact for each agent.

        Args:
            agents (List['Agent']): All agents in the simulation.

        Returns:
            List[Artifact]: One new artifact per agent.
        """
        return [self._generate_for_agent(agent) for agent in agents]

def set_seed(seed):
    """
    Sets the seed for all random number generators to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    """
    Main entry point for the creative agent simulation.

    This function sets up the simulation environment, including configuration,
    logging, artifact generation, and the main scheduler. It then runs the
    simulation for a specified number of steps and handles proper cleanup.
    """
    
    # --- Argument Parsing ---
    # Defaults correspond to paper values unless noted.
    parser = argparse.ArgumentParser(description="Run a creative agent simulation.")
    parser.add_argument('--num_agents', type=int, default=250,
                        help='Number of agents (3.5: paper uses 250).')
    parser.add_argument('--num_steps', type=int, default=2000,
                        help='Simulation steps (3.5: paper uses 2000).')
    parser.add_argument('--share_count', type=int, default=5,
                        help='N: agents to share with (Algorithm 1 step 3).')
    parser.add_argument('--uniform_novelty_pref', action='store_true',
                        help='All agents get preferred_novelty=0.5. '
                             'Default: drawn from N(0.5, 0.155) (3.3.4).')
    parser.add_argument('--mutation_rate', type=float, default=0.05,
                        help='Per-node mutation probability (3.2.2).')
    parser.add_argument('--use_static_noise', action='store_true',
                        help='Replace expression rendering with random RGB noise (debug).')
    parser.add_argument('--time_it', action='store_true', 
                        help='Enable per-function timing instrumentation.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for full reproducibility.')
    parser.add_argument('--feature_dims', type=int, default=0,
                        help='Feature dimensionality. '
                             '0 = raw Layer 2 (128d, no reduction). '
                             '64 = PCA to 64d (paper experiments). '
                             'See DEVIATION note in features.py.')
    parser.add_argument('--pca_calibration_samples', type=int, default=500,
                        help='Random artifacts for PCA fitting. '
                             'Only used when --feature_dims > 0.')
    parser.add_argument('--distance_metric', type=str, default='cosine',
                        choices=['cosine', 'euclidean'],
                        help='kNN metric. DEVIATION(paper 3.3.1): '
                             'paper uses euclidean; code defaults to cosine.')
    parser.add_argument('--boredom_mode', type=str, default='extended',
                        choices=['classic', 'extended'],
                        help='DEVIATION(paper 3.4): "classic" matches paper; '
                             '"extended" adds hedonic retreat (default).')
    parser.add_argument('--save_images', action='store_true',
                        help='Save rendered artifact PNGs (debug).')
    parser.add_argument('--image_output_dir', type=str, default=None,
                        help='Directory for --save_images output.')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Override log output directory.')
    args = parser.parse_args()

    # --- Configuration ---
    # Use the parsed arguments instead of hard-coded values
    num_agents = args.num_agents
    num_steps = args.num_steps
    share_count = args.share_count
    uniform_novelty_pref = args.uniform_novelty_pref
    mutation_rate = args.mutation_rate
    use_static_noise = args.use_static_noise
    
    # Set random seed for reproducibility
    print(f"Setting simulation seed to: {args.seed}")
    set_seed(args.seed) 

    # Setup logging
    if args.log_dir:
        log_dir = args.log_dir
    else:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_dir = os.path.join("logs", run_name)
    os.makedirs(log_dir, exist_ok=True)

    # --- Main Events CSV Logger Setup ---
    csv_log_file = os.path.join(log_dir, "events.csv")
    log_fields = [
        'timestamp', 'event_type', 'step', 'agent_id', 'artifact_id', 'expression',
        'novelty', 'interest', 'sender_id', 'recipient_id', 'adopted',
        'evaluated_novelty', 'evaluated_interest', 'accepted',
        'creator_id', 'evaluator_id', 'domain_size',
        'parent1_id', 'parent2_id',
        'source', 'trigger_novelty'
    ]
    csv_logger = CSVLogger(
        log_file_path=csv_log_file, 
        fieldnames=log_fields,
        allowed_event_types=['generation', 'share', 'boredom_adoption']
    )

    # --- Agent Initialization Logger Setup ---
    agent_init_log_file = os.path.join(log_dir, "agent_init.csv")
    agent_init_log_fields = ['agent_id', 'preferred_novelty', 'reward_mean', 'punishment_mean']
    agent_init_logger = CSVLogger(
        log_file_path=agent_init_log_file,
        fieldnames=agent_init_log_fields,
        allowed_event_types=['agent_init']
    )

    # --- Agent State Logger Setup ---
    agent_state_log_file = os.path.join(log_dir, "agent_state.csv")
    agent_state_log_fields = [
        'step', 'agent_id', 'cumulative_interest', 'repository_size', 'k_value', 
        'boredom_triggered', 'num_self_evals', 'num_other_evals', 'num_shares', 
        'num_domain_adoptions', 'avg_novelty_generated', 'avg_interest_generated'
    ]
    agent_state_logger = CSVLogger(
        log_file_path=agent_state_log_file,
        fieldnames=agent_state_log_fields,
        allowed_event_types=['agent_state']
    )

    # --- TensorBoard Logger Setup ---
    tensorboard_logger = TensorBoardLogger(log_dir=log_dir)

    # --- Composite Logger Setup ---
    logger = CompositeLogger(loggers=[
        csv_logger, 
        tensorboard_logger, 
        agent_init_logger, 
        agent_state_logger
    ])

    # Setup artifact generator
    artifact_generator = ExpressionArtifactGenerator(mutation_rate=mutation_rate)

    # Setup and run the scheduler
    image_output_dir = args.image_output_dir or os.path.join(log_dir, "images")
    scheduler = ParallelScheduler(
            num_agents=num_agents,
            artifact_generator=artifact_generator,
            logger=logger,
            share_count=share_count,
            uniform_novelty_pref=uniform_novelty_pref,
            use_static_noise=use_static_noise,
            feature_dims=args.feature_dims,
            pca_calibration_samples=args.pca_calibration_samples,
            distance_metric=args.distance_metric,
            boredom_mode=args.boredom_mode,
            save_images=args.save_images,
            image_output_dir=image_output_dir
    )

    print(f"Starting simulation with {num_agents} agents for {num_steps} steps.")
    print(f"Sharing with {share_count} agents. Uniform novelty: {uniform_novelty_pref}")
    print(f"Mutation rate: {mutation_rate}")
    print(f"Using static noise: {use_static_noise}")
    print(f"Logs will be saved in: {log_dir}")
    
    if args.time_it:
        timing_stats = TimingStats()
        print("Function timing is ENABLED.")

    try:
        for i in tqdm(range(num_steps), desc="Simulation Progress"):
            scheduler.step()
            
            if args.time_it:
                print(f"\n--- Function Timing Report for Step {i} ---")
                timing_stats.print_step_report()
                timing_stats.reset_step()
        print("Simulation finished successfully.")
    except Exception as e:
        print(f"An error occurred during the simulation: {e}")
    finally:
        scheduler.close()
        print("Logger closed.")

if __name__ == "__main__":
    main()