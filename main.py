# Remove the two lines below if you are not on macOS or do not encounter the "libiomp5.dylib" error.
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import traceback
import random
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
    Generates artifacts via expression tree breeding.

    For each agent, breeds their current expression with one from memory 
    using subtree crossover, followed by mutation. If no prior expression 
    exists, a random expression tree is generated.
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
        Generates a new expression for a single agent.

        If no prior expression exists, creates a random tree of depth gen_depth.
        Otherwise, breeds the current expression with one selected from memory
        and applies mutation.

        Args:
            agent (Agent): The agent for whom to generate.

        Returns:
            Artifact: New artifact containing the generated expression.
        """
        parent1_id, parent2_id = None, None
        
        if not agent.current_expression:
            # Generate a new random expression tree if the agent has no current expression
            new_expr = ExpressionNode.create_random(depth=agent.gen_depth)
        else:
            parent1_id = agent.current_artifact_id
            
            max_attempts = 5
            current_expr_str = agent.current_expression.to_string()
            
            for attempt in range(max_attempts):
                if agent.artifact_memory:
                    # Breed current expression with an expression selected from memory
                    other_artifact_dict = random.choice(agent.artifact_memory)
                    other_expr = other_artifact_dict['expression']
                    parent2_id = other_artifact_dict['id']
                    new_expr = agent.current_expression.breed(other_expr)
                    new_expr.mutate(rate=self.mutation_rate, max_depth=agent.gen_depth)
                else:
                    # Mutate a copy of the current expression if memory is empty
                    new_expr = agent.current_expression._copy()
                    new_expr.mutate(rate=self.mutation_rate, max_depth=agent.gen_depth)

                # Ensure the new expression differs structurally from the current one
                if new_expr.to_string() != current_expr_str:
                    break
            else:
                # Fallback: Apply a higher mutation rate if variations could not be generated
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
        Generates a new artifact for each agent in the provided list.

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

@time_it
def main():
    """
    Main entry point for the creative agent simulation.

    Sets up the simulation environment, including configuration parsing,
    logging, artifact generation, and the parallel execution scheduler. 
    Runs the activation loop for the specified duration and manages resource cleanup.
    """
    
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run a creative agent simulation.")
    parser.add_argument('--num_agents', type=int, default=250,
                        help='Number of agents.')
    parser.add_argument('--num_steps', type=int, default=2500,
                        help='Simulation steps.')
    parser.add_argument('--share_count', type=int, default=5,
                        help='N: agents to share with.')
    parser.add_argument('--uniform_novelty_pref', action='store_true',
                        help='All agents get preferred_novelty=0.5. '
                             'Default: drawn from N(0.5, 0.15).')
    parser.add_argument('--mutation_rate', type=float, default=0.05,
                        help='Per-node mutation probability.')
    parser.add_argument('--use_static_noise', action='store_true',
                        help='Replace expression rendering with random RGB noise (debug).')
    parser.add_argument('--time_it', action='store_true', 
                        help='Enable per-function timing instrumentation.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for full reproducibility.')
    parser.add_argument('--pca_dims', type=int, default=128,
                        help='PCA dimensionality. '
                             'Default is 128. Recommened to set to 16')
    parser.add_argument('--use_personal_threshold', action='store_true',
                        help='Track and apply dynamically evaluated sharing thresholds internally, per-agent.')
    parser.add_argument('--pca_calibration_samples', type=int, default=5000,
                        help='Random artifacts for PCA fitting. '
                             'Only used when --pca_dims > 0.')
    parser.add_argument('--save_images', action='store_true',
                        help='Save rendered artifact PNGs (debug).')
    parser.add_argument('--image_output_dir', type=str, default=None,
                        help='Directory for --save_images output.')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Override log output directory.')

    integrity_group = parser.add_mutually_exclusive_group()
    integrity_group.add_argument('--strict_integrity_mode', dest='strict_integrity_mode', action='store_true',
                                 help='Keep strict synchronization behavior for deterministic trace checks (default).')
    integrity_group.add_argument('--no_strict_integrity_mode', dest='strict_integrity_mode', action='store_false',
                                 help='Allow reduced synchronization for performance experiments (not trace-stable).')
    parser.set_defaults(strict_integrity_mode=True)
    args = parser.parse_args()

    # --- Configuration Setup ---
    num_agents = args.num_agents
    num_steps = args.num_steps
    share_count = args.share_count
    uniform_novelty_pref = args.uniform_novelty_pref
    mutation_rate = args.mutation_rate
    use_static_noise = args.use_static_noise
    
    print(f"Setting simulation seed to: {args.seed}")
    set_seed(args.seed) 

    # Directory configuration for run artifacts and logging
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
        'source', 'trigger_novelty',
        'self_threshold', 'domain_threshold', 'boredom_threshold',
        'avg_accepted_interest', 'avg_rejected_interest',
        'accepted_count', 'rejected_count',
        'avg_knn_size', 'avg_current_interest', 'avg_average_interest', 'avg_current_novelty',
        'total_self_evals', 'total_other_evals', 'total_shares', 'total_domain_adoptions'
    ]
    csv_logger = CSVLogger(
        log_file_path=csv_log_file, 
        fieldnames=log_fields,
        allowed_event_types=['generation', 'share', 'boredom_adoption', 'step_end']
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

    # Initialize components and scheduler execution context
    artifact_generator = ExpressionArtifactGenerator(mutation_rate=mutation_rate)

    image_output_dir = args.image_output_dir or os.path.join(log_dir, "images")
    scheduler = ParallelScheduler(
            num_agents=num_agents,
            artifact_generator=artifact_generator,
            logger=logger,
            share_count=share_count,
            uniform_novelty_pref=uniform_novelty_pref,
            use_static_noise=use_static_noise,
            pca_dims=args.pca_dims,
            pca_calibration_samples=args.pca_calibration_samples,
            strict_integrity_mode=args.strict_integrity_mode,
            save_images=args.save_images,
            image_output_dir=image_output_dir,
            use_personal_threshold=args.use_personal_threshold
    )

    print(f"Starting simulation with {num_agents} agents for {num_steps} steps.")
    print(f"Sharing with {share_count} agents. Uniform novelty: {uniform_novelty_pref}")
    print(f"Mutation rate: {mutation_rate}")
    print(f"Using static noise: {use_static_noise}")
    print(f"PCA dimensions: {args.pca_dims}")
    print(f"PCA calibration samples: {args.pca_calibration_samples}")
    print(f"Use personal threshold: {args.use_personal_threshold}")
    print(f"Strict integrity mode: {args.strict_integrity_mode}")
    print(f"Logs will be saved in: {log_dir}")
    
    if args.time_it:
        import timing_utils
        timing_utils.ENABLE_TIMING = True
        timing_stats = TimingStats()
        print("Function timing is ENABLED.")

    import torch
    if torch.cuda.is_available():
        print(f"GPU reserved after init: "
              f"{torch.cuda.memory_reserved()/1e9:.2f} GB")
              
    try:
        # Main simulation loop
        for i in tqdm(range(num_steps), desc="Simulation Progress"):
            scheduler.step()
            
            if args.time_it:
                print(f"\n--- Function Timing Report for Step {i} ---")
                timing_stats.print_step_report()
                timing_stats.reset_step()
        print("Simulation finished successfully.")
    except Exception as e:
        print(f"An error occurred during the simulation: {e}")
        traceback.print_exc()
    finally:
        # Resource cleanup
        scheduler.close()
        print("Logger closed.")

if __name__ == "__main__":
    main()