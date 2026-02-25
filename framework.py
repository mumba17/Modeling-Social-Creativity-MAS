"""
Core Framework Components
=========================

This module defines the fundamental abstract base classes and data structures
used throughout the simulation. It establishes the contracts for Agents,
Artifacts, Generators, Schedulers, and Loggers, ensuring modularity and
extensibility.
"""

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import torch
from genart import ExpressionNode, QuaternionTensor

class Artifact:
    """
    Represents a generated creative artifact.

    An artifact is the fundamental unit of exchange and evaluation in the system.
    It contains the content (the expression tree), metadata about its creation
    (lineage), and cached features for evaluation.

    Attributes:
        id (int): Unique identifier for the artifact.
        content (Any): The core content of the artifact (e.g., ExpressionNode).
        creator_id (int): The ID of the original creator of this artifact's lineage.
        producer_id (int): The ID of the agent that produced this concrete artifact instance.
        features (Optional[torch.Tensor]): Feature vector representing the artifact
            in the conceptual space (used for distance calculations).
        novelty (Optional[float]): The calculated novelty score of the artifact.
        interest (Optional[float]): The calculated interest score of the artifact.
        parent1_id (Optional[int]): ID of the first parent artifact (if bred).
        parent2_id (Optional[int]): ID of the second parent artifact (if bred).
    """
    next_id = 0
    def __init__(
        self,
        content: Any,
        creator_id: int,
        parent1_id: Optional[int] = None,
        parent2_id: Optional[int] = None,
        producer_id: Optional[int] = None
    ):
        self.id = Artifact.next_id
        Artifact.next_id += 1
        self.content = content
        self.creator_id = creator_id
        self.producer_id = creator_id if producer_id is None else producer_id
        self.features: Optional[torch.Tensor] = None
        self.novelty: Optional[float] = None
        self.interest: Optional[float] = None
        self.parent1_id = parent1_id
        self.parent2_id = parent2_id

class ArtifactGenerator(abc.ABC):
    """
    Abstract base class for artifact generation strategies.

    Subclasses should implement the specific logic for creating new artifacts,
    whether through random generation, evolutionary algorithms, or other
    creative processes.
    """
    @abc.abstractmethod
    def generate(self, agent_ids: List[int]) -> List[Artifact]:
        """
        Generates a batch of new artifacts.

        Args:
            agent_ids (List[int]): A list of agent IDs for whom to generate artifacts.

        Returns:
            List[Artifact]: A list of generated artifacts, corresponding to the input IDs.
        """
        pass

@dataclass
class Agent:
    """
    Represents a creative agent in the simulation.

    Agents are the active entities that generate, evaluate, and share artifacts.
    They maintain an internal state including their preferences (wundt curve),
    memory of past artifacts, and current creative focus.

    Attributes:
        unique_id (int): Unique identifier for the agent.
        knn (Any): The k-Nearest Neighbors instance used for novelty estimation.
        wundt (Any): The WundtCurve instance used for interest evaluation.
        average_interest (float): Moving average of the interest of artifacts seen.
        current_expression (Optional[ExpressionNode]): The agent's current creative output.
        current_artifact_id (Optional[int]): ID of the current artifact.
        current_interest (float): The interest level of the current artifact.
        artifact_memory (List[Dict]): Short-term memory of artifacts encountered.
        self_eval_history (List[float]): History of self-evaluation scores.
        other_eval_history (List[float]): History of evaluations of others' work.
        gen_depth (int): Maximum depth for generated expression trees.
        preferred_novelty (float): The optimal novelty value for this agent.
        alpha (float): Learning rate for updating average interest.
    """
    unique_id: int
    knn: Any  # kNN instance — personal feature repository (3.3.1)
    wundt: Any # WundtCurve instance — hedonic evaluator (3.3.4)
    # Cumulative interest S_i, updated via EMA (3.4)
    average_interest: float = 0.0
    current_expression: Optional[ExpressionNode] = None
    current_artifact_id: Optional[int] = None
    current_creator_id: Optional[int] = None
    current_interest: float = -1.0 
    # Personal artifact memory for breeding partners (3.2.2)
    artifact_memory: List[Dict] = field(default_factory=list) 
    # Rolling evaluation histories — used by StatsTracker for
    # threshold computation (3.4)
    self_eval_history: List[float] = field(default_factory=list)
    other_eval_history: List[float] = field(default_factory=list)
    
    # DEVIATION(paper 3.4): Hall of fame is not described in paper.
    # Paper: No mechanism for retaining top artifacts.
    # Code: Keeps top-N artifacts by interest for hedonic retreat
    #       in the extended boredom mode.
    hall_of_fame: List[tuple] = field(default_factory=list)
    max_fame_size: int = 10

    # Parameters
    gen_depth: int = 5
    # Preferred novelty ~ N(0.5, 0.155), clipped [0,1] (3.3.4)
    # Still testing this as there's a behaviour where novelty=1.0 might not be interpreted correctly
    # as they nearly always return 0.0 interest, which might not be the intended behaviour, especially in a cold start
    preferred_novelty: float = 0.5
    # Exponential moving alpha decay for cumulative interest (3.4)
    alpha: float = 0.35 

    def update_hall_of_fame(self, artifact_content: Any, interest: float):
        """
        Updates the agent's 'Hall of Fame' with high-interest artifacts.
        Keeps only the top N items sorted by interest.

        DEVIATION(paper 3.4): Hall of fame not described in paper.
        Used by extended boredom mode for hedonic retreat.
        """
        entry = (interest, artifact_content)
        self.hall_of_fame.append(entry)

        # Sort by interest descending and keep top N
        self.hall_of_fame.sort(key=lambda x: x[0], reverse=True)
        if len(self.hall_of_fame) > self.max_fame_size:
            self.hall_of_fame = self.hall_of_fame[:self.max_fame_size]


# DIFI: Interaction component — the scheduler orchestrates the
# flow of artifacts between agents (Field evaluation, 3.4).
class Scheduler(abc.ABC):
    """
    Abstract base class for the simulation scheduler.

    The scheduler controls the flow of time and the activation of agents
    within the simulation.
    """
    @abc.abstractmethod
    def step(self):
        """
        Advances the simulation by one step.
        """
        pass

class Logger(abc.ABC):
    """
    Abstract base class for data logging.

    Defines the interface for recording simulation events and statistics
    to various outputs (CSV, TensorBoard, etc.).
    """
    @abc.abstractmethod
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """
        Logs a specific event.

        Args:
            event_type (str): The category of the event (e.g., 'generation', 'share').
            data (Dict[str, Any]): Key-value pairs of data associated with the event.
        """
        pass

    @abc.abstractmethod
    def close(self):
        """
        Finalizes the logging process, flushing buffers and closing files.
        """
        pass
