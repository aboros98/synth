import random
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np


def sample_random_action_and_rubric(rubrics: List[str],
                                    actions: List[str],
                                    n_instructions: int,
                                    prev_samples: Optional[List[int]] = None,
                                    decay_factor: Optional[float] = 0.8) -> Tuple[List[str], List[str], List[int]]:
    """
    Sample a random action and rubric from the provided lists.

    Args:
        rubrics (List[str]): the list of rubrics
        actions (List[str]): the list of actions
        n_instructions (int): the number of instructions to sample
        prev_samples (Optional[List[int]]): the indices of previously sampled instructions
        decay_factor (Optional[float]): the decay factor for the sampling weights

    Returns:
        Tuple[List[str], List[str], List[int]]: the sampled rubric, action, and indices
    """
    sampling_weights = [1] * len(actions)

    if prev_samples is not None and len(prev_samples) > 0:
        sample_counts = Counter(prev_samples)

        for idx, count in sample_counts.items():
            sampling_weights[idx] *= decay_factor ** count

    indices = random.choices(range(len(actions)), k=n_instructions, weights=sampling_weights)

    action = np.array(actions)[indices].tolist()
    rubric = np.array(rubrics)[indices].tolist()

    return rubric, action, indices
