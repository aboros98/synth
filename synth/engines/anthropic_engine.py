import os
from typing import Dict, List

from anthropic import Anthropic

from .abstract_engine import AbstactEngine


class AnthropicEngine(AbstactEngine):
    @staticmethod
    def from_config(config: Dict[str, str]) -> 'AnthropicEngine':
        """
        Create a AnthropicEngine from a configuration dictionary.

        Args:
            config (Dict[str, str]): A dictionary containing the configuration for the engine.

        Returns:
            GenerationEngine: The engine created from the configuration.
        """
        return AnthropicEngine(model=config['model'],
                               seed=int(config['seed']))

    def __init__(self, model: str, seed: int) -> None:
        """
        Engine for generating completions using the Anthropic API.

        Args:
            model (str): The model to use for generation.
            seed (int): The seed to use for generation.

        Returns:
            None
        """
        self.engine = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.seed = seed

    def __call__(self,
                 instructions: List[str],
                 temperature: float = 0.,
                 top_p: float = 1.,
                 max_tokens: int = 2048) -> List[str]:
        """
        Generate completions for a list of task prompts.

        Args:
            instructions (List[str]): A list of task prompts.
            temperature (float): The temperature to use for sampling.
            top_p (float): The top_p to use for sampling.
            max_tokens (int): The maximum number of tokens to generate.

        Returns:
            List[str]: A list of completions for the task prompts.
        """
        outputs = []

        for instruction in instructions:
            message = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
            completion = self.engine.messages.create(
                model=self.model,
                messages=message,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens)
            outputs.append(completion.content[0].text)

        return outputs
