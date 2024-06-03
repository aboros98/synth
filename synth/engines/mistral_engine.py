import os
from typing import Dict, List

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from .abstract_engine import AbstactEngine


class MistralEngine(AbstactEngine):
    @staticmethod
    def from_config(config: Dict[str, str]) -> 'MistralEngine':
        """
        Create a MistralEngine from a configuration dictionary.

        Args:
            config (Dict[str, str]): A dictionary containing the configuration for the engine.

        Returns:
            GenerationEngine: The engine created from the configuration.
        """
        return MistralEngine(model=config['model'],
                             seed=int(config['seed']))

    def __init__(self, model: str, seed: int) -> None:
        """
        Engine for generating completions using the MistralAI API.

        Args:
            model (str): The model to use for generation.
            seed (int): The seed to use for generation.

        Returns:
            None
        """
        self.engine = MistralClient(api_key=os.environ.get('MISTRAL_API_KEY'))
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
        output = []

        for instruction in instructions:
            completion = self.engine.chat(
                model=self.model,
                messages=[ChatMessage(role="user", content=instruction)],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=self.seed)

            output.append(completion.choices[0].message.content)

        return output
