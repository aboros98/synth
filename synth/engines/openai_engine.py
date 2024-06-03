import os
from typing import Dict, List

from openai import OpenAI

from .abstract_engine import AbstactEngine


class OpenAIEngine(AbstactEngine):
    @staticmethod
    def from_config(config: Dict[str, str]) -> 'OpenAIEngine':
        """
        Create a OpenAIEngine from a configuration dictionary.

        Args:
            config (Dict[str, str]): A dictionary containing the configuration for the engine.

        Returns:
            GenerationEngine: The engine created from the configuration.
        """
        return OpenAIEngine(engine=config['engine'],
                            model=config['model'],
                            seed=int(config['seed']))

    def __init__(self,
                 engine: OpenAI,
                 model: str,
                 seed: int) -> None:
        """
        Engine for generating completions using the OpenAI API.

        Args:
            engine (OpenAI): The OpenAI engine to use for generation.
            model (str): The model to use for generation.
            seed (int): The seed to use for generation.

        Returns:
            None
        """
        self.engine = self._build_engine(engine)
        self.model = model
        self.seed = seed

    def _build_engine(self, engine: str) -> OpenAI:
        """
        Build the generation engine.

        Args:
            engine (str): The generation engine to build.

        Returns:
            OpenAI: The generation engine.
        """
        if engine == "openai":
            return OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        elif engine == "togetherai":
            return OpenAI(api_key=os.environ.get('TOGETHER_API_KEY'),
                          base_url='https://api.together.xyz/v1')
        else:
            raise ValueError(f"Invalid engine for OpenAI: {engine}. Please choose from 'openai' or 'togetherai'")

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
            message = [{"role": "user", "content": instruction}]

            completion = self.engine.chat.completions.create(
                model=self.model,
                messages=message,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=self.seed)

            outputs.append(completion.choices[0].message.content)

        return outputs
