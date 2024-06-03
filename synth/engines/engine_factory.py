from .abstract_engine import AbstactEngine
from .anthropic_engine import AnthropicEngine
from .mistral_engine import MistralEngine
from .openai_engine import OpenAIEngine


def build_engine(config: dict) -> AbstactEngine:
    """
    Build an engine from a string.

    Args:
        engine (str): The engine to build.

    Returns:
        AbstactEngine: The engine built from the string.
    """
    if config["engine"] == "anthropic":
        return AnthropicEngine.from_config(config)
    elif config["engine"] == "openai":
        return OpenAIEngine.from_config(config)
    elif config["engine"] == "togetherai":
        return OpenAIEngine.from_config(config)
    elif config["engine"] == "mistralai":
        return MistralEngine.from_config(config)
    else:
        raise ValueError(f"Invalid engine: {config['engine']}. Please choose from \
                         'anthropic', 'openai', 'togetherai', or 'mistralai'")
