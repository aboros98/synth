from typing import Dict, List, Union

import datasets
import pandas as pd
from datasets.dataset_dict import DatasetDict

from .io_utils import load_json, load_jsonl, load_txt
from .text_utils import text_to_list


def load_dataset(dataset_name: str) -> List[str]:
    """
    Load a dataset from the Hugging Face datasets library or from a local file.

    Args:
        dataset_name (str): the name of the dataset to load

    Returns:
        List[str]: the dataset
    """
    try:
        dataset = datasets.load_dataset(dataset_name)
    except Exception:
        dataset = load_local_file(dataset_name)

    instruction_key = find_instructions_key(dataset)

    if isinstance(dataset, pd.DataFrame):
        instructions = dataset[instruction_key].tolist()

    elif isinstance(dataset, DatasetDict):
        instructions = dataset["train"][instruction_key]

        formatted_instructions = []

        for instruction in instructions:
            if instruction.startswith("[") and instruction.endswith("]"):
                instruction = text_to_list(instruction)

            if isinstance(instruction, list):
                instruction = instruction[0]

            formatted_instructions.append(instruction)
        instructions = formatted_instructions

    elif isinstance(dataset, list):
        instructions = []

        for d in dataset:
            if isinstance(d[instruction_key], list):
                instruction = d[instruction_key][0]
            else:
                instruction = d[instruction_key]

            instructions.append(instruction)

    if instruction_key in ["conversations", "conversation"]:
        formated_instructions = []

        for i in instructions:
            for c in i:
                if c["from"] == "human":
                    if "content" in c.keys():
                        formated_instructions.append(c["content"])
                    elif "value" in c.keys():
                        formated_instructions.append(c["value"])

        return formated_instructions
    return instructions


def load_local_file(path: str) -> List[str]:
    """
    Load a file from the local file system.

    Args:
        path (str): the path to the file to load

    Returns:
        List[str]: the file
    """
    if path.endswith(".json"):
        return load_json(path)
    elif path.endswith(".txt"):
        return load_txt(path)
    elif path.endswith(".jsonl"):
        return load_jsonl(path)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)


def find_instructions_key(data: Union[pd.DataFrame, Dict[str, str], DatasetDict]) -> str:
    """
    Find the key in the data that contains instructions.

    Args:
        data (List[str]): the data to search

    Returns:
        str: the key that contains instructions
    """
    possible_keys = ["instructions", "message_1", "instruction", "prompt", "text",
                     "query", "user", "question", "conversation", "conversations"]

    if isinstance(data, pd.DataFrame):
        columns = data.columns

        for key in possible_keys:
            if key in columns:
                return key

    if isinstance(data, DatasetDict):
        for key in possible_keys:
            if key in data.column_names["train"]:
                return key

    if isinstance(data, list):
        for key in possible_keys:
            if key in data[0].keys():
                return key
