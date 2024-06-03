from typing import Dict, List, Optional, Tuple

from synth.engines.abstract_engine import AbstactEngine
from synth.utils.text_utils import (extract_digits, extract_instructions,
                                    extract_reasoning_and_score,
                                    extract_rubric_action, extract_task_skills)

from .prompts import (ACTIONS_GENERATOR, CONTRASTIVE_FILTERING,
                      INSTRUCTION_ANALYZER, INSTRUCTION_ANSWER_REWARD,
                      INSTRUCTION_IMPROVER, INSTRUCTION_WRITER,
                      RUBRIC_AND_ACTION_EXTRACTION_PROMPT,
                      SINGLE_ACTION_GENERATOR, SINGLE_INSTRUCTION_WRITER)


def analyze_instructions(instruction: str,
                         engine: AbstactEngine,
                         generation_config: dict) -> List[dict]:
    """
    Analyze an instruction to extract the task and skills.

    Args:
        instruction (str): The instruction to analyze.
        engine (AbstactEngine): The engine to use for generation.
        generation_config (dict): The configuration for the generation engine.

    Returns:
        List[dict]: A list of dictionaries containing the task and skills for each instruction.
    """
    instruction = INSTRUCTION_ANALYZER.format(instruction=instruction)
    analyzed_instruction = engine([instruction], **generation_config)

    return extract_task_skills(analyzed_instruction[0])


def generate_instructions(number_of_instructions: int,
                          use_case: str,
                          skills: str,
                          engine: AbstactEngine,
                          generation_config: dict) -> List[str]:
    """
    Generate instructions for a given use case and skills.

    Args:
        number_of_instructions (int): The number of instructions to generate.
        use_case (str): The use case for the instructions.
        skills (str): The skills required for the instructions.
        engine (AbstactEngine): The engine to use for generation.
        generation_config (dict): The configuration for the generation engine.

    Returns:
        List[str]: A list of generated instructions.
    """
    if number_of_instructions > 1:
        instructions = INSTRUCTION_WRITER.format(number_of_instructions=number_of_instructions,
                                                 use_case=use_case,
                                                 skills=skills)
    else:
        instructions = SINGLE_INSTRUCTION_WRITER.format(use_case=use_case, skills=skills)

    generated_instructions = engine([instructions], **generation_config)

    return extract_instructions(generated_instructions[0])


def generate_actions(number_of_rubrics: int,
                     use_case: str,
                     skills: str,
                     engine: AbstactEngine,
                     generation_config: dict) -> Tuple[List[str], List[str]]:
    """
    Generate actions for a given use case.

    Args:
        number_of_actions (int): The number of actions to generate.
        use_case (str): The use case for the actions.
        engine (AbstactEngine): The engine to use for generation.
        generation_config (dict): The configuration for the generation engine.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the rubrics and actions.
    """
    if number_of_rubrics > 1:
        instructions = ACTIONS_GENERATOR.format(number_of_rubrics=number_of_rubrics,
                                                use_case=use_case,
                                                skills=skills)
    else:
        instructions = SINGLE_ACTION_GENERATOR.format(use_case=use_case, skills=skills)

    generated_actions = engine([instructions],
                               **generation_config)

    rubrics, actions = extract_rubric_action(generated_actions[0])

    if not rubrics or not actions:
        instructions = RUBRIC_AND_ACTION_EXTRACTION_PROMPT.format(text=generated_actions[0])
        generated_actions = engine([instructions], **generation_config)

        rubrics, actions = extract_rubric_action(generated_actions[0])

    if number_of_rubrics == 1 and len(rubrics) > 1:
        rubrics = rubrics[0]
        actions = actions[0]

    return rubrics, actions


def improve_instructions(instructions: List[str],
                         actions: List[str],
                         engine: AbstactEngine,
                         generation_config: dict) -> List[str]:
    """
    Improve a list of instructions by following a given action.

    Args:
        Instructions (List[str]): The instruction to improve.
        actions (List[str]): The action to improve the instruction.
        engine (AbstactEngine): The engine to use for generation.
        generation_config (dict): The configuration for the generation engine.

    Returns:
        List[str]
    """
    instructions = [instructions] if isinstance(instructions, str) else instructions
    actions = [actions] if isinstance(actions, str) else actions

    instructions = [INSTRUCTION_IMPROVER.format(input_instruction=instruction, action=action)
                    for instruction, action in zip(instructions, actions)]
    instructions = engine(instructions, **generation_config)
    instructions = [i.strip() for i in instructions]

    return instructions


def instruction_answer(instructions: str, engine: AbstactEngine, generation_config: dict) -> str:
    """
    Generate an answer to a given instruction.

    Args:
        instruction (str): The instruction to answer.
        engine (AbstactEngine): The engine to use for generation.
        generation_config (dict): The configuration for the generation engine.

    Returns:
        str: The answer to the instruction.
    """
    instructions = instructions if isinstance(instructions, list) else [instructions]
    answers = engine(instructions, **generation_config)

    return answers


def contrastive_filtering(instruction: str,
                          target_model_answer: str,
                          strong_model_answer: str,
                          engine: AbstactEngine,
                          generation_config: dict) -> Tuple[int, int]:
    """
    Filter out instructions that are not contrastive.

    Args:
        instructions (List[str]): A list of instructions to filter.
        engine (AbstactEngine): The engine to use for generation.
        generation_config (dict): The configuration for the generation engine.

    Returns:
        Tuple[int, int]: The scores for the two answers.
    """
    contrastive_instructions = engine([CONTRASTIVE_FILTERING.format(instruction=instruction,
                                                                    answer_1=target_model_answer,
                                                                    answer_2=strong_model_answer)],
                                      **generation_config)
    target_score1, strong_score1 = extract_digits(contrastive_instructions[0])

    # We do this to mitigate the effect of the order of the answers
    contrastive_instructions = engine([CONTRASTIVE_FILTERING.format(instruction=instruction,
                                                                    answer_1=strong_model_answer,
                                                                    answer_2=target_model_answer)],
                                      **generation_config)
    strong_score2, target_score2 = extract_digits(contrastive_instructions[0])

    if target_score1 is None or target_score2 is None or strong_score1 is None or strong_score2 is None:
        return None, None

    target_score = (target_score1 + target_score2) / 2
    strong_score = (strong_score1 + strong_score2) / 2

    return target_score, strong_score


def rank_instruction_with_judge(instruction: str,
                                answer: str,
                                engine: AbstactEngine,
                                generation_config: dict) -> Tuple[str, float]:
    """
    Generate the final instructions and judge them.

    Args:
        instruction (str): The instruction to generate.
        answer (str): The answer to the instruction.
        engine (AbstactEngine): The engine to use for generation.
        generation_config (dict): The configuration for the generation engine.

    Returns:
        Tuple[str, float]: The final instruction and its score.
    """
    instruction = engine([INSTRUCTION_ANSWER_REWARD.format(instruction=instruction, answer=answer)],
                         **generation_config)
    reasoning, score = extract_reasoning_and_score(instruction[0])

    if reasoning is None or score is None:
        return None, None

    return reasoning, score


def build_final_dataset(all_generated_data: dict,
                        judge_model_name: Optional[str],
                        judge_engine: Optional[AbstactEngine] = None,
                        judge_model_config: Optional[dict] = None) -> List[Dict[str, str]]:
    """
    Build the final instruction dataset.

    Args:
        all_generated_data (dict): The generated data.
        judge_model_name (Optional[str]): The name of the judge model.
        judge_engine (Optional[AbstactEngine], optional): The engine to use for the judge model. Defaults to None.
        judge_model_config (Optional[dict], optional): The configuration for the judge model. Defaults to None.

    Returns:
        List[Dict[str, str]]: The final instruction dataset.
    """
    final_instruction_dataset = []

    for data in all_generated_data:
        instructions = data["improved_instructions"]

        for instruction in instructions:
            output_dict = {}
            output_dict["instruction"] = instruction["improved_instruction"]

            strong_answer_score = instruction["strong_score"]
            target_answer_score = instruction["target_score"]

            if strong_answer_score >= target_answer_score:
                output_dict["answer"] = instruction["strong_answer"]
                output_dict["model"] = data["strong_model"]
                output_dict["contrastive_score"] = instruction["strong_score"]
            else:
                output_dict["answer"] = instruction["target_answer"]
                output_dict["model"] = data["target_model"]
                output_dict["contrastive_score"] = instruction["target_score"]

            if judge_engine is not None:
                judge_reason, judge_score = rank_instruction_with_judge(instruction=output_dict["instruction"],
                                                                        answer=output_dict["answer"],
                                                                        engine=judge_engine,
                                                                        generation_config=judge_model_config["generation_config"])

                if judge_reason is not None:
                    output_dict["judge_instruction_score"] = judge_score
                    output_dict["judge_reason"] = judge_reason
                    output_dict["judge_model_name"] = judge_model_name

            output_dict["topic"] = data["task"]
            output_dict["subtopic"] = data["skills"]

            final_instruction_dataset.append(output_dict)

    return final_instruction_dataset
