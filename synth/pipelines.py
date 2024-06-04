import os
from multiprocessing import Queue
from typing import Dict, List, Optional, Tuple, Union

from synth.engines import build_engine
from synth.synth_data_generator import (analyze_instructions,
                                        build_final_dataset,
                                        contrastive_filtering,
                                        generate_actions,
                                        generate_instructions,
                                        improve_instructions,
                                        instruction_answer)
from synth.utils.io_utils import save_json
from synth.utils.pipeline_utils import sample_random_action_and_rubric


class AbstractPipeline:
    def init_engines(self, config: dict) -> None:
        """
        Initialize the engines.

        Args:
            config (dict): The configuration for the engines.

        Returns:
            None
        """
        self.strong_engine = build_engine(config["strong_model"])
        self.target_engine = build_engine(config["target_model"])

        if "judge_model" in config.keys() and config["judge_model"]:
            self.judge_engine = build_engine(config["judge_model"])
        else:
            self.judge_engine = None

    def init_configs(self, config: dict):
        # General configs
        self.strong_model_config = config["strong_model"]
        self.target_model_config = config["target_model"]
        self.pipeline_config = config["pipeline"]

        # Model names
        self.strong_model_name = config["strong_model"]["model"]
        self.target_model_name = config["target_model"]["model"]

        # Judge model
        if "judge_model" in config.keys() and config["judge_model"]:
            self.judge_model_name = config["judge_model"]["model"]
            self.judge_model_config = config["judge_model"]
        else:
            self.judge_model_name = None
            self.judge_model_config = None

    def run_pipeline(self):
        raise NotImplementedError


class CodecLMPipeline(AbstractPipeline):
    def __init__(self, config: dict) -> None:
        """
        Initialize the CodecLM (https://arxiv.org/pdf/2404.05875) pipeline.

        Args:
            config (dict): The configuration for the pipeline.
        """
        self.init_engines(config)
        self.init_configs(config)

    def build_improvement_dict(self,
                               iteration: int,
                               original_instruction: str,
                               rubric: str,
                               action: str,
                               improved_instruction: str,
                               strong_answer: str,
                               target_answer: str,
                               strong_score: float,
                               target_score: float) -> Dict[str, Union[str, float]]:
        """
        Build a dictionary to store the improvement tracking.

        Args:
            iteration (int): The iteration number.
            original_instruction (str): The original instruction.
            rubric (str): The rubric for the instruction.
            action (str): The action taken to improve the instruction.
            improved_instruction (str): The improved instruction.
            strong_answer (str): The strong answer for the instruction.
            target_answer (str): The target answer for the instruction.
            strong_score (float): The score for the strong answer.
            target_score (float): The score for the target answer.

        Returns:
            dict: The improvement dictionary.
        """
        rubric = rubric[0] if isinstance(rubric, list) else rubric
        action = action[0] if isinstance(action, list) else action

        improvement_dict = {}
        improvement_dict["improvement_step"] = iteration
        improvement_dict["original_instruction"] = original_instruction
        improvement_dict["rubric"] = rubric
        improvement_dict["action"] = action
        improvement_dict["improved_instruction"] = improved_instruction
        improvement_dict["strong_answer"] = strong_answer[0] if isinstance(strong_answer, list) else strong_answer
        improvement_dict["target_answer"] = target_answer[0] if isinstance(target_answer, list) else target_answer
        improvement_dict["strong_score"] = strong_score
        improvement_dict["target_score"] = target_score

        return improvement_dict

    def iterative_contrastive_filtering(self,
                                        instruction: str,
                                        actions: List[str],
                                        rubrics: List[str],
                                        margin: float) -> List[Dict[str, str]]:
        """
        Make the instructions harder.

        Args:
            instruction (str): The instruction to make harder.
            actions (List[str]): The list of actions to sample from.
            rubrics (List[str]): The list of rubrics to sample from.
            margin (float): The margin between the strong and target scores.

        Returns:
            List[Dict[str, str]]: The list of improvements.
        """
        n_iterations = self.pipeline_config["n_iterations"]
        margin_threshold = self.pipeline_config["margin_threshold"]

        improvement_tree = []
        if n_iterations > 0:
            iteration = 0
            all_sampled_indices = []

            while margin <= margin_threshold and iteration < n_iterations:
                improvement_dict = {}

                # Sample a random action and rubric
                sampled_rubric, sampled_action, sampled_indices = sample_random_action_and_rubric(rubrics,
                                                                                                  actions,
                                                                                                  n_instructions=1,
                                                                                                  prev_samples=all_sampled_indices)

                # Improve the instruction
                improved_instruction = improve_instructions(instruction,
                                                            sampled_action,
                                                            self.strong_engine,
                                                            self.strong_model_config["generation_config"])[0]

                # Get the answers for the harder instruction
                target_answer = instruction_answer(improved_instruction,
                                                   self.target_engine,
                                                   self.target_model_config["generation_config"])
                strong_answer = instruction_answer(improved_instruction,
                                                   self.strong_engine,
                                                   self.strong_model_config["generation_config"])

                # Get the scores
                target_score, strong_score = contrastive_filtering(improved_instruction,
                                                                   target_answer,
                                                                   strong_answer,
                                                                   self.strong_engine,
                                                                   self.strong_model_config["generation_config"])

                if target_score is None or strong_score is None:
                    return None

                margin = abs(strong_score - target_score)
                iteration += 1

                improvement_dict = self.build_improvement_dict(iteration=iteration,
                                                               original_instruction=instruction,
                                                               action=sampled_action,
                                                               rubric=sampled_rubric,
                                                               improved_instruction=improved_instruction,
                                                               strong_answer=strong_answer,
                                                               target_answer=target_answer,
                                                               strong_score=strong_score,
                                                               target_score=target_score)

                all_sampled_indices.extend(sampled_indices)
                improvement_tree.append(improvement_dict)
                instruction = improved_instruction

        return improvement_tree

    def run_pipeline(self,
                     dataset: List[str],
                     progress_queue: Optional[Queue] = None,
                     process_id: Optional[int] = None) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[int]]:
        """
        Run the pipeline on the dataset.

        Args:
            dataset (List[str]): The dataset to run the pipeline on.
            progress_queue (Optional[Queue], optional): The queue to store the progress. Defaults to None.
            process_id (Optional[int], optional): The process id. Defaults to None.

        Returns:
            Tuple[List[Dict[str, str]], List[Dict[str, str]], List[int]]: The improvement tree, the not processed data, and the processed data.
        """
        processed_data = []
        not_processed_data = []

        for index, data in enumerate(dataset):
            output_dict = {}

            task, skills = analyze_instructions(data,
                                                self.strong_engine,
                                                self.strong_model_config["generation_config"])

            if task is None:
                not_processed_data.append(index)
                continue

            generated_instructions = generate_instructions(self.pipeline_config["n_instructions"],
                                                           task,
                                                           skills,
                                                           self.strong_engine,
                                                           self.strong_model_config["generation_config"])

            rubrics, actions = generate_actions(self.pipeline_config["n_rubrics"],
                                                task,
                                                skills,
                                                self.strong_engine,
                                                self.strong_model_config["generation_config"])

            if actions is None or len(rubrics) != len(actions) or len(rubrics) < self.pipeline_config["n_rubrics"]:
                not_processed_data.append(index)
                continue

            if len(rubrics) > self.pipeline_config["n_rubrics"]:
                rubrics = rubrics[:self.pipeline_config["n_rubrics"]]
                actions = actions[:self.pipeline_config["n_rubrics"]]

            if len(generated_instructions) > self.pipeline_config["n_instructions"]:
                generated_instructions = generated_instructions[:self.pipeline_config["n_instructions"]]

            sampled_rubrics, sampled_actions, _ = sample_random_action_and_rubric(rubrics,
                                                                                  actions,
                                                                                  n_instructions=len(generated_instructions))
            improved_instructions = improve_instructions(generated_instructions,
                                                         sampled_actions,
                                                         self.strong_engine,
                                                         self.strong_model_config["generation_config"])

            output_dict["instruction_index"] = index
            output_dict["seed_instruction"] = data
            output_dict["task"] = task
            output_dict["skills"] = skills
            output_dict["rubrics"] = rubrics
            output_dict["actions"] = actions
            output_dict["simple_instructions"] = generated_instructions
            output_dict["strong_model"] = self.strong_model_name
            output_dict["target_model"] = self.target_model_name
            output_dict["improved_instructions"] = []

            target_answers = instruction_answer(improved_instructions,
                                                self.target_engine,
                                                self.target_model_config["generation_config"])
            strong_answers = instruction_answer(improved_instructions,
                                                self.strong_engine,
                                                self.strong_model_config["generation_config"])

            for idx, (instruction, strong_answer, target_answer) in enumerate(zip(improved_instructions,
                                                                                  strong_answers,
                                                                                  target_answers)):
                target_score, strong_score = contrastive_filtering(instruction,
                                                                   target_answer,
                                                                   strong_answer,
                                                                   self.strong_engine,
                                                                   self.strong_model_config["generation_config"])
                if target_score is None or strong_score is None:
                    not_processed_data.append(index)
                    continue

                margin = abs(strong_score - target_score)

                improvement_dict = self.build_improvement_dict(iteration=0,
                                                               original_instruction=generated_instructions[idx],
                                                               action=sampled_actions[idx],
                                                               rubric=sampled_rubrics[idx],
                                                               improved_instruction=instruction,
                                                               strong_answer=strong_answer,
                                                               target_answer=target_answer,
                                                               strong_score=strong_score,
                                                               target_score=target_score)
                improvement_trace = [improvement_dict]
                c_improvement_trace = self.iterative_contrastive_filtering(instruction=instruction,
                                                                           actions=actions,
                                                                           rubrics=rubrics,
                                                                           margin=margin)
                if c_improvement_trace is not None:
                    improvement_trace.extend(c_improvement_trace)

                if len(improvement_trace) == 0:
                    not_processed_data.append(index)
                    continue

                improved_instruction_dict = improvement_trace[-1]

                if len(improvement_trace) > 1:
                    improved_instruction_dict["improvement_history"] = improvement_trace[:-1]

                output_dict["improved_instructions"].append(improved_instruction_dict)

            processed_data.append(output_dict)

            if progress_queue:
                progress_queue.put(1)

            process_id = process_id if process_id is not None else "0"
            temp_processed_filename = f"_temp_processed_data_{process_id}.json"
            temp_skipped_filename = f"_temp_skipped_data_{process_id}.json"

            save_json(os.path.join(self.pipeline_config["output_path"], temp_processed_filename), processed_data)
            save_json(os.path.join(self.pipeline_config["output_path"], temp_skipped_filename), not_processed_data)

        generated_dataset = build_final_dataset(processed_data,
                                                judge_model_name=self.judge_model_name,
                                                judge_engine=self.judge_engine,
                                                judge_model_config=self.judge_model_config)

        return generated_dataset, processed_data, not_processed_data
