import ast
import re
from typing import Dict, List, Tuple


def extract_task_skills(input_string: str) -> Dict[str, str]:
    """
    Extract the task and skills from a string.

    Args:
        input_string (str): the string to extract the task and skills from

    Returns:
        Dict[str, str]: a dictionary containing the task and skills
    """
    tasks = []
    skills = []

    sections = re.split(r'(?:Task|Use case|Tasks|Use cases):\s*', input_string, flags=re.IGNORECASE)

    for section in sections[1:]:
        try:
            task_part, skills_part = re.split(r'(?:Skills|Skill|Needed skills|Needed skill):\s*', section, flags=re.IGNORECASE)
            task = task_part.strip()
            skill = skills_part.strip()

            tasks.append(task)
            skills.append(skill)
        except Exception:
            tasks = []
            skills = []

            break

    if not tasks:
        pattern = (
            r"(?:Use case|Task|Tasks|Use cases):\s*(?P<task>[^\n]+)\s*\n\s*"
            r"(?:Skills|Skill|Needed skills|Needed skill):\s*(?P<skills>[^\n]+)"
        )

        matches = re.finditer(pattern, input_string, re.IGNORECASE)

        for match in matches:
            task = match.group('task').strip()
            skill = match.group('skills').strip()

            tasks.append(task)
            skills.append(skill)

    if not tasks:
        return None, None

    return tasks[0], skills[0]


def extract_instructions(input_string: str) -> List[str]:
    """
    Extract the instructions from a string.

    Args:
        input_string (str): the string to extract the instructions from

    Returns:
        List[str]: a list of instructions
    """
    splits = input_string.replace("*", "").split(":\n\n")

    if len(splits) > 2:
        input_string = ""

        for split in splits[1:]:
            input_string += split
    elif len(splits) == 2:
        input_string = splits[-1]
    elif len(splits) < 2:
        input_string = splits[0]

    filtered_instructions = []

    sections = input_string.split("Instruction ")
    for section in sections[1:]:
        try:
            instruction_number, rest = section.split(":", 1)
            instruction_text = rest.strip()

            filtered_instructions.append(f"Instruction {instruction_number.strip()}: {instruction_text}")
        except Exception:
            filtered_instructions = []
            break

    # Backup method: split by "Task" if no instructions found
    if not filtered_instructions:
        sections = input_string.split("Task ")
        for section in sections[1:]:
            try:
                task_number, rest = section.split(":", 1)
                task_text = rest.strip()
                filtered_instructions.append(f"Task {task_number.strip()}: {task_text}")
            except Exception:
                filtered_instructions = []
                break

    # Further backup method: regex if no instructions or tasks found
    if not filtered_instructions:
        pattern = (
            r"(?:"
            r"(?:(?:Instruction(?:s)?|Task(?:s)?|Question(?:s)?|\d+|\-|\*|\•)\.?\s*\d*[.:]*\s*(.+?))"
            r"(?:\n(?=\d+|Instruction(?:s)?|Task(?:s)?|Question(?:s)?|\-|\*|\•|$)|$)"
            r"|"
            r"•\s*(.*?)(?=\n•|$)"
            r")"
        )

        matches = re.findall(pattern, input_string, re.DOTALL)

        # Extract non-empty matches and strip whitespace
        instructions = [match[0] or match[1] for match in matches if (match[0].strip() or match[1].strip())]

        # Filter out lines that are too short, look like headers, or contain only the keyword
        filtered_instructions = []
        for instruction in instructions:
            instruction = instruction.strip()

            if len(instruction) > 10 and not re.match(r'^(Instructions?|Tasks?|Questions?|Items?)\s*[:]*$', instruction, re.IGNORECASE):
                filtered_instructions.append(instruction)

    return filtered_instructions


def extract_rubric_action(input_string: str) -> Tuple[List[str], List[str]]:
    """
    Extract the rubric and action from a string.

    Args:
        input_string (str): the string to extract the rubric and action from

    Returns:
        Tuple[List[str], List[str]]: a tuple of two lists containing rubrics and actions
    """
    # Remove any unnecessary characters
    input_string = input_string.replace("*", "").strip()

    rubrics = []
    actions = []

    sections = input_string.lower().split("rubric ")
    for section in sections[1:]:
        try:
            _, rest = section.split(":", 1)
            rubric_text, action_text = rest.lower().split("action:", 1)

            rubric_text = rubric_text.strip()
            action_text = action_text.strip()

            rubrics.append(rubric_text)
            actions.append(action_text)
        except Exception:
            # Force to go to backup method
            rubrics = []
            actions = []

            break

    # Backup method if no rubrics or actions are found
    if not rubrics or not actions:
        pattern_numbered = r'(?i)(\d+\.\s*Rubric(?:s)?(?: for .*?)?(?:[.:])?\s*(?P<rubric>.*?))\s*Actions?(?: to .*?)?(?:[.:])?\s*(?P<action>.*?)(?=\n\d+\.|\Z)'
        pattern = r'Rubric:\s*(.*?)\nAction:\s*(.*?)(?=\nRubric:|\Z)'

        # First, try to match using the numbered pattern
        matches = re.finditer(pattern_numbered, input_string, re.DOTALL)
        for match in matches:
            rubric_text = match.group('rubric').strip()
            action_text = match.group('action').strip()

            rubric_entries = re.split(r'\n\s*-\s*|\n\s*Action:|\n\s*- Action:', rubric_text)
            action_entries = re.split(r'\n\s*-\s*|\n\s*Action:|\n\s*- Action:', action_text)

            # Removing any leading or trailing whitespace from the first entry
            rubric_entries[0] = rubric_entries[0].strip()
            action_entries[0] = action_entries[0].strip()

            # Creating a list of rubrics and actions
            rubrics.extend([entry.strip().replace("- ", "") for entry in rubric_entries if entry.strip()])
            actions.extend([entry.strip().replace("- ", "") for entry in action_entries if entry.strip()])

        # If no matches found using the numbered pattern, use the non-numbered pattern
        if not rubrics and not actions:
            matches = re.finditer(pattern, input_string, re.DOTALL)
            for match in matches:
                rubric_text = match.group(1).strip()
                action_text = match.group(2).strip()
                rubrics.append(rubric_text)
                actions.append(action_text)

    return rubrics, actions


def extract_digits(input_string: str) -> Tuple[int, int]:
    """
    Extract two digits from a string.

    Args:
        input_string (str): the string to extract the digits from

    Returns:
        Tuple[int, int]: a tuple containing the two digits
    """
    input_string = input_string.strip()
    pattern = r'(?:\b((?:10|[0-9])(?:\.\d+)?)\s*[,/\s]\s*((?:10|[0-9])(?:\.\d+)?)\b)|(?:\b(?:Assistant\s+\d+\s+a\s+|\D+\s+)(\d+)\D+(?:Assistant\s+\d+\s+a\s+|\D+\s+)(\d+)\b)'
    match = re.search(pattern, input_string, re.IGNORECASE)

    if match:
        if match.group(1) and match.group(2):
            digit1, digit2 = map(float, match.groups()[:2])

            return int(digit1), int(digit2)
        elif match.group(3) and match.group(4):
            score1, score2 = map(int, match.groups()[2:])

            return score1, score2

    return None, None


def text_to_list(input_string: str) -> List[str]:
    """
    Convert a string to a list of strings.

    Args:
        input_string (str): the string to convert to a list

    Returns:
        List[str]: a list of strings
    """
    try:
        return ast.literal_eval(input_string)
    except Exception:
        return input_string


def extract_reasoning_and_score(input_string: str) -> Tuple[str, float]:
    """
    Extract the reasoning and score from a string.

    Args:
        input_string (str): the string to extract the reasoning and score from

    Returns:
        Tuple[str, float]: a tuple containing the reasoning as a string and the score as a float
    """
    # Patterns to capture the reasoning and score
    reasoning_pattern = re.compile(r'Reasoning:\s*(.*?)\n\n', re.DOTALL | re.IGNORECASE)
    score_pattern = re.compile(r'Score:\s*([-+]?[0-9]*\.?[0-9]+)\s*(?:points?)?', re.DOTALL | re.IGNORECASE)

    # Find matches for reasoning and score
    reasoning_match = reasoning_pattern.search(input_string)
    score_match = score_pattern.search(input_string)

    # Extract and clean the reasoning text
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    # Extract and convert the score to a float
    score = float(score_match.group(1))

    return reasoning, score
