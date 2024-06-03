INSTRUCTION_ANALYZER = """I want you to act as an instruction analyzer.
Given an instruction, you should recognize its use case and the skills (or knowledge) required for a large language model (LLM) to answer the question.
Generate the use case and skills required without any explanation.
List at most 3 skills, each skill must be transferable, so that LLM can leverage them to answer similar questions.
Avoid using "skill", "knowledge" to describe a skill, and each skill must be concise (2-3 words).
Follow the examples below to analyze the given instruction.

#Example 1#
As a sports commentator, describe the winning play in the final seconds of a championship game.
Use case: creative writing
Skills: role-play, sports

#Example 2#
How to read a large file (> 2T) using python?
Task: code generation
Skills: python

#Example 3#
The method section of your paper is too brief and does not explain how your proposed model works in detail. How can you provide more details of the hierarchical encoder and the cascaded selectors, such as their architectures, inputs, outputs, and parameters?
Task: general knowledge question answering
Skills: academic writing, machine learning

{instruction}"""


INSTRUCTION_WRITER = """I want you to act as an instruction writer.
Your objective is to write {number_of_instructions} instructions that are reasonable, understandable, and respondable by humans.
The generated instructions must be self-contained. If any context or external information is required, it must be included in each corresponding instruction.
The generated instructions should not include any answers, additional text, or titles.

The generated instructions must be diverse while following the constraints below:

Use case of the instructions: {use_case}
Skills required to respond to the instructions: {skills}

Ensure that each instruction:
    - Does not reference any external datasets, tables, or information not included within the instruction.
    - Includes all necessary context or information required to respond. If the context or information is needed, generate it within the instruction.
    - Is clear and specific without needing additional clarification.

Generate the instructions without answering and without any additional text in numbered bullet points:"""


SINGLE_INSTRUCTION_WRITER = """I want you to act as an instruction writer.
Your objective is to write one instruction that must be reasonable and must be understood and responded to by a human.
The generated instruction must be self-contained. If any context or external information is required, it must be included within the instruction.
The generated instruction should not include any answer, additional text or title.
The generated instruction should be diverse enough while following the constraint below:

Use case of the instruction: {use_case}
Skill required to respond to the instruction: {skills}

Generate the instruction without answering and any additional text.

Ensure that each instruction:
    - Does not reference any external datasets, tables, or information not included within the instruction.
    - Includes all necessary context or information required to respond. If the context or information is needed, generate it within the instruction.
    - Is clear and specific without needing additional clarification.

Use the following format to generate the instruction:
Instruction: <instruction>"""


ACTIONS_GENERATOR = """I want you to act as an instruction judge with domain expertise.
Your job is to generate {number_of_rubrics} domain-specific rubrics to assess the difficulty and complexity based on the use case of the instruction and the skills required to respond to it.
The generated rubrics should be clear, concise, unambiguous, and written in phrases or sentences. Do not add any title to the rubrics besides the `rubric` itself.

Based on the generated rubrics, generate corresponding actions to improve an instruction by making it more challenging.
The generated actions should not request any physical actions or require any external data, materials, or information (e.g., datasets, tables, graphs).

The use case of the instruction: {use_case}
The skills required to solve the instruction: {skills}

Please use the following format to generate the rubrics and the actions:
Rubric: <rubric>
Action: <action>

Generate the domain-specific rubrics and the corresponding actions without any explanation in numbered bulletin points:"""


SINGLE_ACTION_GENERATOR = """I want you to act as an instruction judge with domain expertise.
Your job is to generate a domain-specific rubric to assess the difficulty and complexity based on the use case of the instruction and the skills required to respond to it.
The generated rubric should be clear, concise, unambiguous, and written in a phrase or sentence. Do not add any title to the rubric besides the `rubric` itself.

Based on the generated rubric, generate corresponding action to improve an instruction by making it more challenging.
The generated action should not request any physical actions or require any external data, materials, or information (e.g., datasets, tables, graphs).

The use case of the instruction: {use_case}
The skills required to solve the instruction: {skills}

Use the following format to generate the rubric and the action:
Rubric: <rubric>
Action: <action>

Generate the domain-specific rubric and the corresponding action without any explanation in a numbered bullet point:"""


INSTRUCTION_IMPROVER = """I want you to act as an instruction improver with domain expertise.
Your job is to make the given instruction more challenging by following the provided improving action item. The generated instruction must be reasonable, self-consistent, and self-contained.
Ensure that all necessary information and context from the original instruction are retained.
Do not directly copy words or phrases from the action.

Output only the improved instruction without any explanation or additional information.

Improving action: {action}
Input instruction: {input_instruction}

Improved instruction:"""


CONTRASTIVE_FILTERING = """You are a helpful and precise assistant for checking the quality of the answer.

{instruction}
[The Start of Assistant 1's Answer]
{answer_1}
[The End of Assistant 1's Answer]
[The Start of Assistant 2's Answer]
{answer_2}
[The End of Assistant 2's Answer]

We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
Please only output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.
Please avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""


RUBRIC_AND_ACTION_EXTRACTION_PROMPT = """Your task is to extract the rubrics and the actions from the given text.
Please do not use any markdown. Each rubric must be followed by its corresponding action.
Please use the following format:
<start_of_the_example>
Rubric: <rubric_content>
Action: <action_content>

Rubric: <rubric_content>
Action: <action_content>
<end_of_the_example>

Here is the text: {text}"""


INSTRUCTION_ANSWER_REWARD = """Review the user’s question and the corresponding response using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the response is relevant and provides some information related to the user’s inquiry, even if it is incomplete or contains some irrelevant content.
- Add another point if the response addresses a substantial portion of the user’s question, but does not completely resolve the query or provide a direct answer.
- Award a third point if the response answers the basic elements of the user’s question in a useful way, regardless of whether it seems to have been written by an AI Assistant or if it has elements typically found in blogs or search results.
- Grant a fourth point if the response is clearly written from an AI Assistant’s perspective, addressing the user’s question directly and comprehensively, and is well-organized and helpful, even if there is slight room for improvement in clarity, conciseness or focus.
- Bestow a fifth point for a response that is impeccably tailored to the user’s question by an AI Assistant, without extraneous information, reflecting expert knowledge, and demonstrating a high-quality, engaging, and insightful answer.

User: {instruction}

<response>{answer}</response>

After examining the user’s instruction and the response:

- Briefly justify your total score, up to 100 words using the format: "Reasoning: <reasoning>"
- Conclude with the score using the format: “Score: <total points>”

Remember to assess from the AI Assistant perspective, utilizing web search knowledge as necessary. To evaluate the response in alignment with this additive scoring model, we’ll systematically attribute points based on the outlined criteria."""
