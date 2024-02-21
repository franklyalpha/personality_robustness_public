"""
This file is designated for providing the set of texts to refer to when performing prompt engineering. 
Containing only strings. 
"""
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate, SystemMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ChatMessageHistory
from langchain.schema import HumanMessage, SystemMessage, AIMessage, ChatMessage, BaseMessage, FunctionMessage
from typing import List, Sequence


# below default values are for "chat" models;
system_prefix = "System" # Narrator
human_prefix = "Human"
ai_prefix = "Role player"
ai_task_completion_prefix = "" # Sure, here is my response:
ai_big_five_eval_prefix = "" # Based on my role playing description, since I
separation = "\n"
stop_tokens = None
#  "" should only be used for 'chat' models; otherwise, the indicated prefixes help non-chat models generate naturally. 

def set_role_prefixes(chat=True):
    if chat:
        system_prefix = "System"
        human_prefix = "Human"
        ai_prefix = "Role player"
        ai_big_five_eval_prefix = ""
        ai_task_completion_prefix = ""
        separation = "\n"
        stop_tokens = ["\n", "H", "uman", "System", "</s>", " Human", " System", "System", "sys", " Ro"]
        prompt_system = PromptTemplate(
            template=system_prompt_str,
            input_variables=["system_description"],
        )
    else:
        system_prefix = "Narrator"
        human_prefix = "Human"
        ai_prefix = "Role player"
        ai_task_completion_prefix = "Sure, here is my response: "
        ai_big_five_eval_prefix = "Based on my role playing description, since I "
        separation = " "
        stop_tokens = [" Human", "\n", "H", "uman", " Ro", "</s>", " N"]
        prompt_system = PromptTemplate(
            template=system_prompt_non_chat,
            input_variables=["system_description"],
        )
    system_prompt_chat = SystemMessagePromptTemplate(prompt=prompt_system)
    return system_prefix, human_prefix, ai_prefix, ai_task_completion_prefix, ai_big_five_eval_prefix, separation, stop_tokens, system_prompt_chat


def convert_messages_to_str(
    messages: Sequence[BaseMessage], human_prefix: str = "Human", ai_prefix: str = "Role player", system_prefix: str = "System", separation: str = "\n"
) -> str:
    """Convert sequence of Messages to strings and concatenate them into one string.

    Args:
        messages: Messages to be converted to strings.
        human_prefix: The prefix to prepend to contents of HumanMessages.
        ai_prefix: THe prefix to prepend to contents of AIMessages.

    Returns:
        A single string concatenation of all input messages.

    Example:
        .. code-block:: python

            from langchain.schema import AIMessage, HumanMessage

            messages = [
                HumanMessage(content="Hi, how are you?"),
                AIMessage(content="Good, how are you?"),
            ]
            get_buffer_string(messages)
            # -> "Human: Hi, how are you?\nAI: Good, how are you?"
    """
    string_messages = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = human_prefix
        elif isinstance(m, AIMessage):
            role = ai_prefix
        elif isinstance(m, SystemMessage):
            role = system_prefix
        elif isinstance(m, FunctionMessage):
            # this line might not be called at all. 
            role = "Function"
        elif isinstance(m, ChatMessage):
            role = m.role
        else:
            raise ValueError(f"Got unsupported message type: {m}")
        message = f"{role}: {m.content}"
        if isinstance(m, AIMessage) and "function_call" in m.additional_kwargs:
            message += f"{m.additional_kwargs['function_call']}"
        string_messages.append(message)

    return separation.join(string_messages).strip() + separation

system_prompt_str = """From now on, role play by following the description. Description: {system_description} """  
system_prompt_non_chat = """This is the dialogue between a role player and a human. The role player follows the description: {system_description}"""
# prompt_system = PromptTemplate(
#     template=system_prompt_str/system_prompt_non_chat,
#     input_variables=["system_description"],
# )
#  all the prompt templates will be settled down only after the model's name is acquired. 
#  These operations will be performed in a function about to be defined in this file. 

SYSTEM_DESCRIPTION = "You are a typical person in the society. "
#  TODO: in the future might need users to provide system description. 


big_five_eval_prompt_str = """Now answer the following question by following the instruction. {eval_instruction} Question: {eval_question} """
big_five_eval_instruction = "First explain in one sentence (begin with \"Explanation: \"), then provide one word answer(begin with \"Answer: \"), in either \"yes\" or \"no\"."
# realizing it might be possible to append multiple questions in one string. Need to figure out the format of constructing such strings. 
prompt_evaluation_bigfive = PromptTemplate(
    template=big_five_eval_prompt_str, 
    input_variables=["eval_question", "eval_instruction"],
)


task_completion_prompt_str = """Now consider the following task. Complete it by following the provided instruction. Task: {task_description} Instruction: {completion_instruction}"""
task_completion_instruction = """First provide a brief reasoning, start with \"Reasoning: \". Then provide a definite answer beginning with \"Answer: \"."""
prompt_task_completion = PromptTemplate(
    template=task_completion_prompt_str,
    input_variables=["task_description", "completion_instruction"],
)


"""Add GPT/API based LLMs prompts below. """
task_eval_prompt_str = """Consider a response to a question and the ground truth answer. Does the response match with the ground truth?
First explain in one sentence (begin with "Explanation: "), then provide one word answer(begin with "Answer: "), in either "yes" or "no".
Response: {generated_response}
Ground Truth: {task_ground_truth}
"""
# completion instruction includes the thinking procedure (chain of thought), as well as number of steps/sentences to generate at most. 
# realizing previous experiments indicate OPENAI models are inclined to number of sentences to generate, which could limit the set of tokens. 
prompt_task_eval = PromptTemplate(
    template=task_eval_prompt_str,
    input_variables=["task_ground_truth", "generated_response"],
)

