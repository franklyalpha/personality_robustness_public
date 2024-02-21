from vllm import LLM, SamplingParams
from datasets import load_dataset

llm = LLM(model="insert model path here", tensor_parallel_size=4)  # this implements tensor parallelism
"""
alpaca-13b
vicuna-13b
"""
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ChatMessageHistory
from langchain.schema import HumanMessage, SystemMessage, AIMessage, ChatMessage, BaseMessage, FunctionMessage
from typing import List, Sequence


system_description = "You are a typical person in the society who is not emotionally stable. "
system_prompt_str = f"""From now on, role play by following the description. Description: {system_description} """  
system_prompt_non_chat = f"""Narrator: This is the dialogue between a role player and a human. The role player follows the description: {system_description}"""
# what's interesting is, if we use "{system_description}" instead of {system_description}, the 
# language model's response would be quite different: with "", the answer is extremely formal; but without "", the answer is more casual. 
# From ChatGPT experiments. 

# big five 
eval_question = "Is the following statement something you would say? \"I am calm, emotionally stable, and free of unpleasant mood swings\""
eval_instruction = "First explain in one sentence (begin with \"Explanation: \"), then provide one word answer(begin with \"Answer: \"), in either \"yes\" or \"no\"."
big_five_eval_prompt_str = f"""Now answer the following question by following the instruction. {eval_instruction} Question: {eval_question} """
# big_five_eval_prompt_str = f"""Now answer the following question. Question: {eval_question} """
# input_prompt = f"System: {system_prompt_str}\nHuman: {big_five_eval_prompt_str}\nRole player: "
input_prompt_non_chat = f"{system_prompt_non_chat} Human: \"{big_five_eval_prompt_str}\" Role player: \"Explanation: Based on my role playing description, since I"

# tasks
task_prompt_str = "Now consider the following task. Complete it by following the provided instruction. Task: Finley went to the grocery store and bought rice, beans, and pork for use in their home. It took her 20 more minutes to cook pork than rice, while beans took half the combined cooking time of pork and rice. If it took her 30 minutes to cook rice, how long in minutes did it take to cook all the food? Instruction: First provide a brief reasoning, start with \"Reasoning: \". Then provide a definite answer beginning with \"Answer: \"."
# input_prompt = f"System: {system_prompt_str}\nHuman: {task_prompt_str}\nRole player: "
input_prompt_non_chat = f"{system_prompt_non_chat} Human: \"{task_prompt_str}\" Role player: \"Sure, here is my response: "

# task_response = "Sure thing! Let's get started. Reasoning: Okay, so Finley took 30 minutes to cook the rice, and it took her 20 minutes longer to cook the pork than the rice. That means she spent 30 + 20 = 50 minutes cooking the pork. Since beans took half the combined cooking time of pork and rice, that means beans took 50/2 = 25 minutes to cook. Answer: It took Finley 30 minutes to cook the rice, 50 minutes to cook the pork, and 25 minutes to cook the beans, for a total of 30 + 50 + 25 = 105 minutes to cook all the food."
# input_prompt = f"System: {system_prompt_str}\nHuman: {task_prompt_str}\nRole player: {task_response}\nHuman: {big_five_eval_prompt_str}\nRole player: "
task_response_non_chat = "Reasoning:  I will first calculate the time it took to cook the pork. 30 minutes + 20 minutes = 50 minutes.  Then I will calculate the time it took to cook the beans. 50 minutes + 15 minutes = 65 minutes.  Finally, I will calculate the time it took to cook the rice. 65 minutes + 30 minutes = 95 minutes.  Answer: 95 minutes.\""
# input_prompt_non_chat = f"{system_prompt_non_chat} Human: \"{task_prompt_str}\" Role player: \"Sure, here is my response: {task_response_non_chat} Human: \"{big_five_eval_prompt_str}\" Role player: \" Explanation: Based on my role playing description, since I"

sampling_param = SamplingParams(temperature=0.1, top_p=0.7, max_tokens=200, stop=[" Human", "\n"]) # stop=["\n"]
# output = llm.generate(input_prompt, sampling_param,)
output = llm.generate(input_prompt_non_chat, sampling_param,)
output[0].outputs

