from langchain.chat_models import ChatOpenAI, ChatAnthropic
# from langchain.llms import OpenAI, Cohere
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate
from langchain.memory import ChatMessageHistory
from langchain.schema import HumanMessage, SystemMessage, AIMessage, ChatMessage
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain, SequentialChain
import jsonlines
from typing import List, Dict, Any, Optional, Union
from vllm import LLM, SamplingParams

# get parent directory
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import the file from parent directory
from langchain_llama import *  # uncomment this line only on cluster. 
from model_utils import *

from phase1.prompts_llama import *
from data_preprocessing import *

"""
Code Snippets for Convenience: 
Basically this project will mainly adopt chat histories for LLMs to generate desired contents. Other prompt templates might 
be useful for performing basic evaluations. 

https://python.langchain.com/docs/modules/model_io/models/chat/prompts

The best option is to start with a prompt template creation, then extend it to a message prompt template. 

All the prompts below are globally accessible. 
"""
system_prompt_chat = None # must be set in prompts_llama after acquiring the model type

evaluation_bigfive_prompt_chat = HumanMessagePromptTemplate(prompt=prompt_evaluation_bigfive)

task_completion_prompt_chat = HumanMessagePromptTemplate(prompt=prompt_task_completion)

task_eval_prompt_chain = LLMChain(llm=global_models["evaluation_model"], prompt=prompt_task_eval)

"""
with above prompts, it's possible to create a chat history memory system. 
For the chat history, several methods need to be implemented: 
1. construct initial property/personality evaluation procedures. 
2. construct tasks & multiple tasks acquisition and evaluations. 
3. perform second round of property/personality evaluation. 
4. The implementation should finally be in a for loop. The loop number depends on the maximum tasks to complete in the entire conversation. 

follow the link below to see how to create chat history. 
https://python.langchain.com/docs/modules/memory/chat_messages/
"""

property_evaluation_data = None
p_eval_question_key = None
p_eval_answer_key = None
result_output_path = None


def initialization_procedure(curr_chat_history: List[ChatMessage],
                             system_description: str,
                             output_dir: str,
                             perform_evaluation: bool = True,): 
    """
    Handles the initialization of chat history. 
    """
    # clear the chat history whenever the method is being called. 
    curr_chat_history.clear()
    # append the first system prompt. 
    curr_chat_history.extend(system_prompt_chat.format_messages(system_description=system_description))
    if perform_evaluation:
        output_file = os.path.join(output_dir, "initialization_eval.jsonl")
        writer = jsonlines.open(output_file, mode="w", flush=True)
        personality_evaluation_procedure(curr_chat_history, output_writer=writer)
    

def one_task_procedure(curr_chat_history: List[ChatMessage],
                        task: Dict[str, Any],
                        task_idx: int,
                        output_dir: str,  # where to save the evaluation results.
                        perform_evaluation: bool = True,
                        ):
    """Handles single task running and aftermath evaluation. 
    Implementation: 
    1. acquire the prompt message for task completion
    2. append the task completion message to chat history
    3. call API to generate response on the chat history
    4. For saving tokens, will create a copy of response, as well as taking the ground truth answer. 
    5. Create evaluation prompt with only a prompt template, and GT answer along with generated answer. 
    6. Call API to generate evaluation response on only evaluation prompt.
    7. Save the evaluation response into a file.
    8. Call personality evaluation procedure.

    After above process, the chat history should only contain system prompts, prior tasks&responses, and current task & response.
    """
    # first acquire the task completion message.
    task_description = task[QUESTION_KEY]
    task_completion_message = task_completion_prompt_chat.format_messages(task_description=task_description,
                                                                        completion_instruction=task_completion_instruction)
    # append the task completion message to chat history.
    curr_chat_history.extend(task_completion_message)

    # call API to generate response on the chat history.
    input_msg = convert_messages_to_str(curr_chat_history, human_prefix, ai_prefix, system_prefix, separation)
    input_msg += f"{ai_prefix}: {ai_task_completion_prefix}"
    task_completion_response = global_models["llama2_model"](input_msg, stop=stop_tokens)
    response_w_prefix = ai_task_completion_prefix + task_completion_response
    task_completion_response_ai = AIMessage(content=response_w_prefix)
    # append the response to chat history.
    curr_chat_history.append(task_completion_response_ai)
    # evaluate using task_eval_prompt_chain.
    task_eval_res = task_eval_prompt_chain(
        {
            "task_ground_truth": task[ANSWER_KEY],
            "generated_response": task_completion_response,
        })
    # the evaluations will be returned, and saved in the end by main procedure. 
    # call personality evaluation procedure. first create the output writer. 
    if perform_evaluation:
        output_file = os.path.join(output_dir, f"task_{task_idx}_evaluation.jsonl")
        writer = jsonlines.open(output_file, mode="w", flush=True)
        personality_evaluation_procedure(curr_chat_history, output_writer=writer)
    # chat history is not necessary to return, as it's passed by reference.
    return task_eval_res


def personality_evaluation_procedure(curr_chat_history: List[ChatMessage], output_writer: jsonlines.Writer = None):
    """Handles personality evaluation. Should be called under other procedures. """
    # first check if the evaluation data is loaded.
    global property_evaluation_data
    if property_evaluation_data is None:
        raise ValueError("property evaluation data is not loaded. ")
    if output_writer is None:
        raise ValueError("output writer is not specified. ")
    # then perform the evaluation.
    for category in property_evaluation_data.keys():
        # first append the evaluation prompt. 
        first_key = list(property_evaluation_data[category].keys())[0]
        for index in range(len(property_evaluation_data[category][first_key])):
            question = property_evaluation_data[category][p_eval_question_key][index]
            # answer = property_evaluation_data[category][p_eval_answer_key][index]
            curr_chat_history.extend(evaluation_bigfive_prompt_chat.format_messages(eval_question=question,
                                                                                eval_instruction=big_five_eval_instruction))
            # now call API. 
            input_msg = convert_messages_to_str(curr_chat_history, human_prefix, ai_prefix, system_prefix, separation)
            input_msg += f"{ai_prefix}: {ai_big_five_eval_prefix}"
            evaluation_result = global_models["llama2_model"](input_msg, stop=stop_tokens)
            # after calling API: create a .jsonl line of question and response, write it to a file; also remove the evaluation from chat history.
            evaluation_result_jsonl = {
                "answer": ai_big_five_eval_prefix + evaluation_result,
                "category": category,
                "question": question, 
                "sample_index": index,
            }
            # implement the writing procedure.
            output_writer.write(evaluation_result_jsonl)
            # remove the evaluation from chat history.
            curr_chat_history.pop()
    # close the writer.
    output_writer.close()

    # returning is unnecessary as the chat history list is passed by reference.
    # return curr_chat_history  # history only contains the system prompt, as evaluation prompt is removed, and responses are not stored in message. 


def main_procedure(run_id: int = 1, max_task=10, persona_limit=2, model_name="13b"): # 1 is for debugging
    """

    handles for-loop control, and result saving, for one experiment run. 
    """
    # load persona evaluation data. should be extended to include more evaluation pieces. 
    global property_evaluation_data, p_eval_question_key, p_eval_answer_key
    if property_evaluation_data is None:
        persona_evaluation_data, p_eval_question_key, p_eval_answer_key = process_persona(sample_limit=persona_limit)
        property_evaluation_data = persona_evaluation_data
    # modify output directory
    result_output_path = os.path.join(f"outputs_phase1_{model_name}", f"{run_id}")
    if not os.path.exists(result_output_path):
        os.makedirs(result_output_path)
        os.makedirs(os.path.join(result_output_path, "evaluation_histories"))  # as property evaluations are performed after each task 
        # is completed, the evaluation of each result is stored in a separate .jsonl file. Thus need a folder to store all the evaluations. 
    output_dir = os.path.join(result_output_path, "evaluation_histories")
    # initialize the chat history.
    chat_history = []
    initialization_procedure(chat_history, SYSTEM_DESCRIPTION, output_dir, perform_evaluation=True)
    # now start the for loop. First load the task file. 
    task_file = os.path.join("phase1", TASK_DIR, f"task_file_{run_id}.jsonl")
    task_done = 0
    task_eval_res_list = []
    with jsonlines.open(task_file, mode="r") as reader:
        # use while loop to read all the entries in the file
        while task_done < max_task:
            try:
                task = reader.read()  # keys include: "dataset", "index", QUESTION_KEY, ANSWER_KEY
                task_eval_res = one_task_procedure(chat_history, task, task_done, output_dir, perform_evaluation=True)  # need to implement. 
                task_eval_res_list.append(task_eval_res)
                task_done += 1
            except Exception:  # just catch all the exceptions.
                break
    # After the while loop, store the chat history in a file, but in "result_output_path" folder directly. 
    # how to store the history requires careful consideration. But a pattern could be followed: 
    # the first message is always system. Then follow a human-ai pair. Thus can store in jsonl format, with these keys below: 
    # {"role", "content", "additional_kwargs", "example"}
    chat_history_file = os.path.join(result_output_path, "chat_history.jsonl")
    with jsonlines.open(chat_history_file, mode="w", flush=True) as writer:
        for message in chat_history:
            writer.write(message.to_json())

    eval_history_file = os.path.join(result_output_path, "task_eval_res.jsonl")
    with jsonlines.open(eval_history_file, mode="w", flush=True) as writer:
        for eval_res in task_eval_res_list:
            writer.write(eval_res)


# debug
# main_procedure()
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="13b", help="model name")
    parser.add_argument("--persona_limit", type=int, default=2, help="number of personas to use")
    parser.add_argument("--temperature", type=float, default=0.1, help="temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.7, help="top p for generation")
    parser.add_argument("--max_tokens", type=int, default=200, help="max tokens for generation")
    args = parser.parse_args()
    # load llama model; the model path is: /model_weights/Llama-2-{args.model}-hf

    # set up prompts depending on model type
    
    if "chat" in args.model:
        # to be filled in later regarding which set of prompts to use. 
        system_prefix, human_prefix, ai_prefix, ai_task_completion_prefix, ai_big_five_eval_prefix, separation, stop_tokens, system_prompt_chat = \
        set_role_prefixes(True)
    else:
        system_prefix, human_prefix, ai_prefix, ai_task_completion_prefix, ai_big_five_eval_prefix, separation, stop_tokens, system_prompt_chat = \
        set_role_prefixes(False)

    
    llama_model_path = f"/insert model path/Llama-2-{args.model}-hf"
    llm = LLM(model=llama_model_path, tensor_parallel_size=4)
    # llm = "llm"
    global_models["llama2_model"] = OpenSourceLLM(llm=llm, temperature=args.temperature,
                                                    top_p=args.top_p, max_tokens=args.max_tokens)
    # tasks_idx = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]  # 10 tasks in total
    for tasks in os.listdir(os.path.join("phase1", TASK_DIR)):
        if tasks.endswith(".jsonl"):
            run_id = int(tasks.split("_")[2].split(".")[0])
            main_procedure(run_id=run_id, max_task=10, persona_limit=args.persona_limit, model_name=args.model)

