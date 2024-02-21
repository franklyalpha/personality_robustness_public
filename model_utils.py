from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.llms import OpenAI, Cohere
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain, SequentialChain

import jsonlines
from vllm import LLM, SamplingParams
from langchain_llama import *

openai_api_key = "insert api key"  # this is the personal API key; for debugging and running small experiments. 
cohere_api_key = "inser api key"
anthropic_api_key = ""

temperature = 1.0
max_tokens = 400
top_p = 1.0
# llm = LLM(model="insert model path", tensor_parallel_size=4)
# llm = "llm"

global_models = {
    "gpt_model":
    ChatOpenAI(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        openai_api_key=openai_api_key,
        model="gpt-3.5-turbo", 
        max_retries=6, 
        ),
    # "llama2_model":
    # OpenSourceLLM(llm=llm, temperature=temperature, 
    # top_p=top_p, max_tokens=100),
    "cohere_model":
    Cohere(
        temperature=temperature,
        max_tokens=max_tokens,
        cohere_api_key=cohere_api_key,
        max_retries=6, 
        # model="",
        ),
    "evaluation_model":
    ChatOpenAI(
        temperature=0.5,
        max_tokens=max_tokens,
        top_p=top_p,
        openai_api_key=openai_api_key,
        model="gpt-3.5-turbo"),
    # "evaluation_model":
    # OpenSourceLLM(llm=llm, temperature=0.2, 
    # top_p=0.7, max_tokens=200),
    # "anthropic_model":
    # ChatAnthropic(
    #     temperature=temperature,
    #     max_tokens=max_tokens,
    #     top_p=top_p,
    #     anthropic_api_key=anthropic_api_key,
    #     # model="",
    # ),
}
