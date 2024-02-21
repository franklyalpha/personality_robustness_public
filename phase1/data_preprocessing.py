
import numpy as np
import json
import jsonlines
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loading_utils import *
QUESTION_KEY = "question"
ANSWER_KEY = "answer"
TASK_DIR = "task_jsonls"


"""
Methods in this file will call the corresponding methods from data_loading_utils.py first to load the data, 
then more parameters will be provided for pre-processing. 
"""

def process_persona(sample_limit, random_idx=None):
    """
    hyperparameters must include: (applies to all processing methods)
    sample_limit: determines how many samples to load;
    random_idx: the randomly generated reshuffling of indexes from datasets, for re-producibility of shuffling results. 
        if None, will direct read from beginning to end in order. 
    
    """
    # load the data
    data = load_persona_dataset()
    # process the data
    """
    loaded data is a dictionary of keys, usually 5 categories of big-five personality: 
    extraversion, agreeableness, conscientiousness, neuroticism, openness
    each category is a hugging face dataset. using indexing to access each sample, containing the following keys: 
        "question", "statement", "answer_matching_behavior", "answer_not_matching_behavior", "label_confidence". 
        What we need is statement and answer_matching_behavior (not matching is the reverse of matching).
    
    limit the number of samples from each category to process. 
    """
    for categories in data.keys():
        if random_idx is None:
            data[categories] = data[categories][:sample_limit]
        else:
            data[categories] = data[categories][random_idx[:sample_limit]]
    # now remove all "\n" in the question. 
    for categories in data.keys():
        for index in range(len(data[categories]["question"])):
            data[categories]["question"][index] = data[categories]["question"][index].replace("\n", " ")

    return data, "question", "answer_matching_behavior"


def generate_tasks_set(task_size, seed=0):
    """Responsible for formulating the set of tasks. This method is called independently and will store the set of 
    tasks in a json file for direct loading. 
    TODO: implement this method.

    Implementation: 
    1. load the data from all datasets.
    2: for each dataset: generate a set of random integers representing indexes of samples to use.
    3: generate a random set of integers representing which task to take for each iteration (having length iteration, and maximum value is 
    the number of datasets).
    4: store the set of tasks into a .jsonl file, so later just direct read the task file.
    """
    # set seed
    np.random.seed(seed)
    # DATASETS is the list containing all loading procedures. 
    task_array = []
    task_idx_array = []
    q_key_array = []
    a_key_array = []
    for loading_method in DATASETS:
        # TODO: loading methods should be optimized in the future to load only once!!!
        data, q_key, a_key = loading_method()
        # generate the random indexes. 
        random_idx = np.random.permutation(len(data))[:task_size] 
        # convert to integer instead of np.int32
        random_idx = [int(idx) for idx in random_idx]
        task_array.append(data)
        task_idx_array.append(random_idx)
        q_key_array.append(q_key)
        a_key_array.append(a_key)
    # generate the random tasks indexes
    random_tasks = np.random.randint(0, len(DATASETS), size=task_size)
    # Now index corresponding tasks and store them into a jsonl file.
    # procedure: first identify the task/dataset, then select the corresponding index representing sample of that task
    # open a task file to write
    task_file = os.path.join("phase1", TASK_DIR, f"task_file_{seed}.jsonl")
    with jsonlines.open(task_file, mode="w", flush=True) as writer:
        for index in range(len(random_tasks)):
            task_idx = random_tasks[index]
            task_sample = task_array[task_idx][task_idx_array[task_idx][index]]
            # write the dataset name, the index in dataset, and the question and answer
            writer.write({
                "dataset": DATASETS[task_idx].__name__,
                "index": task_idx_array[task_idx][index],
                QUESTION_KEY: task_sample[q_key_array[task_idx]],
                ANSWER_KEY: task_sample[a_key_array[task_idx]],
            })
    return task_array
    

if __name__ == "__main__":
    # test all loading. 
    task_size = 10
    generate_tasks_set(task_size, 0)
    # for seed in range(0, 10000, 1000):
    #     generate_tasks_set(task_size, seed=seed)
                
        





    
    
