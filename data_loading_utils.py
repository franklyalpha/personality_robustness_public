import datasets
from datasets import load_dataset
import os
import numpy as np
import json
import jsonlines
# from pathlib import Path
# datasets.config.DOWNLOADED_DATASETS_PATH = Path(os.path.join("datasets", "downloaded_cache"))




def load_persona_dataset(categories=["extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness"]):
    """
    Loads the persona dataset from the given categories
    Currently would load big five personality traits
    """
    
    data = {}
    for category in categories:
        dataset = load_dataset("EleutherAI/persona", category)
        data[category] = dataset["validation"]
    return data

def load_gsm8k_dataset():
    """
    Loads the GSM8K test dataset
    """
    dataset = load_dataset("gsm8k", "main", split="test")
    return dataset, "question", "answer"

def load_csqa_dataset():
    """
    Loads the CSQA dataset
    """
    dataset = load_dataset("commonsense_qa", split="train")
    return dataset, "question", "answerKey"

def load_strategyqa_dataset():
    """
    Loads the StrategyQA dataset
    """
    dataset = load_dataset("ChilleD/StrategyQA", split="test")
    return dataset, "question", "answer"

# TODO: add the method here always.
DATASETS = [load_gsm8k_dataset, load_csqa_dataset, load_strategyqa_dataset]


if __name__ == "__main__":
    # test all loading. 
    # persona = load_persona_dataset()
    gsm8k = load_gsm8k_dataset()
    csqa = load_csqa_dataset()
    strategyqa = load_strategyqa_dataset()
    a = 3