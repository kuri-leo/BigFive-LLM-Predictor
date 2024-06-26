import os
import re
from glob import glob

import numpy as np

from constant import Constant


def parse_session_data(data_path: str):
    file_li = sorted(glob(os.path.join(data_path, "*.txt")))
    print(f"Total number of files: {len(file_li)}")
    session_li = []
    for file_path in file_li:
        utterance_li = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            utterance_li.append({
                "speaker": line.split(":")[0],
                "utter": line.split(":")[1].strip()
                })
        session_li.append({
            "session_id": file_path.split("/")[-1].split(".")[0],
            "user": file_path.split("/")[-1].split(".")[0].split('_')[0],
            "chat_round": file_path.split("/")[-1].split(".")[0].split('_')[2],
            "timestamp": file_path.split("/")[-1].split(".")[0].split('_')[3],
            "utterance_li": utterance_li
            })

    return session_li


def generate_bash_script(working_path):
    request_file = f"{working_path}/request.jsonl"
    response_file = f"{working_path}/response.jsonl"
    error_file = f"{working_path}/error.jsonl"

    bash_script = f"""#!/bin/bash

REQUEST_FILE="{request_file}"
RESPONSE_FILE="{response_file}"
ERROR_FILE="{error_file}"

SCRIPT_FILE="{Constant.SCRIPT_FILE}"
API_KEY="{Constant.API_KEY}"
REQUEST_URL="{Constant.REQUEST_URL}"

python "$SCRIPT_FILE" \\
    --request_url="$REQUEST_URL" \\
    --api_key="$API_KEY" \\
    --requests_filepath="$REQUEST_FILE" \\
    --save_filepath="$RESPONSE_FILE" \\
    --error_filepath="$ERROR_FILE" \\
    --max_attempts=5 \\
    --max_requests_per_minute=2048 \\
    --max_tokens_per_minute=99999999 \\
    --max_task=120 \\
    --logging_level=30 \\
    --resume"""

    with open(f"{working_path}/run.sh", "w") as f:
        f.write(bash_script)

    print("Bash script generated:")
    print(f"bash {working_path}/run.sh")


# Define choices for extract_choice function
CHOICES = {
    "非常不同意": 1,
    "不太同意": 2,
    "中立": 3,
    "比较同意": 4,
    "非常同意": 5
    }

TEXT_PATTERN = re.compile(r'非常不同意|不太同意|中立|比较同意|非常同意')
NUMBER_PATTERN = re.compile(r'([1-5])')


def extract_choice(sentence):
    text_match = TEXT_PATTERN.search(sentence)
    if text_match and len(text_match.group(0)) == 1:
        return CHOICES[text_match.group(0)]

    number_match = NUMBER_PATTERN.search(sentence)
    if number_match and len(number_match.group(1)) == 1:
        return int(number_match.group(1))

    return -1


def calculate_trait_scores(series, trait_indices):
    pred_scores = np.mean([series['pred'][str(i)] for i in trait_indices])
    return pred_scores


def convert_scores(series):
    traits = {
        "extraversion": range(0, 56, 5),
        "agreeableness": range(1, 57, 5),
        "conscientiousness": range(2, 58, 5),
        "negative_emotionality": range(3, 59, 5),
        "open_mindedness": range(4, 60, 5)
        }

    scores = {}
    for trait, indices in traits.items():
        pred_score = calculate_trait_scores(series, indices)
        scores[f"pred_{trait}"] = pred_score

    return scores
