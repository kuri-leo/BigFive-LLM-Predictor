import argparse
import json
import os
import random
from datetime import datetime

import numpy as np

from constant import Constant
from utils import generate_bash_script, parse_session_data

# Constants
TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
RANDOM_SEED = 1337

REQUEST_TEMPLATE = """在本次心理咨询结束之前，请根据聊天内容与你自身的情况完成下面这道选择题：
---
题目：{}
选项：
1. 非常不同意
2. 不太同意
3. 中立
4. 比较同意
5. 非常同意
---
请告诉我你的选项并说明理由："""
SYSTEM_PROMPT = ("Act like a real human and do not mention anything with AI. "
                 "表现得像个真正的人类，不要提及任何与人工智能有关的事情。"
                 "作为这次心理咨询的来访者，你将与你的咨询师进行对话。")

NUM_TRIES = 1
STOP_TOKENS = []  # stop tokens for the model, follow chat template and vllm implementation

# Seed initialization
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def initialize_messages(model_name, system_prompt):
    if model_name in ['gemma-7b-it', 'gemma-1.1-7b-it']:
        return [
            {"role": "user", "content": system_prompt + "如果你理解任务，请回答 \"我理解任务了\"。"},
            {"role": "assistant", "content": "我理解任务了"}
            ]
    else:
        return [{"role": "system", "content": system_prompt}]


def build_messages(example, model_name):
    messages = initialize_messages(model_name, SYSTEM_PROMPT)
    for idx, utterance in enumerate(example):
        role = "user" if utterance['speaker'] == '咨询师' else "assistant"
        if idx == 0 and role == "assistant":
            messages.append({"role": "user", "content": "你好"})
        if messages[-1]['role'] == role:
            messages[-1]['content'] += "\n" + utterance['utter']
        else:
            messages.append({"role": role, "content": utterance['utter']})
    if messages[-1]['role'] == "user":
        messages.append({"role": "assistant", "content": "嗯嗯"})
    return messages


def create_jobs(session_li, questionnaire, model_name):
    jobs = []
    for session in session_li:
        example = session["utterance_li"]
        user_id = session["user"]
        messages = build_messages(example, model_name)
        for item_idx, item in enumerate(questionnaire):
            for try_idx in range(NUM_TRIES):
                uuid_str = f"{session['session_id']}_{item_idx}_{try_idx}"
                job = {
                    "uuid": uuid_str,
                    "model": model_name,
                    "messages": messages + [{"role": "user", "content": REQUEST_TEMPLATE.format(item[4:])}],
                    "max_tokens": 256,
                    "skip_special_tokens": True,
                    "stop": STOP_TOKENS,
                    }
                jobs.append(job)
    return jobs


def save_jobs(jobs, output_path):
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'request.jsonl'), "w") as f:
        for job in jobs:
            json_string = json.dumps(job, ensure_ascii=False)
            f.write(json_string + "\n")


def main(args):
    session_li = parse_session_data(args.source_path)

    jobs = create_jobs(session_li, Constant.BFI_ITEM_LI, args.model_name)
    print(f"Total number of jobs: {len(jobs)}")

    save_jobs(jobs, args.output_path)
    generate_bash_script(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--source_path", type=str, help="Path to the source data")
    parser.add_argument("--output_path", type=str, help="Path to save the output files")
    args = parser.parse_args()
    main(args)
