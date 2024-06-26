import json

import pandas as pd
from tqdm import tqdm

from utils import convert_scores, extract_choice

# Load response data
response_files = [
    '/path/to/response.jsonl',
    ]

# Output file
output_file = '/path/to/output.csv'


def calculate_big_five_scores(series):
    scores = convert_scores(series)

    return pd.Series({**scores})


def load_responses(file_paths):
    results = []

    for response_file in tqdm(file_paths):
        model_name = response_file.split('/')[-2].split('_')[1]
        role = response_file.split('/')[-2].split('_')[3]

        with open(response_file, "r") as file:
            response_data = file.readlines()

        for line in response_data:
            raw_data = json.loads(line)
            uuid = raw_data[1]['uuid']
            user_id, chat_round, question_id, try_id = uuid.split('_')[0], uuid.split('_')[2], \
                uuid.split('_')[4], uuid.split('_')[5]
            session_id = "_".join(uuid.split('_')[:-3])

            if 'choices' not in raw_data[1]:
                content = -1
            else:
                content = raw_data[1]['choices'][0]['message']['content'].replace("\n", "").replace("\t", "").strip()
            pred = extract_choice(content.replace("/n", "").strip())

            results.append({
                "model_name": model_name,
                "role": role,
                "session_id": session_id,
                "user_id": user_id,
                "chat_round": chat_round,
                "question_id": question_id,
                "try_id": try_id,
                "pred": pred,
                "content": content
                })

    return pd.DataFrame(results)


def process_data(df):
    df['pred'] = df['pred'].astype(int)

    pivot_df = df.pivot_table(
        index=["model_name", "role", "user_id", "session_id", "chat_round", "try_id"],
        columns=["question_id"],
        values=["pred"]
        )

    return pivot_df.apply(calculate_big_five_scores, axis=1).reset_index()


def main():
    response_df = load_responses(response_files)

    # Process and calculate big five scores
    results_df = process_data(response_df)
    print(results_df.head())

    results_df.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()
