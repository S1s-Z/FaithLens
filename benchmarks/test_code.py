import pandas as pd
import json
from datasets import load_dataset, load_from_disk
from sklearn.metrics import balanced_accuracy_score, f1_score
from minicheck.minicheck import MiniCheck
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import re


def extract_model_name(model_path: str) -> str:

    path_parts = model_path.rstrip("/").split("/")
    if len(path_parts) >= 2:
        model_dir = path_parts[-2]
        checkpoint = path_parts[-1]
        return f"{model_dir}_{checkpoint}.json"
    else:
        return f"{path_parts[-1]}.json"


def main():
    parser = argparse.ArgumentParser(description="Run Evaluation")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Bespoke-MiniCheck-7B",
        help="Model name or path for MiniCheck (e.g., 'Bespoke-MiniCheck-7B', 'roberta-large', local path)"
    )
    args = parser.parse_args()

    datasets = []
    with open("./data/aggrefact.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            datasets.append(json.loads(line.strip()))

    df = pd.DataFrame(datasets)

    docs = [' '.join(doc_list) for doc_list in df.reference_documents.values]
    claims = df.statement.values
    df['label'] = df['label'].replace({'S': 1, 'NS': 0})
    df['label'] = df['label'].astype(int)

    scorer = MiniCheck(model_name=args.model_name,
                        enable_prefix_caching=False,
                        max_model_len=32768,
                        max_tokens=8192,
                        )

    # for ours
    pred_label, explanation_texts = scorer.score(docs=docs, claims=claims)

    df['preds'] = pred_label

    result_category = pd.DataFrame(columns=['category', 'Macro_F1'])
    for category in df.category.unique():
        sub_df_category = df[df.category == category]
        macro_f1 = f1_score(sub_df_category.label, sub_df_category.preds, average='macro') * 100
        result_category.loc[len(result_category)] = [category, macro_f1]

    print(result_category.round(1))

    print('Avg:', result_category.Macro_F1.mean())

    output_data = []
    for i in range(len(df)):
        sample_info = {
            "category": df.iloc[i]['category'],
            "doc": df.iloc[i]['reference_documents'],
            "claim": df.iloc[i]['statement'],
            "true_label": int(df.iloc[i]['label']),
            "pred_label": int(pred_label[i]),
            "explanation_text": explanation_texts[i],
        }
        output_data.append(sample_info)

    output_filename = extract_model_name(args.model_name)
    output_path = os.path.join('./aggrefact_results', output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)



    datasets = []
    with open("./data/hover.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            datasets.append(json.loads(line.strip()))

    df = pd.DataFrame(datasets)
    docs = [' '.join(doc_list) for doc_list in df.reference_documents.values]
    claims = df.statement.values
    df['label'] = df['label'].replace({'S': 1, 'NS': 0})
    df['label'] = df['label'].astype(int)

    pred_label, explanation_texts = scorer.score(docs=docs, claims=claims)

    df['preds'] = pred_label
    result_df = pd.DataFrame(columns=['topic', 'Macro_F1'])
    for topic in df.topic.unique():
        sub_df = df[df.topic == topic]
        macro_f1 = f1_score(sub_df.label, sub_df.preds, average='macro') * 100
        result_df.loc[len(result_df)] = [topic, macro_f1]

    print(result_df.round(1))

    output_data = []
    for i in range(len(df)):
        sample_info = {
            "topic": df.iloc[i]['topic'],
            "doc": df.iloc[i]['reference_documents'],
            "claim": df.iloc[i]['statement'],
            "true_label": int(df.iloc[i]['label']),
            "pred_label": int(pred_label[i]),
            "explanation_text": explanation_texts[i],
        }
        output_data.append(sample_info)

    output_filename = extract_model_name(args.model_name)
    output_path = os.path.join('./hover_results', output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":

    main()