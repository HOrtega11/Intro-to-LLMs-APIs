import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_PATH = Path("results.json")
OUTPUT_DIR = Path("analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_results():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    return pd.DataFrame(results)


def clean_predictions(df):
    df = df.copy()

    df["response_clean"] = (
        df["response"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    def sentiment_correct(row):
        if row["task"] != "sentiment_analysis":
            return None

        true_label = str(row["true_label"]).lower().strip()
        response = str(row["response_clean"])

        return true_label in response

    def qa_correct(row):
        if row["task"] != "question_answering":
            return None

        true_answer = str(row["true_answer"]).lower().strip()
        response = str(row["response_clean"])

        # exact or substring match
        return (
            true_answer in response
            or response in true_answer
        )

    df["sentiment_correct"] = df.apply(sentiment_correct, axis=1)
    df["qa_correct"] = df.apply(qa_correct, axis=1)

    return df


def make_summary_tables(df):
    summary = (
        df.groupby(["task", "technique"])
        .agg(
            num_examples=("response", "count"),
            avg_latency_s=("latency_s", "mean"),
            avg_total_tokens=("total_tokens", "mean"),
            avg_prompt_tokens=("prompt_tokens", "mean"),
            avg_completion_tokens=("completion_tokens", "mean"),
        )
        .reset_index()
    )

    # Force logical ordering for plots/tables
    technique_order = [
        "zero_shot",
        "few_shot",
        "chain_of_thought",
        "self_consistency",
    ]

    task_order = [
        "sentiment_analysis",
        "question_answering",
    ]

    summary["technique"] = pd.Categorical(
        summary["technique"],
        categories=technique_order,
        ordered=True,
    )

    summary["task"] = pd.Categorical(
        summary["task"],
        categories=task_order,
        ordered=True,
    )

    summary = summary.sort_values(["task", "technique"])
    sentiment_df = df[df["task"] == "sentiment_analysis"].copy()

    sentiment_accuracy = (
        sentiment_df.groupby("technique")
        .agg(
            accuracy=("sentiment_correct", "mean"),
            num_examples=("sentiment_correct", "count"),
        )
        .reset_index()
    )

    qa_df = df[df["task"] == "question_answering"].copy()

    qa_accuracy = (
        qa_df.groupby("technique")
        .agg(
            accuracy=("qa_correct", "mean"),
            num_examples=("qa_correct", "count"),
        )
        .reset_index()
    )

    summary.to_csv(OUTPUT_DIR / "summary_by_task_and_technique.csv", index=False)
    sentiment_accuracy.to_csv(OUTPUT_DIR / "sentiment_accuracy.csv", index=False)

    qa_accuracy.to_csv(
        OUTPUT_DIR / "qa_accuracy.csv",
        index=False,
    )

    print("\nSummary by task and technique:")
    print(summary)

    print("\nSentiment accuracy:")
    print(sentiment_accuracy)

    print("\nQA accuracy:")
    print(qa_accuracy)

    return summary, sentiment_accuracy, qa_accuracy


def plot_avg_latency(summary):
    plot_df = summary.copy()
    labels = plot_df["task"].astype(str) + "\n" + plot_df["technique"].astype(str)

    plt.figure(figsize=(12, 6))
    plt.bar(labels, plot_df["avg_latency_s"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average latency (seconds)")
    plt.title("Average Latency by Task and Prompting Technique")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "avg_latency_by_task_technique.png", dpi=300)
    plt.close()


def plot_avg_tokens(summary):
    plot_df = summary.copy()
    labels = plot_df["task"].astype(str) + "\n" + plot_df["technique"].astype(str)

    plt.figure(figsize=(12, 6))
    plt.bar(labels, plot_df["avg_total_tokens"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average total tokens")
    plt.title("Average Token Usage by Task and Prompting Technique")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "avg_tokens_by_task_technique.png", dpi=300)
    plt.close()


def plot_sentiment_accuracy(sentiment_accuracy):

    technique_order = [
        "zero_shot",
        "few_shot",
        "chain_of_thought",
        "self_consistency",
    ]

    sentiment_accuracy["technique"] = pd.Categorical(
        sentiment_accuracy["technique"],
        categories=technique_order,
        ordered=True,
    )

    sentiment_accuracy = sentiment_accuracy.sort_values("technique")

    plt.figure(figsize=(8, 5))
    plt.bar(
        sentiment_accuracy["technique"].astype(str),
        sentiment_accuracy["accuracy"],
    )

    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Sentiment Analysis Accuracy by Prompting Technique")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "sentiment_accuracy_by_technique.png",
        dpi=300,
    )

    plt.close()

def plot_qa_accuracy(qa_accuracy):

    technique_order = [
        "zero_shot",
        "few_shot",
        "chain_of_thought",
        "self_consistency",
    ]

    qa_accuracy["technique"] = pd.Categorical(
        qa_accuracy["technique"],
        categories=technique_order,
        ordered=True,
    )

    qa_accuracy = qa_accuracy.sort_values("technique")

    plt.figure(figsize=(8, 5))
    plt.bar(
        qa_accuracy["technique"].astype(str),
        qa_accuracy["accuracy"],
    )

    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Question Answering Accuracy by Prompting Technique")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "qa_accuracy_by_technique.png",
        dpi=300,
    )

    plt.close()


def main():
    df = load_results()
    df = clean_predictions(df)

    df.to_csv(OUTPUT_DIR / "full_results_table.csv", index=False)

    summary, sentiment_accuracy, qa_accuracy = make_summary_tables(df)

    plot_avg_latency(summary)
    plot_avg_tokens(summary)
    plot_sentiment_accuracy(sentiment_accuracy)
    plot_qa_accuracy(qa_accuracy)

    print("\nSaved analysis outputs to:")
    print(f" - {OUTPUT_DIR / 'full_results_table.csv'}")
    print(f" - {OUTPUT_DIR / 'summary_by_task_and_technique.csv'}")
    print(f" - {OUTPUT_DIR / 'sentiment_accuracy.csv'}")
    print(f" - {OUTPUT_DIR / 'avg_latency_by_task_technique.png'}")
    print(f" - {OUTPUT_DIR / 'avg_tokens_by_task_technique.png'}")
    print(f" - {OUTPUT_DIR / 'sentiment_accuracy_by_technique.png'}")
    print(f" - {OUTPUT_DIR / 'qa_accuracy.csv'}")
    print(f" - {OUTPUT_DIR / 'qa_accuracy_by_technique.png'}")


if __name__ == "__main__":
    main()