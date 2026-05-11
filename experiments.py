# Uses Llama server

import os
import json
import time
import random
from collections import Counter
from datasets import load_dataset

from openai import OpenAI
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)



####################################################################
# API wrapper: send a prompt to LLM and return the model's response #
####################################################################

def query_llm(
    client,
    prompt,
    model="meta-llama/Llama-3.1-8B-Instruct",
    temperature=0.2,
    max_tokens=200,
    timeout_s=30.0,
    max_retries=5,
    backoff_base_s=1.0,
    backoff_cap_s=16.0,
    **kwargs
):
    def is_transient(e: Exception) -> bool:
        return isinstance(e, (APIConnectionError, APITimeoutError, RateLimitError))

    last_exc = None

    for attempt in range(max_retries + 1):
        try:
            t0 = time.perf_counter()

            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout_s,
                **kwargs,
            )

            latency = time.perf_counter() - t0
            text = resp.choices[0].message.content.strip()
            usage = getattr(resp, "usage", None)

            meta = {
                "latency_s": latency,
                "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
            }

            return text, meta

        except AuthenticationError as e:
            raise RuntimeError("Authentication failed. Check MYAPIKEY1.") from e

        except Exception as e:
            last_exc = e

            if attempt >= max_retries or not is_transient(e):
                raise RuntimeError(
                    f"Request failed after {attempt + 1} attempt(s): {e}"
                ) from e

            sleep_s = min(backoff_cap_s, backoff_base_s * (2 ** attempt))
            jitter = random.uniform(0, 0.25 * sleep_s)
            time.sleep(sleep_s + jitter)

    raise RuntimeError(f"Request failed: {last_exc}")


############################
# Self-Consistency Helpers #
############################

SELF_CONSISTENCY_RUNS = 5

def extract_sentiment_label(response: str) -> str:
    text = response.lower()

    if "positive" in text:
        return "positive"
    if "negative" in text:
        return "negative"
    if "neutral" in text:
        return "neutral"

    return "unknown"


def majority_vote(answers):
    counts = Counter(answers)
    return counts.most_common(1)[0][0]


def run_self_consistency(
    client,
    prompt,
    model,
    temperature,
    max_tokens,
    task,
):
    raw_responses = []
    extracted_answers = []

    total_latency = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    for _ in range(SELF_CONSISTENCY_RUNS):
        response, meta = query_llm(
            client,
            prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        raw_responses.append(response)

        if task == "sentiment_analysis":
            extracted_answers.append(extract_sentiment_label(response))
        else:
            extracted_answers.append(response.strip())

        total_latency += meta.get("latency_s") or 0
        total_prompt_tokens += meta.get("prompt_tokens") or 0
        total_completion_tokens += meta.get("completion_tokens") or 0
        total_tokens += meta.get("total_tokens") or 0

    final_answer = majority_vote(extracted_answers)

    meta = {
        "latency_s": total_latency,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "self_consistency_runs": SELF_CONSISTENCY_RUNS,
        "raw_responses": raw_responses,
        "extracted_answers": extracted_answers,
    }

    return final_answer, meta


#####################
# Prompt Engineering #
#####################

def sentiment_prompt(review: str, technique: str) -> str:
    review = review.strip()

    if technique == "zero_shot":
        return (
            "Task: Sentiment Analysis.\n"
            "Classify the review as exactly one label: positive, negative, or neutral.\n"
            "Return ONLY the label.\n\n"
            f"Review:\n{review}\n"
        )

    if technique == "few_shot":
        return (
            "Task: Sentiment Analysis.\n"
            "Classify the review as exactly one label: positive, negative, or neutral.\n"
            "Return ONLY the label.\n\n"
            "Examples:\n"
            "Review: 'An excellent film with great acting and a compelling story.'\n"
            "Label: positive\n\n"
            "Review: 'The movie was painfully slow and the plot made no sense.'\n"
            "Label: negative\n\n"
            "Review: 'The film had some interesting moments, but overall it was average.'\n"
            "Label: neutral\n\n"
            f"Review:\n{review}\n"
            "Label:"
        )

    if technique == "chain_of_thought":
        return (
            "Task: Sentiment Analysis.\n"
            "Classify the review as positive, negative, or neutral.\n"
            "First identify key sentiment phrases.\n"
            "Then provide the final label.\n\n"
            f"Review:\n{review}\n"
            "Final label:"
        )

    if technique == "self_consistency":
        return (
            "Task: Sentiment Analysis.\n"
            "Reason step-by-step about the emotional tone of the review.\n"
            "Then provide the final label.\n\n"
            f"Review:\n{review}\n"
            "Final label:"
        )

    raise ValueError(f"Unknown technique: {technique}")


def qa_prompt(context: str, question: str, technique: str) -> str:
    if technique == "zero_shot":
        return (
            "Task: Question Answering.\n"
            "Answer the factual question using ONLY the context below.\n"
            "Return a short and precise answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    if technique == "few_shot":
        return (
            "Task: Question Answering.\n"
            "Answer the question using ONLY the information in the context.\n\n"
            "Examples:\n"
            "Context: Water boils at 100 degrees Celsius at sea level.\n"
            "Question: At what temperature does water boil?\n"
            "Answer: 100 degrees Celsius.\n\n"
            "Context: The heart has four chambers.\n"
            "Question: How many chambers does the heart have?\n"
            "Answer: Four.\n\n"
            "Context: The Pacific Ocean is the largest ocean.\n"
            "Question: Which ocean is the largest?\n"
            "Answer: The Pacific Ocean.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    if technique == "chain_of_thought":
        return (
            "Task: Question Answering.\n"
            "Step 1: Identify the important facts in the context.\n"
            "Step 2: Use them to answer the question.\n"
            "Step 3: Provide the final answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    if technique == "self_consistency":
        return (
            "Task: Question Answering.\n"
            "Reason step-by-step using ONLY the provided context.\n"
            "Then provide the final answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Final answer:"
        )

    raise ValueError(f"Unknown technique: {technique}")


###################
# Load Test Inputs #
###################

def load_test_inputs(path="test_inputs.json"):

    # If file already exists, load it
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("Creating test_inputs.json from Hugging Face datasets...")

    random.seed(42)

    ##################################
    # Sentiment Analysis (IMDb)
    ##################################

    imdb_dataset = load_dataset("imdb")

    sampled_reviews = random.sample(list(imdb_dataset["train"]), 10)

    sentiment_inputs = []

    for i, ex in enumerate(sampled_reviews):
        sentiment_inputs.append({
            "id": i,
            "text": ex["text"][:1500],
            "true_label": "positive" if ex["label"] == 1 else "negative",
        })

    ##################################
    # Question Answering (SQuAD)
    ##################################

    squad_dataset = load_dataset("squad")

    sampled_qa = random.sample(list(squad_dataset["train"]), 10)

    qa_inputs = []

    for i, ex in enumerate(sampled_qa):
        qa_inputs.append({
            "id": i,
            "context": ex["context"][:2000],
            "question": ex["question"],
            "true_answer": ex["answers"]["text"][0]
            if len(ex["answers"]["text"]) > 0 else "",
        })

    ##################################
    # Combine datasets
    ##################################

    data = {
        "sentiment_analysis": sentiment_inputs,
        "question_answering": qa_inputs,
    }

    ##################################
    # Save JSON file
    ##################################

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved test inputs to: {path}")

    return data


###################
# Experimentation #
###################

def run_experiments(
    client,
    model,
    test_data,
    temperature,
    max_tokens,
    out_json="results.json",
):
    techniques = [
        "zero_shot",
        "few_shot",
        "chain_of_thought",
        "self_consistency",
    ]

    results = []

    # Sentiment Analysis
    for technique in techniques:
        for item in test_data["sentiment_analysis"]:
            full_prompt = sentiment_prompt(item["text"], technique)

            if technique == "self_consistency":
                response, meta = run_self_consistency(
                    client=client,
                    prompt=full_prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    task="sentiment_analysis",
                )
            else:
                response, meta = query_llm(
                    client,
                    full_prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            record = {
                "task": "sentiment_analysis",
                "technique": technique,
                "input_id": item["id"],
                "true_label": item["true_label"],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "model": model,
                "prompt": full_prompt,
                "response": response,
                **meta,
            }

            results.append(record)

    # Question Answering
    for technique in techniques:
        for item in test_data["question_answering"]:
            full_prompt = qa_prompt(
                item["context"],
                item["question"],
                technique,
            )

            if technique == "self_consistency":
                response, meta = run_self_consistency(
                    client=client,
                    prompt=full_prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    task="question_answering",
                )
            else:
                response, meta = query_llm(
                    client,
                    full_prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            record = {
                "task": "question_answering",
                "technique": technique,
                "input_id": item["id"],
                "question": item["question"],
                "true_answer": item["true_answer"],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "model": model,
                "prompt": full_prompt,
                "response": response,
                **meta,
            }

            results.append(record)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved experiment results to: {out_json}")


################
# Main Function #
################

def main() -> None:
    TEMPERATURE = 0.2
    MAX_TOKENS = 200
    MODEL = "meta-llama/Llama-3.1-8B-Instruct"

    api_key = os.getenv("MYAPIKEY1")

    if not api_key:
        raise EnvironmentError("Set MYAPIKEY1 in your environment first.")

    client = OpenAI(
        base_url="http://149.165.173.247:8888/v1",
        api_key=api_key,
    )

    test_data = load_test_inputs("test_inputs.json")

    run_experiments(
        client=client,
        model=MODEL,
        test_data=test_data,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        out_json="results.json",
    )

    print("Files used/created:")
    print(" - test_inputs.json")
    print(" - results.json")


if __name__ == "__main__":
    main()