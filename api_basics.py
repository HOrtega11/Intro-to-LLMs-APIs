"""
API Provider Used:
OpenAI-compatible Llama server through the UTSA course proxy.

Base URL:
http://149.165.173.247:8888/v1

Default model:
meta-llama/Llama-3.1-8B-Instruct
"""

import os
import time
import random
import socket
from typing import Optional, Any

from openai import OpenAI
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
    InternalServerError,
)


# LLM Query + Robust Retry Logic
def query_llm(
    client: OpenAI,
    prompt: str,
    *,
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    temperature: float = 0.2,
    max_tokens: int = 200,
    timeout_s: float = 20.0,
    max_retries: int = 5,
    backoff_base_s: float = 1.0,
    backoff_cap_s: float = 20.0,
    **kwargs: Any,
) -> str:
    """
    Sends a prompt to the LLM and returns the response text.

    Includes:
    - temperature and max_tokens parameters
    - authentication error handling
    - connection, timeout, rate limit, and server error handling
    - retry mechanism with exponential backoff
    """

    def is_transient(exc: Exception) -> bool:
        """
        Transient errors are temporary errors worth retrying.
        """
        return isinstance(
            exc,
            (
                APIConnectionError,
                APITimeoutError,
                RateLimitError,
                InternalServerError,
                socket.timeout,
            ),
        )

    last_exc: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout_s,
                **kwargs,
            )

            return resp.choices[0].message.content.strip()

        except AuthenticationError as e:
            raise RuntimeError(
                "Authentication failed. Check that MYAPIKEY1 is set correctly "
                "and that the course server expects this key."
            ) from e

        except Exception as e:
            last_exc = e

            if attempt >= max_retries or not is_transient(e):
                raise RuntimeError(
                    f"LLM request failed after {attempt + 1} attempt(s): {e}"
                ) from e

            sleep_s = min(backoff_cap_s, backoff_base_s * (2 ** attempt))
            jitter = random.uniform(0, 0.25 * sleep_s)

            print(
                f"Transient error occurred. Retrying in {sleep_s + jitter:.2f} seconds..."
            )

            time.sleep(sleep_s + jitter)

    raise RuntimeError(f"LLM request failed: {last_exc}")


# Main Function
def main() -> None:
    api_key = os.getenv("MYAPIKEY1")

    if not api_key:
        raise EnvironmentError(
            "Environment variable MYAPIKEY1 is not set. "
            "Example: export MYAPIKEY1='your_key_here'"
        )

    client = OpenAI(
        base_url="http://149.165.173.247:8888/v1",
        api_key=api_key,
    )

    prompts = [
        "Explain the difference between supervised and unsupervised learning in 3 sentences.",
        "What is the capital of China?",
        "What language is spoken in Brazil?",
    ]

    for i, prompt in enumerate(prompts, start=1):
        print(f"\n=== Prompt {i} ===")
        print(prompt)
        print("\n--- Response ---")

        try:
            answer = query_llm(
                client,
                prompt,
                temperature=0.2,
                max_tokens=200,
                timeout_s=20.0,
                max_retries=5,
            )
            print(answer)

        except Exception as e:
            print(f"[Error] {e}")


if __name__ == "__main__":
    main()
