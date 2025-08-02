#!/usr/bin/env python3
"""
Quick test to check LM Studio API directly
"""

import requests
import time


def test_lmstudio_direct():
    print("Testing LM Studio API directly...")

    try:
        # Test with a very simple request
        start_time = time.time()

        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": "qwen/qwen3-4b",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
                "stream": False,
            },
            timeout=5,  # 5 second timeout
        )

        elapsed = time.time() - start_time
        print(f"Request completed in {elapsed:.2f}s")
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            print(f"Response: {content}")
            return True
        else:
            print(f"Error: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print("Request timed out after 5 seconds")
        return False
    except Exception as e:
        print(f"Exception: {e}")
        return False


if __name__ == "__main__":
    test_lmstudio_direct()
