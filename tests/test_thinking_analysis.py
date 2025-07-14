#!/usr/bin/env python3
"""
Test script to understand how the model is actually responding and
experiment with different approaches to handle thinking tags.
"""

import llm
import re


def extract_thinking_and_response(text):
    """
    Extract thinking content and actual response separately.
    Returns tuple: (thinking_content, actual_response)
    """
    if not text:
        return "", ""

    # Pattern to match <think>...</think> tags with content
    thinking_pattern = r"<think>(.*?)</think>"

    # Extract all thinking content
    thinking_matches = re.findall(thinking_pattern, text, re.DOTALL)
    thinking_content = "\n".join(thinking_matches).strip()

    # Remove thinking tags to get actual response
    actual_response = re.sub(thinking_pattern, "", text, flags=re.DOTALL)
    actual_response = re.sub(r"\n\s*\n+", "\n", actual_response.strip())

    return thinking_content, actual_response


def main():
    print("=== Thinking Tags Analysis ===\n")

    try:
        model = llm.get_model("qwen/qwen3-4b")

        # Test 1: Simple question without tools
        print("Test 1: Simple question (no tools)")
        print("-" * 40)

        response1 = model.prompt("What is 2 + 2?")
        raw_text1 = response1.text()

        print(f"Raw response:\n{repr(raw_text1)}\n")

        thinking, actual = extract_thinking_and_response(raw_text1)
        if thinking:
            print(f"Thinking content:\n{thinking}\n")
            print(f"Actual response:\n{actual}\n")
        else:
            print("No thinking tags detected\n")

        print("=" * 60 + "\n")

        # Test 2: Question with tools
        print("Test 2: Question with tools")
        print("-" * 40)

        tools = [
            llm.Tool(
                name="calculate",
                description="Perform basic arithmetic calculations",
                input_schema={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate",
                        }
                    },
                    "required": ["expression"],
                },
            )
        ]

        response2 = model.prompt("Calculate 15 * 8", tools=tools)
        raw_text2 = response2.text()

        print(f"Raw response:\n{repr(raw_text2)}\n")

        thinking2, actual2 = extract_thinking_and_response(raw_text2)
        if thinking2:
            print(f"Thinking content:\n{thinking2}\n")
            print(f"Actual response:\n{actual2}\n")
        else:
            print("No thinking tags detected\n")

        # Check for tool calls
        try:
            tool_calls = response2.tool_calls_or_raise()
            print(f"Tool calls found: {len(tool_calls)}")
            for tc in tool_calls:
                print(f"  - {tc.name}: {tc.arguments}")
        except Exception as e:
            print(f"No tool calls detected: {e}")

        print("\n" + "=" * 60 + "\n")

        # Test 3: More complex scenario
        print("Test 3: Complex multi-step thinking")
        print("-" * 40)

        complex_prompt = """I need you to help me plan a dinner party. 
        Consider the following:
        1. I have 8 guests coming
        2. Two are vegetarian
        3. One has a nut allergy
        4. Budget is $200
        
        What should I make?"""

        response3 = model.prompt(complex_prompt)
        raw_text3 = response3.text()

        print(f"Raw response:\n{repr(raw_text3)}\n")

        thinking3, actual3 = extract_thinking_and_response(raw_text3)
        if thinking3:
            print(f"Thinking content:\n{thinking3}\n")
            print(f"Actual response:\n{actual3}\n")
        else:
            print("No thinking tags detected\n")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
