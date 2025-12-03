#!/usr/bin/env python3
"""
Demonstration of our enhanced llm-lmstudio plugin with tools support.
"""

import os

import llm


def demo_tools():
    """Demonstrate tools functionality."""
    print("=== Tools Demo ===\n")

    # Ensure adequate timeout
    os.environ["LMSTUDIO_TIMEOUT"] = "15"

    try:
        model = llm.get_model("qwen/qwen3-4b")

        # Define a simple tool
        tools = [
            llm.Tool(
                name="get_weather",
                description="Get weather information for a location",
                input_schema={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city or location to get weather for",
                        }
                    },
                    "required": ["location"],
                },
            )
        ]

        print("Testing with tools...")
        response = model.prompt("What's the weather like in Paris?", tools=tools)

        print("Response received!")
        print(f"Raw text (first 200 chars): {repr(response.text()[:200])}")

        # Check for tool calls
        try:
            tool_calls = response.tool_calls_or_raise()
            print(f"\n✓ Tool calls found: {len(tool_calls)}")
            for tc in tool_calls:
                print(f"  - Tool: {tc.name}")
                print(f"    Arguments: {tc.arguments}")
        except Exception as e:
            print(f"\nNo tool calls detected: {e}")

    except Exception as e:
        print(f"Error: {e}")


def main():
    print("🚀 LM Studio Plugin Enhanced Demo\n")
    print("This demonstrates our enhanced plugin with:")
    print("- Tools support")
    print("- Enhanced timeout handling")
    print("- Better error messages\n")

    demo_tools()


if __name__ == "__main__":
    main()
