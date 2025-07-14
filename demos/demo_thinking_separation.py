#!/usr/bin/env python3
"""
Enhanced demo showing separated thinking content and responses.
"""

import llm
import re


class ThinkingAwareResponse:
    """Wrapper to separate thinking content from actual response."""

    def __init__(self, raw_response, original_response):
        self.original_response = original_response
        self.raw_text = raw_response
        self.thinking_content, self.actual_response = self._extract_thinking(
            raw_response
        )

    def _extract_thinking(self, text):
        """Extract thinking content and actual response."""
        if not text:
            return "", ""

        # Pattern to match <think>...</think> tags
        thinking_pattern = r"<think>(.*?)</think>"

        # Extract thinking content
        thinking_matches = re.findall(thinking_pattern, text, re.DOTALL)
        thinking_content = "\n".join(match.strip() for match in thinking_matches)

        # Remove thinking tags to get actual response
        actual_response = re.sub(thinking_pattern, "", text, flags=re.DOTALL)
        actual_response = re.sub(r"\n\s*\n+", "\n", actual_response.strip())

        return thinking_content, actual_response

    def has_thinking(self):
        """Check if response contains thinking content."""
        return bool(self.thinking_content)

    def print_separated(self):
        """Print thinking and response separately."""
        if self.has_thinking():
            print("🤔 Model's Thinking Process:")
            print("-" * 40)
            print(self.thinking_content)
            print("\n" + "=" * 50 + "\n")
            print("💬 Actual Response:")
            print("-" * 20)
            print(self.actual_response)
        else:
            print("💬 Response:")
            print("-" * 20)
            print(self.actual_response)

    def text(self):
        """Return the actual response text (without thinking)."""
        return self.actual_response

    def full_text(self):
        """Return the complete raw text including thinking tags."""
        return self.raw_text

    def tool_calls_or_raise(self):
        """Delegate tool calls to original response."""
        return self.original_response.tool_calls_or_raise()


def enhanced_prompt(model, prompt_text, **kwargs):
    """Enhanced prompt function that returns ThinkingAwareResponse."""
    response = model.prompt(prompt_text, **kwargs)
    raw_text = response.text()
    return ThinkingAwareResponse(raw_text, response)


def main():
    print("=== Enhanced Thinking-Aware Demo ===\n")

    try:
        model = llm.get_model("qwen/qwen3-4b")

        # Test 1: Simple question
        print("Test 1: Simple question")
        print("=" * 60)

        response1 = enhanced_prompt(
            model, "Explain why the sky appears blue in simple terms."
        )
        response1.print_separated()

        print("\n" + "=" * 80 + "\n")

        # Test 2: Question with tools
        print("Test 2: Tool usage with thinking")
        print("=" * 60)

        tools = [
            llm.Tool(
                name="calculate",
                description="Perform mathematical calculations",
                input_schema={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression to calculate",
                        }
                    },
                    "required": ["expression"],
                },
            )
        ]

        response2 = enhanced_prompt(
            model, "What's 47 multiplied by 23? Show your work.", tools=tools
        )

        response2.print_separated()

        # Check for tool calls
        try:
            tool_calls = response2.tool_calls_or_raise()
            if tool_calls:
                print(f"\n🔧 Tool Calls Made: {len(tool_calls)}")
                for tc in tool_calls:
                    print(f"  - {tc.name}: {tc.arguments}")
        except Exception:
            print("\n🔧 No tool calls detected")

        print("\n" + "=" * 80 + "\n")

        # Test 3: Complex reasoning
        print("Test 3: Complex multi-step reasoning")
        print("=" * 60)

        complex_prompt = """
        I have a rectangular garden that is 15 feet long and 8 feet wide.
        I want to plant tomatoes, which need 2 square feet per plant.
        I also want to leave a 1-foot border around the entire garden.
        How many tomato plants can I fit?
        """

        response3 = enhanced_prompt(model, complex_prompt.strip())
        response3.print_separated()

        print("\n" + "=" * 80 + "\n")

        # Show the different ways to access content
        print("Test 4: Different ways to access content")
        print("=" * 60)

        sample_response = enhanced_prompt(model, "What is the capital of France?")

        print("📋 Content Access Methods:")
        print(f"- Has thinking: {sample_response.has_thinking()}")
        print(f"- Clean response: '{sample_response.text()}'")
        print(f"- Full raw text: '{sample_response.full_text()}'")
        if sample_response.has_thinking():
            print(f"- Thinking only: '{sample_response.thinking_content}'")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
