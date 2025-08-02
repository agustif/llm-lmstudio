#!/usr/bin/env python3
"""
Demo script showing how to use the updated llm-lmstudio plugin with tools support.
This demonstrates the newly implemented tools functionality with qwen models.
"""

import llm
from datetime import datetime


def get_current_time():
    """Get the current time in a readable format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    """
    Safely calculate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate (e.g., "2+2", "10*5")

    Returns:
        The result of the calculation as a string
    """
    try:
        # Simple safety check - only allow basic math operations
        allowed_chars = set("0123456789+-*/()., ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"

        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def get_weather(location: str) -> str:
    """
    Get weather information for a location (mock function for demo).

    Args:
        location: The city or location to get weather for

    Returns:
        Mock weather information as a string
    """
    # This is a mock function for demonstration
    mock_weather = {
        "new york": "Partly cloudy, 72°F (22°C), light breeze",
        "london": "Overcast, 65°F (18°C), chance of rain",
        "tokyo": "Sunny, 78°F (26°C), clear skies",
        "san francisco": "Foggy, 60°F (16°C), typical marine layer",
    }

    location_lower = location.lower()
    for city in mock_weather:
        if city in location_lower:
            return f"Weather in {location}: {mock_weather[city]}"

    return f"Weather in {location}: Sunny, 75°F (24°C), pleasant conditions (mock data)"


def main():
    print("=== LLM-LMStudio Tools Demo ===\n")

    # Initialize the model
    try:
        model = llm.get_model("qwen/qwen3-4b")
        print(f"✓ Using model: {model}")
        print(f"✓ Supports tools: {getattr(model, 'supports_tools', False)}")
        print()
    except Exception as e:
        print(f"✗ Error getting model: {e}")
        print("Make sure LM Studio is running and qwen/qwen3-4b is loaded")
        return

    # Define tools
    tools = [
        llm.Tool(
            name="get_current_time",
            description="Get the current date and time",
            input_schema={"type": "object", "properties": {}, "required": []},
        ),
        llm.Tool(
            name="calculate",
            description="Calculate a mathematical expression safely",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '2+2', '10*5')",
                    }
                },
                "required": ["expression"],
            },
        ),
        llm.Tool(
            name="get_weather",
            description="Get current weather information for a location",
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
        ),
    ]

    # Test prompts that should trigger tool usage
    test_prompts = [
        "What time is it right now?",
        "What's 47 * 23?",
        "What's the weather like in Tokyo?",
        "Can you tell me the current time and calculate 15 + 27?",
    ]

    print("=== Running Tool Tests ===\n")

    for i, prompt_text in enumerate(test_prompts, 1):
        print(f"Test {i}: {prompt_text}")
        print("-" * 50)

        try:
            # Create conversation to track tool interactions
            conversation = model.conversation()

            # Send prompt with tools
            response = conversation.prompt(prompt_text, tools=tools)

            # Check if model made tool calls
            try:
                tool_calls = response.tool_calls_or_raise()
                if tool_calls:
                    print(f"✓ Model made {len(tool_calls)} tool call(s):")

                    # Execute the tool calls
                    tool_results = []
                    for tc in tool_calls:
                        print(f"  - Calling {tc.name} with {tc.arguments}")

                        # Execute the appropriate function
                        if tc.name == "get_current_time":
                            result = get_current_time()
                        elif tc.name == "calculate":
                            result = calculate(tc.arguments.get("expression", ""))
                        elif tc.name == "get_weather":
                            result = get_weather(tc.arguments.get("location", ""))
                        else:
                            result = f"Unknown tool: {tc.name}"

                        print(f"    Result: {result}")
                        tool_results.append(
                            llm.ToolResult(
                                name=tc.name,
                                tool_call_id=tc.tool_call_id,
                                output=result,
                            )
                        )

                    # Send tool results back to get final response
                    print("\n  Sending results back to model...")
                    final_response = conversation.prompt("", tool_results=tool_results)
                    print(f"  Final response: {final_response.text()}")
                else:
                    print("  No tool calls made")
                    print(f"  Response: {response.text()}")

            except Exception:
                print("  No tool calls (this is normal)")
                print(f"  Response: {response.text()}")

        except Exception as e:
            print(f"✗ Error with prompt: {e}")

        print("\n")

    print("=== Advanced Example: Multi-turn with Tools ===\n")

    try:
        # Create a new conversation for multi-turn example
        conversation = model.conversation()

        # First turn
        print("User: What's the current time and weather in San Francisco?")
        response1 = conversation.prompt(
            "What's the current time and weather in San Francisco?", tools=tools
        )

        # Handle tool calls from first response
        try:
            tool_calls = response1.tool_calls_or_raise()
            if tool_calls:
                print(f"Assistant made {len(tool_calls)} tool calls")
                tool_results = []
                for tc in tool_calls:
                    if tc.name == "get_current_time":
                        result = get_current_time()
                    elif tc.name == "get_weather":
                        result = get_weather(tc.arguments.get("location", ""))
                    else:
                        result = f"Unknown tool: {tc.name}"

                    tool_results.append(
                        llm.ToolResult(
                            name=tc.name, tool_call_id=tc.tool_call_id, output=result
                        )
                    )

                # Get response with tool results
                response1_final = conversation.prompt("", tool_results=tool_results)
                print(f"Assistant: {response1_final.text()}")
            else:
                print(f"Assistant: {response1.text()}")
        except Exception:
            print(f"Assistant: {response1.text()}")

        # Second turn - follow up
        print("\nUser: Now calculate how many hours until midnight")
        response2 = conversation.prompt(
            "Now calculate how many hours until midnight", tools=tools
        )

        # Handle any tool calls from second response
        try:
            tool_calls = response2.tool_calls_or_raise()
            if tool_calls:
                tool_results = []
                for tc in tool_calls:
                    if tc.name == "calculate":
                        result = calculate(tc.arguments.get("expression", ""))
                    elif tc.name == "get_current_time":
                        result = get_current_time()
                    else:
                        result = f"Unknown tool: {tc.name}"

                    tool_results.append(
                        llm.ToolResult(
                            name=tc.name, tool_call_id=tc.tool_call_id, output=result
                        )
                    )

                response2_final = conversation.prompt("", tool_results=tool_results)
                print(f"Assistant: {response2_final.text()}")
            else:
                print(f"Assistant: {response2.text()}")
        except Exception:
            print(f"Assistant: {response2.text()}")

    except Exception as e:
        print(f"✗ Error in multi-turn example: {e}")

    print("\n=== Demo Complete ===")
    print("The llm-lmstudio plugin now fully supports tools! 🎉")


if __name__ == "__main__":
    main()
