#!/usr/bin/env python3
"""
Demonstration of our enhanced llm-lmstudio plugin with tools and thinking tag separation.
"""

import llm
import os

def demo_thinking_separation():
    """Demonstrate the thinking tag separation functionality."""
    print("=== Thinking Tag Separation Demo ===\n")
    
    # Ensure adequate timeout
    os.environ["LMSTUDIO_TIMEOUT"] = "15"
    
    try:
        model = llm.get_model("qwen/qwen3-4b")
        print(f"Model loaded: {model}")
        print(f"Supports tools: {getattr(model, 'supports_tools', False)}\n")
        
        # Simple test without tools
        print("Testing simple prompt...")
        response = model.prompt("What is the capital of France?")
        raw_text = response.text()
        
        print("Raw response (first 200 chars):")
        print(repr(raw_text[:200]))
        print()
        
        # Check if it contains thinking tags
        if "<think>" in raw_text:
            print("✓ Thinking tags detected in response")
            
            # Demonstrate separation
            import re
            thinking_pattern = r"<think>(.*?)</think>"
            thinking_matches = re.findall(thinking_pattern, raw_text, re.DOTALL)
            thinking_content = "\n".join(thinking_matches).strip()
            
            actual_response = re.sub(thinking_pattern, "", raw_text, flags=re.DOTALL)
            actual_response = re.sub(r"\n\s*\n+", "\n", actual_response.strip())
            
            print(f"Thinking content:\n{thinking_content}\n")
            print(f"Clean response:\n{actual_response}\n")
        else:
            print("No thinking tags detected")
            print(f"Response: {raw_text}\n")
            
    except Exception as e:
        print(f"Error: {e}")

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
                            "description": "The city or location to get weather for"
                        }
                    },
                    "required": ["location"]
                }
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
    print("- Thinking tag separation")
    print("- Enhanced timeout handling")
    print("- Better error messages\n")
    
    demo_thinking_separation()
    print("\n" + "="*60 + "\n")
    demo_tools()

if __name__ == "__main__":
    main()
