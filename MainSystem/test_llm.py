"""
Test script to verify OpenRouter API and Groq API connection and LLM functionality.
"""

import json
import requests
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

# OpenRouter API configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"

def load_openrouter_api_key() -> str:
    """Load OpenRouter API key from environment."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY in .env file")
    return api_key.strip()

def load_groq_api_key() -> str:
    """Load Groq API key from environment."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in .env file")
    return api_key.strip()

def test_groq_api():
    """Test Groq API with openai/gpt-oss-120b model."""
    print("="*80)
    print("TESTING GROQ API CONNECTION")
    print("="*80)
    
    # Load API key
    try:
        api_key = load_groq_api_key()
        print(f"✓ API key loaded: {api_key[:20]}...")
    except Exception as e:
        print(f"✗ Error loading API key: {e}")
        return False
    
    # Import Groq client
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        print("✓ Groq client initialized")
    except ImportError:
        print("✗ Groq package not installed. Install with: pip install groq")
        return False
    except Exception as e:
        print(f"✗ Error initializing Groq client: {e}")
        return False
    
    # Test request
    print(f"\nSending test request to Groq...")
    print(f"Model: qwen/qwen3-32b")
    print(f"Parameters: temperature=1, max_completion_tokens=8192")
    print(f"\nGroq Rate Limits (Free Tier):")
    print(f"  - Requests per Minute (RPM): 30")
    print(f"  - Requests per Day (RPD): 14,400")
    print(f"  - Tokens per Minute (TPM): 6,000")
    
    try:
        completion = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[
                {
                    "role": "user",
                    "content": "How many r's are in the word 'strawberry'? Answer with just a number."
                }
            ],
            temperature=1,
            max_completion_tokens=8192,
            top_p=1,
            stream=False,  # Changed to False for easier testing
            stop=None
        )
        
        print(f"\n✓ API call successful!")
        
        # Extract response
        if hasattr(completion, 'choices') and len(completion.choices) > 0:
            message = completion.choices[0].message
            content = message.content if hasattr(message, 'content') else str(message)
            
            # Check if reasoning is embedded in content (some models include reasoning in content)
            if content and len(content) > 100:
                # Try to extract final answer (may be after reasoning)
                content_lower = content.lower()
                if '<think>' in content_lower or 'reasoning' in content_lower:
                    # Extract just the answer (usually at the end)
                    lines = content.split('\n')
                    answer = lines[-1].strip() if lines else content[-50:].strip()
                    print(f"\nResponse (with reasoning): '{content[:200]}...'")
                    print(f"Extracted answer: '{answer}'")
                else:
                    print(f"\nResponse content: '{content[:200]}...'")
            else:
                print(f"\nResponse content: '{content}'")
            print(f"Content length: {len(content) if content else 0}")
            
            # Check for usage info
            if hasattr(completion, 'usage'):
                usage = completion.usage
                print(f"\nUsage information:")
                print(f"  - Prompt tokens: {usage.prompt_tokens if hasattr(usage, 'prompt_tokens') else 'N/A'}")
                print(f"  - Completion tokens: {usage.completion_tokens if hasattr(usage, 'completion_tokens') else 'N/A'}")
                print(f"  - Total tokens: {usage.total_tokens if hasattr(usage, 'total_tokens') else 'N/A'}")
            
            # Check for reasoning details (if supported by model as separate field)
            if hasattr(message, 'reasoning') and message.reasoning:
                print(f"Reasoning field found: {message.reasoning[:200]}...")
            elif content and ('<think>' in content.lower() or len(content) > 500):
                print(f"Reasoning: Embedded in content (model includes reasoning in response)")
            else:
                print(f"Reasoning: Not available (model may not support reasoning)")
            
            # Print full response structure
            print(f"\nFull response structure:")
            print(f"  - Model: {completion.model if hasattr(completion, 'model') else 'N/A'}")
            print(f"  - Choices: {len(completion.choices) if hasattr(completion, 'choices') else 0}")
            print(f"  - Object type: {type(completion).__name__}")
            
            return True
        else:
            print(f"✗ No choices in response")
            print(f"  Response: {completion}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Check if it's a rate limit error
        error_str = str(e).lower()
        if 'rate' in error_str or 'limit' in error_str or '429' in error_str:
            print(f"\n⚠️  Rate limit detected!")
            print(f"  Groq Free Tier limits:")
            print(f"    - 30 requests per minute")
            print(f"    - 14,400 requests per day")
            print(f"    - 6,000 tokens per minute")
            print(f"  Consider adding delays between requests.")
        
        return False

def test_groq_json_extraction():
    """Test Groq API with qwen/qwen3-32b for JSON extraction."""
    print("\n" + "="*80)
    print("TESTING GROQ JSON EXTRACTION (qwen/qwen3-32b)")
    print("="*80)
    
    # Load API key
    try:
        api_key = load_groq_api_key()
        from groq import Groq
        client = Groq(api_key=api_key)
        print("✓ Groq client initialized")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    # Test JSON extraction (similar to what claim_extractor and question_generator need)
    print(f"\nTesting JSON extraction with qwen/qwen3-32b...")
    print(f"Request: Extract events from a sample backstory and return JSON array")
    
    try:
        sample_backstory = """John was born in Paris in 1800. His father was a sailor. 
        John learned navigation at age 12. He traveled to London in 1815."""
        
        completion = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert information extraction system. Extract events and return ONLY valid JSON array format. No explanations, just JSON."
                },
                {
                    "role": "user",
                    "content": f"""Extract all explicit events from the following character backstory text.

CHARACTER BACKSTORY TEXT:
{sample_backstory}

Extract every explicit event mentioned. For each event, identify:
- Subject: Who performed the action
- Relation: What action/event occurred
- Object: The target/recipient (if applicable)
- Time: When this occurred (if mentioned)
- Location: Where this occurred (if mentioned)

Return a JSON array in this exact format:
[
  {{
    "subject": "exact subject from text",
    "relation": "exact action/event from text",
    "object": "object if mentioned, else empty string",
    "time": "temporal information if mentioned, else empty string",
    "location": "location if mentioned, else empty string"
  }}
]

CRITICAL: You MUST output ONLY a valid JSON array. Do NOT include any explanatory text, reasoning, or markdown formatting. Start directly with [ and end with ]. No code blocks, no explanations, just pure JSON."""
                }
            ],
            temperature=0.0,  # Deterministic for extraction
            max_completion_tokens=1024,
            stream=False
        )
        
        message = completion.choices[0].message
        content = message.content if hasattr(message, 'content') else str(message)
        
        print(f"\nRaw response length: {len(content)}")
        print(f"Raw response (first 300 chars): {content[:300]}")
        
        # Try to extract JSON (similar to claim_extractor logic)
        raw = content.strip()
        
        # Remove markdown code blocks if present
        if "```json" in raw:
            start = raw.index("```json") + 7
            end = raw.index("```", start)
            raw = raw[start:end].strip()
        elif "```" in raw:
            start = raw.index("```") + 3
            end = raw.index("```", start)
            raw = raw[start:end].strip()
        
        # Find JSON array
        if "[" in raw and "]" in raw:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            json_str = raw[start:end]
        else:
            json_str = raw
        
        # Try to parse JSON
        try:
            parsed = json.loads(json_str)
            print(f"\n✓ JSON extracted successfully!")
            print(f"Number of events extracted: {len(parsed) if isinstance(parsed, list) else 0}")
            
            if isinstance(parsed, list) and len(parsed) > 0:
                print(f"\nFirst event example:")
                print(json.dumps(parsed[0], indent=2))
                return True
            else:
                print(f"✗ Empty or invalid JSON array")
                return False
        except json.JSONDecodeError as e:
            print(f"\n✗ JSON parsing error: {e}")
            print(f"Extracted JSON string: {json_str[:500]}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_openrouter_api():
    """Test OpenRouter API with a simple request."""
    print("="*80)
    print("TESTING OPENROUTER API CONNECTION")
    print("="*80)
    
    # Load API key
    try:
        api_key = load_openrouter_api_key()
        print(f"✓ API key loaded: {api_key[:20]}...")
    except Exception as e:
        print(f"✗ Error loading API key: {e}")
        return False
    
    # Test request
    print(f"\nSending test request to OpenRouter...")
    print(f"Model: {OPENROUTER_MODEL}")
    print(f"URL: {OPENROUTER_API_URL}")
    
    try:
        response = requests.post(
            url=OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": OPENROUTER_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": "How many r's are in the word 'strawberry'? Answer with just a number."
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 50
            }),
            timeout=30
        )
        
        print(f"\nResponse status code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"✗ Error response: {response.text}")
            return False
        
        response_data = response.json()
        
        # Check for errors
        if 'error' in response_data:
            error_msg = response_data.get('error', {}).get('message', 'Unknown error')
            print(f"✗ API error: {error_msg}")
            return False
        
        # Extract response
        if 'choices' not in response_data or len(response_data['choices']) == 0:
            print(f"✗ No choices in response: {response_data}")
            return False
        
        message = response_data['choices'][0]['message']
        content = message.get('content', '')
        
        print(f"\n✓ API call successful!")
        print(f"Response content: '{content}'")
        print(f"Content length: {len(content) if content else 0}")
        
        # Check for reasoning details
        if 'reasoning_details' in message:
            print(f"Reasoning details found: {message['reasoning_details']}")
        
        # Print full response for debugging
        print(f"\nFull response structure:")
        print(f"  - Choices: {len(response_data.get('choices', []))}")
        print(f"  - Model: {response_data.get('model', 'N/A')}")
        print(f"  - Usage: {response_data.get('usage', {})}")
        print(f"  - Message keys: {list(message.keys())}")
        print(f"  - Full message: {json.dumps(message, indent=2)[:500]}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Request error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Status code: {e.response.status_code}")
            print(f"  Response: {e.response.text[:500]}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_json_extraction():
    """Test JSON extraction from LLM response."""
    print("\n" + "="*80)
    print("TESTING JSON EXTRACTION")
    print("="*80)
    
    api_key = load_openrouter_api_key()
    
    try:
        response = requests.post(
            url=OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": OPENROUTER_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that returns JSON."
                    },
                    {
                        "role": "user",
                        "content": "Return a JSON array with two numbers: [1, 2]"
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 100
            }),
            timeout=30
        )
        
        response.raise_for_status()
        response_data = response.json()
        raw = response_data['choices'][0]['message']['content'].strip()
        
        print(f"Raw response: {raw}")
        
        # Test JSON extraction
        try:
            if "```json" in raw:
                start = raw.index("```json") + 7
                end = raw.index("```", start)
                raw = raw[start:end].strip()
            elif "```" in raw:
                start = raw.index("```") + 3
                end = raw.index("```", start)
                raw = raw[start:end].strip()
            
            if "[" in raw and "]" in raw:
                start = raw.index("[")
                end = raw.rindex("]") + 1
                parsed = json.loads(raw[start:end])
                print(f"✓ JSON extracted successfully: {parsed}")
                return True
            else:
                parsed = json.loads(raw)
                print(f"✓ JSON parsed directly: {parsed}")
                return True
        except (ValueError, json.JSONDecodeError) as e:
            print(f"✗ JSON parsing error: {e}")
            print(f"  Raw text: {raw[:200]}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n")
    success1 = test_openrouter_api()
    success2 = test_json_extraction()
    success3 = test_groq_api()
    success4 = test_groq_json_extraction()
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"OpenRouter API: {'✓ PASSED' if success1 else '✗ FAILED'}")
    print(f"OpenRouter JSON Extraction: {'✓ PASSED' if success2 else '✗ FAILED'}")
    print(f"Groq API (qwen/qwen3-32b): {'✓ PASSED' if success3 else '✗ FAILED'}")
    print(f"Groq JSON Extraction (qwen/qwen3-32b): {'✓ PASSED' if success4 else '✗ FAILED'}")
    print("="*80)
    
    if success3 and success4:
        print("✓ Groq with qwen/qwen3-32b is ready for use in the framework!")
    elif success3:
        print("⚠️  Groq API works, but JSON extraction needs verification")
    
    if success1 and success2 and success3 and success4:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*80)

