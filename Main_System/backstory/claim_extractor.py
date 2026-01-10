import json
import time
import os
from models.schemas import Event
from config import load_groq_api_key

# Suppress tokenizers warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Groq API configuration
GROQ_MODEL = "qwen/qwen3-32b"

SYSTEM_PROMPT = """You are an expert information extraction system specialized in narrative analysis and character backstory processing.

Your task is to extract EXPLICIT, FACTUAL EVENTS from character backstory text. You must be precise, deterministic, and avoid any inference or assumption.

CRITICAL RULES:
1. Extract ONLY explicit events mentioned in the text - do NOT infer, assume, or explain
2. Each event must be structured with: subject (who), relation (what action/event), object (to whom/what), time (when if mentioned), location (where if mentioned)
3. Do NOT assume the subject is the main character - extract the actual subject mentioned
4. Resolve possessives explicitly (e.g., "his mother" → "mother of [character name]", "their home" → "home of [character name]")
5. Preserve temporal information exactly as stated (dates, ages, periods, "when he was X years old", etc.)
6. Preserve location information exactly as stated (places, regions, specific locations)
7. If information is not explicitly stated, use empty string "" or null
8. Output ONLY valid JSON array format - no additional text, explanations, or markdown

EXTRACTION GUIDELINES:
- Subject: The entity performing the action (person, character, or entity)
- Relation: The action, event, or state described (verb or action phrase)
- Object: The target, recipient, or object of the action (if applicable)
- Time: Temporal information (dates, ages, periods, "in 1800", "at age 12", etc.)
- Location: Geographic or spatial information (places, regions, specific locations)

OUTPUT FORMAT:
Return a JSON array where each object represents one distinct event from the text."""

def extract_claims(text: str, main_character: str, delay: float = 0.0):
    """
    Extract explicit events/claims from backstory text using Groq LLM (qwen/qwen3-32b).
    
    Args:
        text: The backstory text to extract events from
        main_character: The main character name for possessive resolution
        delay: Delay parameter (deprecated, kept for compatibility)
        
    Returns:
        List of Event objects extracted from the text
    """
    # Delay removed for speed
    
    # Validate input
    if not text or not text.strip():
        print("  ⚠️  WARNING: Empty backstory text provided", flush=True)
        return []
    
    if not main_character or not main_character.strip():
        print("  ⚠️  WARNING: Empty character name provided", flush=True)
        return []
    
    # Check API key before making call
    try:
        api_key = load_groq_api_key()
        if not api_key:
            print("  ✗ ERROR: GROQ_API_KEY not found in environment variables", flush=True)
            print("  → Please set GROQ_API_KEY in Main System/.env file", flush=True)
            return []
    except RuntimeError as e:
        print(f"  ✗ ERROR: {e}", flush=True)
        print("  → Please create Main System/.env file with GROQ_API_KEY=your_key", flush=True)
        return []
    except Exception as e:
        print(f"  ✗ ERROR loading API key: {e}", flush=True)
        return []
    
    user_prompt = f"""Extract all explicit events from the following character backstory text.

CHARACTER BACKSTORY TEXT:
{text}

MAIN CHARACTER NAME (for possessive resolution): {main_character}

Extract every explicit event mentioned in this backstory. For each event, identify:
- Subject: Who performed the action or was involved
- Relation: What action, event, or state occurred
- Object: The target, recipient, or object of the action (if applicable)
- Time: When this occurred (if mentioned - dates, ages, periods, etc.)
- Location: Where this occurred (if mentioned - places, regions, etc.)

IMPORTANT:
- Only extract events that are EXPLICITLY stated in the text
- Resolve possessives using the main character name (e.g., "his father" → "father of {main_character}")
- Preserve all temporal and location information exactly as stated
- If information is not mentioned, use empty string ""

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

    try:
        # Import Groq client
        from groq import Groq
        
        # Load API key from .env
        api_key = load_groq_api_key()
        client = Groq(api_key=api_key)
        
        # Retry logic for rate limits
        max_retries = 3
        retry_delay = 0
        
        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
                    temperature=0.0,
                    max_completion_tokens=2048,
                    stream=False
                )
                
                # Extract content from response
                if not completion.choices or len(completion.choices) == 0:
                    raise Exception(f"No choices in response")
                
                message = completion.choices[0].message
                raw = message.content if hasattr(message, 'content') else str(message)
                if not raw:
                    raise Exception(f"Empty response content")
                
                break  # Success, exit retry loop
            except Exception as e:
                error_str = str(e).lower()
                if 'rate' in error_str or 'limit' in error_str or '429' in error_str:
                    if attempt < max_retries - 1:
                        print(f"  Rate limit hit, retrying immediately... (attempt {attempt + 1}/{max_retries})", flush=True)
                        continue
                # If not a rate limit error or max retries reached, raise
                raise
        
        # Extract JSON from response (handle markdown code blocks and reasoning if present)
        try:
            raw = raw.strip()
            import re
            
            print(f"  → Raw response length: {len(raw)}", flush=True)
            print(f"  → First 200 chars: {raw[:200]}...", flush=True)
            
            # Remove markdown code blocks if present
            if "```json" in raw:
                start = raw.index("```json") + 7
                end = raw.index("```", start)
                raw = raw[start:end].strip()
                print(f"  → Extracted from markdown json block", flush=True)
            elif "```" in raw:
                start = raw.index("```") + 3
                end = raw.index("```", start)
                raw = raw[start:end].strip()
                print(f"  → Extracted from markdown code block", flush=True)
            
            # Remove reasoning tags if present (<think>, etc.)
            if "<think>" in raw.lower():
                # Find all reasoning blocks and remove them
                raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL | re.IGNORECASE)
                raw = raw.strip()
                print(f"  → Removed reasoning tags", flush=True)
            
            # Handle reasoning text if present (qwen/qwen3-32b may include reasoning before JSON)
            # Strategy: Look for ALL occurrences of [ { and use the LAST one (reasoning usually comes first)
            all_array_starts = list(re.finditer(r'\[\s*\{', raw, re.DOTALL))
            
            if all_array_starts:
                # Use the last match - it's likely the actual JSON output
                json_array_start = all_array_starts[-1]
                start = json_array_start.start()
                print(f"  → Found JSON array at position {start} (using last of {len(all_array_starts)} matches)", flush=True)
                
                # Find matching closing bracket by counting
                bracket_count = 0
                in_string = False
                escape_next = False
                end = start
                
                for i in range(start, len(raw)):
                    char = raw[i]
                    
                    if escape_next:
                        escape_next = False
                        continue
                    
                    if char == '\\':
                        escape_next = True
                        continue
                    
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    
                    if not in_string:
                        if char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                end = i + 1
                                break
                
                if end > start:
                    # Found matching brackets
                    json_str = raw[start:end]
                    try:
                        parsed = json.loads(json_str)
                        print(f"  ✓ Successfully parsed JSON array with {len(parsed)} items", flush=True)
                    except json.JSONDecodeError as je:
                        print(f"  ⚠️  JSON decode error: {je}", flush=True)
                        # Try next match if available
                        if len(all_array_starts) > 1:
                            json_array_start = all_array_starts[-2]
                            start = json_array_start.start()
                            # Retry bracket counting...
                            raise ValueError("Retrying with previous match...")
                        else:
                            raise
                else:
                    # Bracket counting failed, try regex pattern
                    raise ValueError("Bracket counting failed - will try fallback")
            elif "[" in raw and "]" in raw:
                # Fallback: try to find JSON array using regex
                json_array_pattern = r'\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]'
                json_match = re.search(json_array_pattern, raw, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                else:
                    # Last resort: find first [ and last ] (might include reasoning text)
                    start = raw.rindex("[")  # Use last [ instead of first
                    end = raw.rindex("]") + 1
                    parsed = json.loads(raw[start:end])
            else:
                # Try to parse the entire response as JSON
                parsed = json.loads(raw)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"  ⚠️  JSON parsing error: {e}", flush=True)
            print(f"  Raw response (first 500 chars): {raw[:500]}...", flush=True)
            # Try more aggressive extraction
            import re
            
            # Strategy 1: Try to find all [ { patterns and extract from the last one
            all_matches = list(re.finditer(r'\[\s*\{', raw, re.DOTALL))
            if all_matches:
                print(f"  → Found {len(all_matches)} potential JSON arrays, trying each...", flush=True)
                for match_idx, json_match in enumerate(reversed(all_matches)):  # Try from last to first
                    start = json_match.start()
                    try:
                        # Try bracket counting from this position
                        bracket_count = 0
                        in_string = False
                        escape_next = False
                        end = start
                        
                        for i in range(start, len(raw)):
                            char = raw[i]
                            if escape_next:
                                escape_next = False
                                continue
                            if char == '\\':
                                escape_next = True
                                continue
                            if char == '"' and not escape_next:
                                in_string = not in_string
                                continue
                            if not in_string:
                                if char == '[':
                                    bracket_count += 1
                                elif char == ']':
                                    bracket_count -= 1
                                    if bracket_count == 0:
                                        end = i + 1
                                        break
                        
                        if end > start:
                            json_str = raw[start:end]
                            parsed = json.loads(json_str)
                            print(f"  ✓ Successfully extracted JSON from match {len(all_matches) - match_idx}", flush=True)
                            break
                    except (json.JSONDecodeError, ValueError):
                        continue
                else:
                    # None of the matches worked, try regex patterns
                    pass
            else:
                # No [ { patterns found, try other strategies
                pass
            
            # Strategy 2: Try regex patterns for JSON array of objects
            if 'parsed' not in locals():
                json_patterns = [
                    r'\[\s*\{[^}]+\}(?:\s*,\s*\{[^}]+\})*\s*\]',  # Array of objects (non-greedy)
                    r'\[[^\]]*\{[^}]*\}[^\]]*\]',  # Array containing objects
                    r'\[.*?\]',  # Any array (last resort)
                ]
                
                for pattern in json_patterns:
                    json_match = re.search(pattern, raw, re.DOTALL)
                    if json_match:
                        try:
                            json_str = json_match.group()
                            parsed = json.loads(json_str)
                            print(f"  ✓ Successfully extracted JSON using regex pattern", flush=True)
                            break
                        except json.JSONDecodeError:
                            continue
            
            # Strategy 3: Last resort - try to find first [ and last ]
            if 'parsed' not in locals():
                if "[" in raw and "]" in raw:
                    try:
                        start = raw.rindex("[")  # Use last [ instead of first
                        end = raw.rindex("]") + 1
                        parsed = json.loads(raw[start:end])
                        print(f"  ✓ Successfully extracted JSON using last [ and ]", flush=True)
                    except (ValueError, json.JSONDecodeError):
                        raise Exception(f"Could not parse JSON from response. First 500 chars: {raw[:500]}")
                else:
                    raise Exception(f"Could not parse JSON from response. First 500 chars: {raw[:500]}")
        
        # Now parse events from the extracted JSON
        events = []
        for e in parsed:
            events.append(
                Event(
                    subject=e.get("subject", ""),
                    relation=e.get("relation", ""),
                    object=e.get("object", ""),
                    time=e.get("time", ""),
                    location=e.get("location", ""),
                    confidence=1.0
                )
            )
        
        return events
        
    except Exception as e:
        print(f"  ✗ ERROR extracting claims: {e}", flush=True)
        import traceback
        print(f"  → Full traceback:", flush=True)
        traceback.print_exc()
        print(f"  → Returning empty list - check your Groq API key in .env file", flush=True)
        return []
