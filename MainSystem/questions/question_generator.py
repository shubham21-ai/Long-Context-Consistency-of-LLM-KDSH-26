import json
import time
import os
from models.schemas import Event
from config import load_groq_api_key

# Suppress tokenizers warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Groq API configuration
GROQ_MODEL = "qwen/qwen3-32b"

SYSTEM_PROMPT = """You are an expert question generation system specialized in narrative consistency checking and story verification.

Your task is to generate specific, testable, and verifiable questions based on events extracted from character backstories. These questions will be used to verify whether the main story narrative is consistent with the character's backstory events.

QUESTION GENERATION PRINCIPLES:
1. Specificity: Questions must be specific enough to be answerable from the story text
2. Verifiability: Questions should focus on factual information that can be found or verified in the story
3. Context Preservation: Include temporal and location context when available
4. Clarity: Questions should be clear, unambiguous, and directly related to the event
5. Coverage: Generate 2-3 questions per event to cover different aspects (what, when, where, how, why)

QUESTION TYPES TO GENERATE:
- Factual verification questions (e.g., "What does the story say about...")
- Temporal questions (e.g., "When did [event] occur according to the story?")
- Location questions (e.g., "Where did [event] take place in the story?")
- Relationship questions (e.g., "How does the story describe the relationship between...")
- Action/Event questions (e.g., "How does the story describe [character]'s [action]?")

OUTPUT FORMAT:
Return a JSON array of question strings. Each question should be a complete, grammatically correct interrogative sentence ending with a question mark.

IMPORTANT: Generate only 2 questions per event to minimize API usage. Focus on the most critical aspects (what and when/where)."""

def generate_questions_from_event(event: Event, main_character: str, delay: float = 0.0) -> list[str]:
    """
    Generate specific, testable questions from an event using Groq LLM (qwen/qwen3-32b).
    
    These questions are designed to verify if the main story narrative mentions
    or is consistent with the backstory event.
    
    Args:
        event: The Event object containing subject, relation, object, time, location
        main_character: The main character name for context
        delay: Delay parameter (deprecated, kept for compatibility)
        
    Returns:
        List of question strings that can be used to query the story
    """
    # Delay removed for speed
    # Build comprehensive event description
    event_parts = []
    if event.subject:
        event_parts.append(f"Subject (Who): {event.subject}")
    if event.relation:
        event_parts.append(f"Action/Event (What): {event.relation}")
    if event.object:
        event_parts.append(f"Object/Target (To whom/what): {event.object}")
    if event.time:
        event_parts.append(f"Time (When): {event.time}")
    if event.location:
        event_parts.append(f"Location (Where): {event.location}")
    
    event_description = "\n".join(event_parts) if event_parts else "Event details not fully specified"
    
    user_prompt = f"""Generate 2-3 specific, testable questions based on the following backstory event. These questions will be used to verify if the main story narrative mentions or is consistent with this event.

BACKSTORY EVENT:
{event_description}

MAIN CHARACTER: {main_character}

INSTRUCTIONS:
1. Generate exactly 2 questions that can verify if the main story mentions or describes this event
2. Make questions specific and verifiable - they should be answerable by searching the story text
3. Include temporal context in questions if time information is available (e.g., "When did [event] occur?", "What happened at [time]?")
4. Include location context in questions if location information is available (e.g., "Where did [event] take place?", "What happened in [location]?")
5. Focus on what the story says about the event, not what should be true
6. Vary question types: factual, temporal, location, relationship, or action-based questions
7. Ensure questions are clear, unambiguous, and directly related to the event

EXAMPLES OF GOOD QUESTIONS:
- "What does the story say about {event.subject} {event.relation}?"
- "When did {event.subject} {event.relation} according to the story?"
- "Where did {event.subject} {event.relation} in the story?"
- "How does the story describe {event.subject}'s {event.relation}?"
- "What is mentioned about {event.subject} {event.relation} {event.object if event.object else ''}?"

Return a JSON array in this exact format with exactly 2 questions:
[
  "First specific question about the event?",
  "Second specific question about the event?"
]

Output ONLY the JSON array, no additional text or explanations."""

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
                    temperature=0.3,
                    max_completion_tokens=1024,
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
            
            # Remove markdown code blocks if present
            if "```json" in raw:
                start = raw.index("```json") + 7
                end = raw.index("```", start)
                raw = raw[start:end].strip()
            elif "```" in raw:
                start = raw.index("```") + 3
                end = raw.index("```", start)
                raw = raw[start:end].strip()
            
            # Handle reasoning text - find JSON array using pattern matching
            # Strategy: Look for JSON array pattern [ "string" ] to avoid brackets in reasoning text
            # Try to find a proper JSON array start: [ followed by whitespace and "
            json_array_start = re.search(r'\[\s*"', raw, re.DOTALL)
            
            if json_array_start:
                start = json_array_start.start()
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
                    parsed = json.loads(json_str)
                else:
                    # Bracket counting failed, try regex pattern
                    json_array_pattern = r'\[\s*"[^"]*"\s*(?:,\s*"[^"]*"\s*)*\]'
                    json_match = re.search(json_array_pattern, raw, re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group())
                    else:
                        raise ValueError("Could not find complete JSON array")
            elif "[" in raw and "]" in raw:
                # Fallback: try regex pattern for array of strings
                json_array_pattern = r'\[\s*"[^"]*"\s*(?:,\s*"[^"]*"\s*)*\]'
                json_match = re.search(json_array_pattern, raw, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                else:
                    # Last resort: find first [ and last ] (might include reasoning text)
                    start = raw.index("[")
                    end = raw.rindex("]") + 1
                    parsed = json.loads(raw[start:end])
            else:
                # Try to parse the entire response as JSON
                parsed = json.loads(raw)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"  ⚠️  JSON parsing error: {e}", flush=True)
            print(f"  Raw response (first 500 chars): {raw[:500]}...", flush=True)
            # Try more aggressive extraction patterns
            import re
            json_patterns = [
                r'\[\s*"[^"]*"\s*(?:,\s*"[^"]*"\s*)*\]',  # Array of strings
                r'\[.*?\]',  # Any array
            ]
            
            for pattern in json_patterns:
                json_match = re.search(pattern, raw, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group()
                        parsed = json.loads(json_str)
                        print(f"  ✓ Successfully extracted JSON using fallback pattern", flush=True)
                        break
                    except json.JSONDecodeError:
                        continue
            else:
                raise Exception(f"Could not parse JSON from response. First 500 chars: {raw[:500]}")
        
        # Validate and return questions
        questions = []
        for q in parsed:
            if isinstance(q, str) and q.strip():
                # Ensure question ends with question mark
                q = q.strip()
                if not q.endswith("?"):
                    q += "?"
                questions.append(q)
        
        return questions if questions else []
        
    except Exception as e:
        print(f"Error generating questions: {e}")
        # Fallback: generate a simple question if LLM parsing fails
        fallback_parts = []
        if event.subject:
            fallback_parts.append(f"what does the story say about {event.subject}")
        if event.relation:
            fallback_parts.append(f"{event.relation}")
        if event.object:
            fallback_parts.append(f"{event.object}")
        if event.time:
            fallback_parts.append(f"at {event.time}")
        if event.location:
            fallback_parts.append(f"in {event.location}")
        
        fallback = " ".join(fallback_parts).capitalize()
        if not fallback.endswith("?"):
            fallback += "?"
        return [fallback]
