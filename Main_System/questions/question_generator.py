import json
import time
import os
from models.schemas import Event
from config import load_groq_api_key

# Suppress tokenizers warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Groq API configuration
GROQ_MODEL = "qwen/qwen3-32b"

SYSTEM_PROMPT = """You are an expert question generation system for RAG-based narrative consistency checking.

Your task is to generate ONE optimal, pinpoint-accurate question per backstory event that can be answered from story text using semantic search.

CRITICAL RULE - CHARACTER NAME USAGE:
- ALWAYS use the explicit character name provided (never use pronouns: he, she, they, his, her, etc.)
- Pronouns are BAD for RAG retrieval - they don't match keywords in the story
- Character names are ESSENTIAL for accurate semantic search
- Even if the event mentions other subjects, always focus on the main character name in questions

RAG-FRIENDLY QUESTION PRINCIPLES:
1. Pinpoint Accuracy: Question must target the EXACT factual claim from the backstory
2. RAG-Optimized: Use explicit character names (NO pronouns), specific actions, locations, times
3. Single Focus: One question should test ONE specific claim (not multiple aspects)
4. Answerable: Must be answerable by finding text in the story (avoid abstract questions)
5. Verifiable: Question should have a clear yes/no/factual answer in the story
6. Explicit Names: ALWAYS use character names explicitly - this is CRITICAL for RAG retrieval

QUESTION STRUCTURE (choose the most critical aspect):
- If time is mentioned: Focus on temporal aspect ("When did [CHARACTER_NAME] [action]?")
- If location is mentioned: Focus on location ("Where did [CHARACTER_NAME] [action]?")
- If specific action/event: Focus on the action ("What does the story say about [CHARACTER_NAME] [action]?")
- If relationship/object: Focus on the relationship ("How does the story describe [CHARACTER_NAME]'s [relationship]?")

CRITICAL: Generate ONLY 1 question per event. It must:
1. Use the character name explicitly (NO pronouns - pronouns break RAG retrieval)
2. Be the MOST critical question that can definitively verify or contradict the backstory claim

OUTPUT FORMAT:
Return a JSON array with exactly ONE question string ending with a question mark. The question MUST use the character name explicitly."""

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
    
    user_prompt = f"""Generate ONE optimal, RAG-friendly question to verify if the story confirms or contradicts this backstory claim.

BACKSTORY EVENT:
{event_description}

MAIN CHARACTER: {main_character}

CRITICAL RULE - CHARACTER NAME USAGE (MANDATORY):
- You MUST use the character name "{main_character}" EXPLICITLY in the question
- NEVER use pronouns (he, she, they, his, her, their, him, her, them, etc.)
- Pronouns are TERRIBLE for RAG retrieval - they don't match keywords in story text
- ALWAYS use "{main_character}" directly, even if the event mentions other subjects
- Example: Use "When did {main_character} learn to track?" NOT "When did he learn to track?"

TASK: Generate the SINGLE most critical question that can definitively verify this backstory claim using semantic search.

SELECTION CRITERIA (choose the MOST critical aspect):
1. If TIME is mentioned → Temporal question (highest priority)
   Example: "When did {main_character} {event.relation}?"
   
2. If LOCATION is mentioned → Location question (second priority)
   Example: "Where did {main_character} {event.relation}?"
   
3. If SPECIFIC ACTION/EVENT → Action question (third priority)
   Example: "What does the story say about {main_character} {event.relation}?"
   
4. If RELATIONSHIP/OBJECT → Relationship question
   Example: "How does the story describe {main_character}'s {event.relation} {event.object if event.object else ''}?"

RAG-OPTIMIZATION REQUIREMENTS (MANDATORY):
- MUST include "{main_character}" explicitly in question (NO pronouns - this is CRITICAL)
- Include specific action/event keywords ({event.relation}) that appear in backstory
- Use concrete terms (names, places, actions) not abstract concepts
- Question should match phrases likely to appear in story text

BAD EXAMPLES (DO NOT USE - pronouns break RAG):
❌ "When did he learn to track?"
❌ "What happened to him?"
❌ "Where did they go?"

GOOD EXAMPLES (USE THESE - explicit names):
✅ "When did {main_character} learn to track?"
✅ "What does the story say about {main_character} learning to track?"
✅ "Where did {main_character} go?"

CRITICAL: Generate ONLY 1 question. It must:
1. Use "{main_character}" explicitly (NO pronouns)
2. Be the MOST definitive question to verify this specific claim

Return ONLY a JSON array with exactly ONE question using "{main_character}" explicitly:
[
  "Question using {main_character} explicitly?"
]

CRITICAL: Output ONLY valid JSON. Start with [ and end with ]. No markdown, no code blocks, no explanations."""

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
                    temperature=0.1,  # Lower temperature for more consistent, focused questions
                    max_completion_tokens=256,  # Reduced since we only need 1 question
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
                        import time
                        retry_delay = 0.5 * (attempt + 1)  # Exponential backoff
                        time.sleep(retry_delay)
                        continue
                # If not a rate limit error or max retries reached, raise
                if attempt == max_retries - 1:
                    raise
        
        # Robust JSON extraction with multiple fallbacks
        import re
        raw = raw.strip()
        
        # Remove markdown code blocks if present
        if "```json" in raw:
            start = raw.index("```json") + 7
            end = raw.index("```", start)
            raw = raw[start:end].strip()
        elif "```" in raw:
            start = raw.index("```") + 3
            end = raw.index("```", start)
            raw = raw[start:end].strip()
        
        # Remove reasoning tags (qwen model sometimes adds these)
        if "<think>" in raw.lower():
            raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL | re.IGNORECASE)
            raw = raw.strip()
        
        # Remove common prefixes that LLMs add
        prefixes_to_remove = [
            r'^okay,?\s*',
            r'^sure,?\s*',
            r'^here.*?:?\s*',
            r'^the.*?:?\s*',
        ]
        for prefix in prefixes_to_remove:
            raw = re.sub(prefix, '', raw, flags=re.IGNORECASE)
            raw = raw.strip()
        
        parsed = None
        
        # Strategy 1: Try direct JSON parse
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract JSON array using bracket counting
        if parsed is None:
            json_start = raw.find('[')
            if json_start != -1:
                try:
                    bracket_count = 0
                    in_string = False
                    escape_next = False
                    
                    for i in range(json_start, len(raw)):
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
                                    parsed = json.loads(raw[json_start:i+1])
                                    break
                except (ValueError, json.JSONDecodeError):
                    pass
        
        # Strategy 3: Regex pattern matching
        if parsed is None:
            json_patterns = [
                r'\[\s*"[^"]*"\s*(?:,\s*"[^"]*"\s*)*\]',  # Array of strings
                r'\[.*?\]',  # Any array (last resort)
            ]
            for pattern in json_patterns:
                json_match = re.search(pattern, raw, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                        break
                    except json.JSONDecodeError:
                        continue
        
        # Strategy 4: Find first [ and last ] (last resort)
        if parsed is None and "[" in raw and "]" in raw:
            try:
                start = raw.index("[")
                end = raw.rindex("]") + 1
                parsed = json.loads(raw[start:end])
            except (ValueError, json.JSONDecodeError):
                pass
        
        # If all strategies failed, use fallback (don't raise, let fallback handle it)
        if parsed is None:
            raise Exception(f"JSON parsing failed")
        
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
        # Fallback: generate a simple question if LLM parsing fails
        # (Error message suppressed to reduce noise - fallback will generate valid question)
        # IMPORTANT: Use main_character name explicitly (no pronouns)
        fallback_parts = []
        fallback_parts.append(f"what does the story say about {main_character}")  # Always use character name
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
