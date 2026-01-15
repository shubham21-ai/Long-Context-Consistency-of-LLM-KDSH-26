"""
Question generation module for RAG-based narrative consistency checking.

Generates RAG-friendly questions directly from backstory using Gemini.
"""

import json
import re
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "config"))
from config import load_gemini_api_key
from google import genai

# Gemini model configuration
GEMINI_MODEL = "gemini-2.5-flash"  # Using latest Gemini model

SYSTEM_PROMPT = """You are an expert question generation system for RAG-based narrative consistency checking.

Your task is to generate a MINIMUM set of optimal, RAG-friendly questions from a character backstory that can be answered from story text using semantic search.

CRITICAL RULES:
1. CHARACTER NAME USAGE: ALWAYS use the explicit character name provided (NEVER use pronouns: he, she, they, his, her, their, him, her, them)
   - Pronouns are TERRIBLE for RAG retrieval - they don't match keywords in the story
   - Character names are ESSENTIAL for accurate semantic search
   
2. MINIMUM SET: Generate the MINIMUM number of questions needed to verify the backstory
   - Focus on the MOST critical and verifiable facts
   - Avoid redundant questions
   - Each question should test a DISTINCT aspect of the backstory
   
3. RAG-OPTIMIZATION:
   - Use explicit character names (NO pronouns)
   - Include specific actions, locations, times, relationships
   - Use concrete terms that will appear in story text
   - Questions should match phrases likely to appear in the story

4. QUESTION TYPES (prioritize in this order):
   - Temporal facts (when did X happen?)
   - Location facts (where did X happen?)
   - Specific actions/events (what did X do?)
   - Relationships (who did X interact with?)

OUTPUT FORMAT:
Return a JSON array of question strings, each ending with a question mark.
Example: ["When did [CHARACTER_NAME] [action]?", "Where did [CHARACTER_NAME] [action]?"]

Generate 3-7 questions (minimum set that covers all critical backstory claims)."""


def generate_questions_from_backstory(backstory: str, character: str) -> list[str]:
    """
    Generate RAG-friendly questions directly from backstory using Gemini.
    
    Args:
        backstory: The character backstory text
        character: The main character name
        
    Returns:
        List of question strings optimized for RAG retrieval
    """
    if not backstory or not backstory.strip():
        return []
    
    if not character or not character.strip():
        return []
    
    try:
        # Load Gemini API key and set environment variable (new API reads from env)
        api_key = load_gemini_api_key()
        os.environ['GEMINI_API_KEY'] = api_key
        
        # Initialize Gemini client (new API format)
        client = genai.Client()
        
        # Build prompt
        user_prompt = f"""Generate a MINIMUM set of RAG-friendly questions to verify this character backstory.

CHARACTER: {character}

BACKSTORY:
{backstory}

TASK: Generate 3-7 questions (minimum set) that can verify the critical facts in this backstory.

CRITICAL REQUIREMENTS:
1. ALWAYS use "{character}" explicitly in EVERY question (NO pronouns)
2. Generate MINIMUM number of questions needed (3-7 questions)
3. Each question should test a DISTINCT critical fact
4. Questions must be RAG-friendly (use keywords that appear in story text)
5. Focus on verifiable facts: times, locations, actions, relationships

BAD EXAMPLES (DO NOT USE - pronouns break RAG):
❌ "When did he learn to track?"
❌ "What happened to him?"
❌ "Where did they go?"

GOOD EXAMPLES (USE THESE - explicit names):
✅ "When did {character} learn to track?"
✅ "What does the story say about {character} learning to track?"
✅ "Where did {character} go during boyhood?"

Return ONLY a JSON array of questions:
[
  "Question 1 using {character} explicitly?",
  "Question 2 using {character} explicitly?",
  "Question 3 using {character} explicitly?"
]"""

        # Generate questions with retry logic
        max_retries = 3
        raw = None
        
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=user_prompt
                )
                raw = response.text
                break  # Success
            except Exception as e:
                error_str = str(e).lower()
                if 'rate' in error_str or 'limit' in error_str or '429' in error_str:
                    if attempt < max_retries - 1:
                        import time
                        retry_delay = 0.5 * (attempt + 1)
                        time.sleep(retry_delay)
                        continue
                if attempt == max_retries - 1:
                    raise
        
        if not raw:
            raise Exception("Empty response from Gemini")
        
        # Robust JSON extraction
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
        
        # Remove reasoning tags
        if "<think>" in raw.lower():
            raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL | re.IGNORECASE)
            raw = raw.strip()
        
        # Remove common prefixes
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
        
        # Strategy 1: Direct JSON parse
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
        
        # If all strategies failed, use fallback
        if parsed is None:
            raise Exception(f"Could not parse JSON from response")
        
        # Validate and clean questions
        questions = []
        for q in parsed:
            if isinstance(q, str) and q.strip():
                q = q.strip()
                
                # Ensure question ends with question mark
                if not q.endswith("?"):
                    q += "?"
                questions.append(q)
        
        return questions if questions else []
        
    except Exception as e:
        # Fallback: generate simple questions from backstory
        # Split backstory into sentences and create basic questions
        sentences = re.split(r'[.!?]+', backstory)
        fallback_questions = []
        
        for sentence in sentences[:5]:  # Limit to first 5 sentences
            sentence = sentence.strip()
            if sentence and len(sentence) > 20:  # Meaningful sentences
                # Clean sentence and create a simple question
                clean_sentence = re.sub(r'[^\w\s]', '', sentence[:50]).strip().lower()
                question = f"What does the story say about {character} {clean_sentence}?"
                if not question.endswith("?"):
                    question += "?"
                fallback_questions.append(question)
        
        return fallback_questions if fallback_questions else [f"What does the story say about {character}?"]
