"""
Evaluation module for narrative consistency checking.

This module handles:
- Single evaluation: Compare retrieved answers with backstory facts
- Decision rule: Aggregate evaluation results into final verdict
"""

import json
import time
import re
from typing import Dict, List
from config import load_groq_api_key

# Groq API configuration
GROQ_MODEL = "qwen/qwen3-32b"


def call_groq_api(messages: List[Dict], temperature: float = 0.0, max_tokens: int = 512, 
                  max_retries: int = 3, delay: float = 0.0, model: str = GROQ_MODEL) -> Dict:
    """
    Helper function to call Groq API with retry logic.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        max_retries: Maximum number of retry attempts
        delay: Delay in seconds before making the API call (deprecated, kept for compatibility)
        model: Model name to use
        
    Returns:
        Response data as dictionary with 'choices' key compatible with existing code
    """
    # Delay removed for speed
    
    # Import Groq client
    from groq import Groq
    
    # Load API key from .env when needed
    api_key = load_groq_api_key()
    client = Groq(api_key=api_key)
    retry_delay = 0
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                stream=False
            )
            
            # Convert Groq response to OpenRouter-compatible format
            if not completion.choices or len(completion.choices) == 0:
                raise Exception(f"No choices in response")
            
            # Extract content and format as OpenRouter-compatible response
            message = completion.choices[0].message
            content = message.content if hasattr(message, 'content') else str(message)
            
            # Format as OpenRouter-compatible response
            response_data = {
                'choices': [{
                    'message': {
                        'content': content,
                        'role': 'assistant'
                    }
                }],
                'model': completion.model if hasattr(completion, 'model') else model,
                'usage': {
                    'prompt_tokens': completion.usage.prompt_tokens if hasattr(completion, 'usage') and hasattr(completion.usage, 'prompt_tokens') else 0,
                    'completion_tokens': completion.usage.completion_tokens if hasattr(completion, 'usage') and hasattr(completion.usage, 'completion_tokens') else 0,
                    'total_tokens': completion.usage.total_tokens if hasattr(completion, 'usage') and hasattr(completion.usage, 'total_tokens') else 0
                }
            }
            
            return response_data
        except Exception as e:
            error_str = str(e).lower()
            if 'rate' in error_str or 'limit' in error_str or '429' in error_str:
                if attempt < max_retries - 1:
                    print(f"      ⚠️  Rate limit hit, retrying immediately... (attempt {attempt + 1}/{max_retries})", flush=True)
                    continue
            # If not a rate limit error or max retries reached, raise
            if attempt == max_retries - 1:
                raise
    
    raise Exception("Max retries reached")



def evaluate_consistency(question: str, retrieved_chunks: List[str], 
                        backstory_facts: str, character: str = "") -> Dict:  # ADDED: character parameter
    """
    Single evaluation: Compare retrieved answer from story chunks with backstory facts.
    
    This is the ONLY evaluation step - it:
    1. Extracts answer from retrieved story chunks
    2. Compares it with backstory facts
    3. Returns verdict (CONSISTENT, INCONSISTENT, or UNCERTAIN)
    
    Args:
        question: Question about the backstory fact
        retrieved_chunks: Retrieved text chunks from the story
        backstory_facts: The relevant backstory facts/claim
        character: Character name to focus on (ADDED)
        
    Returns:
        Dictionary with verdict, answer, and reasoning
    """
    if not retrieved_chunks:
        return {
            'verdict': 'UNCERTAIN',
            'answer': 'NOT_MENTIONED',
            'consistent': False,
            'confidence': 0.0,
            'reasoning': 'No chunks retrieved to answer the question'
        }
    
    # CHANGE 1: Use more chunks (up to 8 instead of 5)
    combined_context = "\n\n".join(retrieved_chunks[:8])
    
    print(f"      → Starting evaluation for question...", flush=True)
    
    # CHANGE 2: Improved evaluation prompt with character focus
    char_context = f" about the character {character}" if character else ""
    
    evaluation_prompt = f"""You are evaluating narrative consistency. Extract information from story passages{char_context}.

CHARACTER: {character if character else "Not specified"}
QUESTION: {question}
BACKSTORY CLAIM: {backstory_facts}

STORY PASSAGES:
{combined_context[:6000]}

INSTRUCTIONS:

STEP 1: SEARCH FOR INFORMATION
Look for ANY of these in the story passages:
- Mentions of "{character}" (the character)
- Events, actions, or situations related to the question
- Indirect references or implications
- Related characters or events even if described differently

IMPORTANT: The story may use different words than the backstory. Examples:
- Backstory: "engineered" → Story: "arranged", "orchestrated", "planned"
- Backstory: "Villefort" → Story: "the prosecutor", "his son"
- Look for the MEANING, not exact words

STEP 2: EXTRACT WHAT THE STORY SAYS
Write what you found in the passages. Be specific. Quote relevant parts if needed.
If you find NOTHING related to {character} or the question topic, write "NOT_MENTIONED"

STEP 3: COMPARE WITH BACKSTORY
- Does the story SUPPORT the backstory claim? → CONSISTENT
- Does the story CONTRADICT the backstory claim? → INCONSISTENT  
- Is there NO relevant information in the passages? → UNCERTAIN

VERDICT DEFINITIONS:
✓ CONSISTENT: Story mentions events/facts that match or support the backstory
  - Same character doing similar actions
  - Compatible information (even if described differently)
  - Supporting evidence for the claim

✗ INCONSISTENT: Story actively contradicts the backstory
  - Different facts (X did Y, but story says X did Z)
  - Contradictory dates, events, or relationships
  - Story explicitly denies or contradicts the claim

? UNCERTAIN: Story has NO relevant information
  - Character {character} not mentioned in these passages
  - No events related to the question
  - Passages are about completely different topics
  
CRITICAL RULES:
1. Extract information actively - if you see {character} or related events, extract them
2. Only use UNCERTAIN if passages have ZERO relevant information
3. Different wording is OK - focus on whether facts align or contradict
4. Be precise: "not mentioned" ≠ "contradicted"

Output ONLY this JSON (no markdown, no explanation, no other text):
{{
  "answer": "extracted information from story, or NOT_MENTIONED",
  "verdict": "CONSISTENT|INCONSISTENT|UNCERTAIN",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation of your verdict"
}}"""

    try:
        # Single API call to evaluate
        print(f"      → Calling Groq API for evaluation...", flush=True)
        response_data = call_groq_api(
            messages=[
                {"role": "system", "content": f"You are an expert at evaluating narrative consistency. Extract information carefully about {character if character else 'the story'} and provide detailed comparisons. Be aggressive in finding relevant information."},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.1,
            max_tokens=1024
        )
        print(f"      → API call completed, parsing response...", flush=True)
        
        # Parse JSON response (existing parsing logic remains the same)
        message = response_data['choices'][0]['message']
        response_text = message.get('content', '').strip()
        if not response_text:
            raise Exception(f"Empty response")
        
        # Extract JSON (handle reasoning if present)
        raw = response_text.strip()
        
        # Remove markdown code blocks if present
        if "```json" in raw:
            start = raw.index("```json") + 7
            end = raw.index("```", start)
            raw = raw[start:end].strip()
        elif "```" in raw:
            start = raw.index("```") + 3
            end = raw.index("```", start)
            raw = raw[start:end].strip()
        
        # Look for JSON object pattern { ... }
        json_object_start = re.search(r'\{\s*"', raw, re.DOTALL)
        
        if json_object_start:
            start = json_object_start.start()
            # Find matching closing brace by counting
            brace_count = 0
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
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i + 1
                            break
            
            if end > start:
                # Found matching braces
                json_str = raw[start:end]
                try:
                    result = json.loads(json_str)
                    verdict = result.get('verdict', 'UNCERTAIN').upper()
                    if verdict not in ['CONSISTENT', 'INCONSISTENT', 'UNCERTAIN']:
                        verdict = 'UNCERTAIN'
                    
                    print(f"      ✓ Successfully parsed evaluation result: {verdict}", flush=True)
                    return {
                        'verdict': verdict,
                        'answer': result.get('answer', 'NOT_MENTIONED'),
                        'consistent': verdict == 'CONSISTENT',
                        'confidence': float(result.get('confidence', 0.5)),
                        'reasoning': result.get('reasoning', '')
                    }
                except json.JSONDecodeError as je:
                    print(f"      ⚠️  First parsing attempt failed, trying fallback...", flush=True)
                    pass
        
        # Fallback: try to find JSON object using bracket counting from any { position
        if "{" in raw:
            start = raw.index("{")
            brace_count = 0
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
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i + 1
                            break
            
            if end > start:
                try:
                    json_str = raw[start:end]
                    result = json.loads(json_str)
                    verdict = result.get('verdict', 'UNCERTAIN').upper()
                    if verdict not in ['CONSISTENT', 'INCONSISTENT', 'UNCERTAIN']:
                        verdict = 'UNCERTAIN'
                    
                    return {
                        'verdict': verdict,
                        'answer': result.get('answer', 'NOT_MENTIONED'),
                        'consistent': verdict == 'CONSISTENT',
                        'confidence': float(result.get('confidence', 0.5)),
                        'reasoning': result.get('reasoning', '')
                    }
                except json.JSONDecodeError:
                    pass
        
        # If all parsing fails, try to extract verdict and answer from text
        verdict_match = re.search(r'(CONSISTENT|INCONSISTENT|UNCERTAIN)', raw, re.IGNORECASE)
        answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', raw)
        
        verdict = 'UNCERTAIN'
        if verdict_match:
            verdict = verdict_match.group(1).upper()
        
        answer = 'NOT_MENTIONED'
        if answer_match:
            answer = answer_match.group(1)
        
        return {
            'verdict': verdict,
            'answer': answer,
            'consistent': verdict == 'CONSISTENT',
            'confidence': 0.3,
            'reasoning': f'Extracted from response text (parsing failed): {raw[:200]}'
        }
    except Exception as e:
        print(f"  ⚠️  Evaluation error: {e}", flush=True)
        return {
            'verdict': 'UNCERTAIN',
            'answer': 'NOT_MENTIONED',
            'consistent': False,
            'confidence': 0.0,
            'reasoning': f'Evaluation error: {str(e)[:100]}'
        }


# CHANGE 3: Improved decision rule (optional improvement)
def apply_decision_rule(evaluations: List[Dict]) -> Dict:
    """
    Decision rule: Aggregate results from evaluations.
    
    LOGIC:
    1. ANY inconsistency found → INCONSISTENT (backstory contradicted)
    2. Mostly UNCERTAIN (70%+) → INCONSISTENT (backstory not supported by evidence)
    3. Majority CONSISTENT (50%+) with no inconsistencies → CONSISTENT
    4. Some CONSISTENT but many UNCERTAIN → LIKELY_CONSISTENT
    5. All UNCERTAIN → UNCERTAIN
    
    Args:
        evaluations: List of evaluation results, each with 'verdict'
        
    Returns:
        Dictionary with final verdict and aggregated signals
    """
    # Count verdicts
    consistent_count = 0
    inconsistent_count = 0
    uncertain_count = 0
    
    for eval_result in evaluations:
        verdict = eval_result.get('verdict', 'UNCERTAIN').upper()
        if verdict == 'CONSISTENT':
            consistent_count += 1
        elif verdict == 'INCONSISTENT':
            inconsistent_count += 1
        else:
            uncertain_count += 1
    
    total_evaluations = len(evaluations)
    
    # RULE 1: ANY inconsistency found → INCONSISTENT
    # If story contradicts backstory even once, backstory is wrong
    if inconsistent_count > 0:
        verdict = "INCONSISTENT"
        verdict_reason = f"{inconsistent_count} inconsistency/inconsistencies found out of {total_evaluations} evaluations"
        confidence = min(0.7 + (inconsistent_count / total_evaluations) * 0.3, 1.0)
    
    # RULE 2: Mostly UNCERTAIN (70%+) → INCONSISTENT
    # If we can't find evidence for most claims, backstory is not supported
    elif uncertain_count >= total_evaluations * 0.7:
        verdict = "INCONSISTENT"
        verdict_reason = f"Backstory not supported by text: {uncertain_count}/{total_evaluations} claims have no evidence"
        confidence = 0.65
    
    # RULE 3: Majority CONSISTENT (50%+) and no inconsistencies → CONSISTENT
    elif consistent_count >= total_evaluations * 0.5 and inconsistent_count == 0:
        verdict = "CONSISTENT"
        verdict_reason = f"{consistent_count}/{total_evaluations} evaluations support the backstory"
        confidence = min(0.6 + (consistent_count / total_evaluations) * 0.3, 0.95)
    
    # RULE 4: Some CONSISTENT but many UNCERTAIN → LIKELY_CONSISTENT
    # We found some supporting evidence, but missing evidence for many claims
    elif consistent_count > 0 and inconsistent_count == 0:
        verdict = "LIKELY_CONSISTENT"
        verdict_reason = f"Partial support: {consistent_count} consistent, {uncertain_count} uncertain out of {total_evaluations}"
        confidence = 0.4 + (consistent_count / total_evaluations) * 0.2
    
    # RULE 5: All UNCERTAIN → UNCERTAIN
    else:
        verdict = "UNCERTAIN"
        verdict_reason = f"No evidence found: {uncertain_count}/{total_evaluations} evaluations are uncertain"
        confidence = 0.3
    
    return {
        'verdict': verdict,
        'verdict_reason': verdict_reason,
        'confidence': confidence,
        'signals': {
            'consistent_count': consistent_count,
            'inconsistent_count': inconsistent_count,
            'uncertain_count': uncertain_count,
            'total_evaluations': total_evaluations
        }
    }