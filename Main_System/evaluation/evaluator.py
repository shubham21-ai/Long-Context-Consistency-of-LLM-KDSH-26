"""
Evaluation module for narrative consistency checking.

This module handles:
- Single evaluation: Compare retrieved answers with backstory facts
- Decision rule: Aggregate evaluation results into final verdict
"""

import json
import time
import re
import os
from typing import Dict, List
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "config"))
from config import load_gemini_api_key
from google import genai

# Gemini API configuration
GEMINI_MODEL = "gemini-2.5-flash"


def call_gemini_api(prompt: str, temperature: float = 0.0, max_tokens: int = 512, 
                    max_retries: int = 3, model: str = GEMINI_MODEL) -> Dict:
    """
    Helper function to call Gemini API with retry logic.
    
    Args:
        prompt: The prompt text to send to Gemini
        temperature: Temperature for generation (0.0-2.0)
        max_tokens: Maximum tokens to generate
        max_retries: Maximum number of retry attempts
        model: Model name to use
        
    Returns:
        Response data as dictionary with 'choices' key compatible with existing code
    """
    # Load API key and set environment variable
    api_key = load_gemini_api_key()
    os.environ['GEMINI_API_KEY'] = api_key
    
    # Initialize Gemini client
    client = genai.Client()
    
    for attempt in range(max_retries):
        try:
            # Gemini API call
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            
            # Extract content from response
            content = response.text if hasattr(response, 'text') else str(response)
            
            # Format as OpenRouter-compatible response for compatibility
            response_data = {
                'choices': [{
                    'message': {
                        'content': content,
                        'role': 'assistant'
                    }
                }],
                'model': model,
                'usage': {
                    'prompt_tokens': 0,  # Gemini doesn't expose token counts easily
                    'completion_tokens': 0,
                    'total_tokens': 0
                }
            }
            
            return response_data
        except Exception as e:
            error_str = str(e).lower()
            if 'rate' in error_str or 'limit' in error_str or '429' in error_str or 'quota' in error_str:
                if attempt < max_retries - 1:
                    print(f"      ⚠️  Rate limit hit, retrying... (attempt {attempt + 1}/{max_retries})", flush=True)
                    retry_delay = 0.5 * (2 ** attempt)  # Exponential backoff
                    time.sleep(retry_delay)
                    continue
            # If not a rate limit error or max retries reached, raise
            if attempt == max_retries - 1:
                raise
    
    raise Exception("Max retries reached")



def evaluate_consistency(question: str, retrieved_chunks: List[str], 
                        backstory_facts: str, character: str = "") -> Dict:
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
        character: Character name to focus on
        
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
    
    # Use more chunks (up to 8 instead of 5)
    combined_context = "\n\n".join(retrieved_chunks[:8])
    
    print(f"      → Starting evaluation for question...", flush=True)
    
    # Improved evaluation prompt with character focus
    char_context = f" about the character {character}" if character else ""
    
    evaluation_prompt = f"""You are evaluating narrative consistency. Extract information from story passages{char_context}.

CHARACTER: {character if character else "Not specified"}
QUESTION: {question}
BACKSTORY CONTEXT (for reference): {backstory_facts}

STORY PASSAGES:
{combined_context[:6000]}

INSTRUCTIONS:

STEP 1: UNDERSTAND THE QUESTION
Focus ONLY on what the QUESTION is asking. The question specifies exactly what aspect to verify.

STEP 2: SEARCH FOR INFORMATION (related to the QUESTION only)
Look for information in the story passages that answers the QUESTION:
- Mentions of "{character}" related to what the question asks
- Events, actions, or situations that answer the question
- DO NOT look for information NOT mentioned in the question
- DO NOT check for facts from the backstory that are not in the question

IMPORTANT: The story may use different words than the question. Examples:
- Question: "learned to track" → Story: "taught tracking", "mastered tracking", "tracking skills"
- Question: "{character}" → Story: "the character", descriptive phrases
- Look for the MEANING, not exact words

STEP 3: EXTRACT WHAT THE STORY SAYS (about the QUESTION)
Write what you found in the passages that relates to the QUESTION. Be specific. Quote relevant parts if needed.
If you find NOTHING related to the question, write "NOT_MENTIONED"

STEP 4: EVALUATE CONSISTENCY (for the QUESTION only)
- Does the story answer the question in a way that SUPPORTS the backstory? → CONSISTENT
- Does the story answer the question in a way that CONTRADICTS the backstory? → INCONSISTENT  
- Is there NO relevant information in the passages about the question? → UNCERTAIN

VERDICT DEFINITIONS:
✓ CONSISTENT: Story provides information that supports what the question asks about
  - Story confirms or supports the fact being questioned
  - Compatible information found (even if described differently)
  - Supporting evidence for what the question asks

✗ INCONSISTENT: Story provides information that contradicts what the question asks about
  - Story explicitly contradicts the fact being questioned
  - Contradictory information found
  - Story denies or contradicts what the question asks

? UNCERTAIN: Story has NO relevant information about what the question asks
  - No information found related to the question
  - Character {character} not mentioned in relation to the question
  - Passages are about completely different topics
  
CRITICAL RULES:
1. Focus ONLY on what the QUESTION asks - ignore other backstory facts not in the question
2. Extract information actively - if you see {character} or related events, extract them
3. Only use UNCERTAIN if passages have ZERO relevant information about the question
4. Different wording is OK - focus on whether facts align or contradict
5. Be precise: "not mentioned" ≠ "contradicted"
6. The backstory context is for reference only - focus on answering the question

Output ONLY this JSON (no markdown, no explanation, no other text):
{{
  "answer": "extracted information from story, or NOT_MENTIONED",
  "verdict": "CONSISTENT|INCONSISTENT|UNCERTAIN",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation of your verdict"
}}"""

    try:
        # Single API call to evaluate
        print(f"      → Calling Gemini API for evaluation...", flush=True)
        
        # Combine system and user messages into a single prompt for Gemini
        system_message = f"You are an expert at evaluating narrative consistency. Extract information carefully about {character if character else 'the story'} and provide detailed comparisons. Be aggressive in finding relevant information."
        full_prompt = f"{system_message}\n\n{evaluation_prompt}"
        
        response_data = call_gemini_api(
            prompt=full_prompt,
            temperature=0.1,
            max_tokens=1024,
            model=GEMINI_MODEL
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
        
        # Remove reasoning tags
        if "<think>" in raw.lower():
            raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL | re.IGNORECASE)
            raw = raw.strip()
        
        # Try to parse JSON
        parsed = None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON object
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
        
        if not parsed:
            raise Exception(f"Could not parse JSON from response: {raw[:200]}")
        
        # Extract fields
        verdict = parsed.get('verdict', 'UNCERTAIN').upper()
        answer = parsed.get('answer', 'NOT_MENTIONED')
        confidence = float(parsed.get('confidence', 0.5))
        reasoning = parsed.get('reasoning', '')
        
        # Validate verdict
        if verdict not in ['CONSISTENT', 'INCONSISTENT', 'UNCERTAIN']:
            verdict = 'UNCERTAIN'
        
        # Determine consistency boolean
        consistent = (verdict == 'CONSISTENT')
        
        print(f"      ✓ Successfully parsed evaluation result: {verdict}", flush=True)
        
        return {
            'verdict': verdict,
            'answer': answer,
            'consistent': consistent,
            'confidence': confidence,
            'reasoning': reasoning
        }
        
    except Exception as e:
        print(f"      ✗ Error in evaluation: {str(e)}", flush=True)
        import traceback
        print(f"      Traceback: {traceback.format_exc()[:300]}...", flush=True)
        return {
            'verdict': 'UNCERTAIN',
            'answer': 'ERROR',
            'consistent': False,
            'confidence': 0.0,
            'reasoning': f'Error during evaluation: {str(e)[:100]}'
        }


def apply_decision_rule(evaluations: List[Dict]) -> Dict:
    """
    Apply decision rule to aggregate evaluation results into final verdict.
    
    Simplified decision rule:
    1. If at least 1 INCONSISTENT → INCONSISTENT (any contradiction invalidates backstory)
    2. All other cases → CONSISTENT (no contradictions found)
    
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
    
    # Apply decision rule: Any contradiction → INCONSISTENT, else CONSISTENT
    if inconsistent_count > 0:
        final_verdict = 'INCONSISTENT'
        verdict_reason = f"Found {inconsistent_count} contradiction(s): {consistent_count} consistent, {uncertain_count} uncertain out of {len(evaluations)} evaluations"
    else:
        final_verdict = 'CONSISTENT'
        verdict_reason = f"No contradictions found: {consistent_count} consistent, {uncertain_count} uncertain out of {len(evaluations)} evaluations"
    
    # Calculate confidence (simplified)
    total_evaluations = len(evaluations)
    if total_evaluations == 0:
        confidence = 0.0
    elif inconsistent_count > 0:
        # High confidence if contradictions found
        confidence = min(0.9, 0.5 + (inconsistent_count / total_evaluations) * 0.4)
    else:
        # Moderate confidence if no contradictions
        confidence = min(0.9, 0.5 + (consistent_count / total_evaluations) * 0.4)
    
    return {
        'verdict': final_verdict,
        'verdict_reason': verdict_reason,
        'confidence': confidence,
        'signals': {
            'consistent_count': consistent_count,
            'inconsistent_count': inconsistent_count,
            'uncertain_count': uncertain_count,
            'total_evaluations': total_evaluations
        }
    }
