import json
import os
from config import load_groq_api_key

# Suppress tokenizers warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ----------------------------
# MODEL CONFIG (DETERMINISTIC)
# ----------------------------
GROQ_MODEL = "llama-3.1-8b-instant"


# ----------------------------
# SYSTEM PROMPT (STRICT)
# ----------------------------
SYSTEM_PROMPT = """
You are a deterministic question-generation system for narrative consistency checking.

Your task:
Read a character backstory and generate a SMALL set of factual, testable questions
that can be answered by searching the main story text.

Rules (MANDATORY):
- Use ONLY explicit facts stated in the backstory
- Do NOT infer, assume, or paraphrase creatively
- One factual claim per question
- No yes/no questions
- Questions must be answerable from the story text

OUTPUT FORMAT (STRICT):
Return ONLY a valid JSON object in this exact form:

{
  "questions": [
    "Question 1?",
    "Question 2?"
  ]
}

Do NOT include explanations.
Do NOT include markdown.
Return JSON ONLY.
"""


# ----------------------------
# QUESTION GENERATION
# ----------------------------
def generate_questions_from_backstory(
    backstory: str,
    main_character: str,
) -> list[str]:

    user_prompt = f"""
CHARACTER: {main_character}

BACKSTORY:
{backstory}

TASK:
Generate 5–8 precise verification questions derived ONLY from explicit facts
in the backstory.

Return JSON ONLY using the required format.
"""

    from groq import Groq

    client = Groq(api_key=load_groq_api_key())

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            # 🔒 Determinism knobs
            temperature=0.0,
            top_p=0.1,
            max_completion_tokens=512,
            response_format={"type": "json_object"},
        )

        raw = completion.choices[0].message.content.strip()
        data = json.loads(raw)

        # ----------------------------
        # VALIDATION
        # ----------------------------
        if not isinstance(data, dict) or "questions" not in data:
            raise ValueError(f"Invalid JSON structure: {data}")

        questions = []
        for q in data["questions"]:
            if isinstance(q, str):
                q = q.strip()
                if q and not q.endswith("?"):
                    q += "?"
                questions.append(q)

        if not (5 <= len(questions) <= 8):
            raise ValueError(f"Expected 5–8 questions, got {len(questions)}")

        return questions

    except Exception as e:
        print(f"❌ Question generation failed (FATAL): {e}", flush=True)
        # Fail fast — never poison downstream evaluation
        raise RuntimeError("Question generation failed; aborting test case")
