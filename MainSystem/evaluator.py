"""
Evaluation module for narrative consistency checking
(FINAL – HARDENED, RAW-VISIBLE, PARSER-SAFE, TEMP=0)
"""

from typing import Dict, List
from pydantic import BaseModel
from config import load_gemini_api_key
import time
import re

# ----------------------------
# MODEL CONFIG
# ----------------------------
GEMINI_MODEL = "gemini-3-pro-preview"


# ----------------------------
# SCHEMA (SAFE DEFAULTS)
# ----------------------------
class EvalResult(BaseModel):
    answer: str = "NOT_MENTIONED"
    verdict: str = "uncertain"     # consistent | contradict | uncertain
    confidence: float = 0.0
    reasoning: str = "No explicit information found"


# ----------------------------
# JSON CLEANER (CRITICAL)
# ----------------------------
def clean_json(text: str) -> str:
    """
    Remove markdown, thinking tags, and junk around JSON.
    """
    if not text:
        return ""

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?", "", text)
    text = re.sub(r"```", "", text)

    return text.strip()


# ----------------------------
# GEMINI CALL (RAW FIRST)
# ----------------------------
def call_gemini_api(
    messages: List[Dict],
    model: str = GEMINI_MODEL,
    max_tokens: int = 512,
) -> EvalResult:

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.exceptions import OutputParserException

    api_key = load_gemini_api_key()

    print(f"        🧠 Initializing Gemini [{model}]", flush=True)

    if not api_key:
        print("        ❌ FATAL: Gemini API key missing", flush=True)
        return EvalResult(reasoning="Missing Gemini API key")

    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0.0,
        max_output_tokens=max_tokens,
    )

    # Convert messages to LangChain format
    lc_messages = []
    for msg in messages:
        if msg["role"] == "system":
            lc_messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))

    print(f"        📤 Sending {len(lc_messages)} messages to Gemini", flush=True)

    start = time.time()

    # ----------------------------
    # 1️⃣ RAW GEMINI CALL (ALWAYS)
    # ----------------------------
    try:
        raw_response = llm.invoke(lc_messages)
        raw_text = (
            raw_response.content
            if hasattr(raw_response, "content")
            else str(raw_response)
        )
    except Exception as e:
        print("        ❌ Gemini API call failed", flush=True)
        print(str(e), flush=True)
        return EvalResult(
            verdict="uncertain",
            reasoning="Gemini API call failed"
        )

    elapsed = time.time() - start

    print(f"\n{'='*80}")
    print("🧾 RAW GEMINI RESPONSE (EXACT)")
    print(repr(raw_text))
    print(f"{'='*80}\n")

    # ----------------------------
    # 2️⃣ CLEAN + PARSE
    # ----------------------------
    cleaned = clean_json(raw_text)
    parser = PydanticOutputParser(pydantic_object=EvalResult)

    try:
        result = parser.parse(cleaned)

    except OutputParserException as e:
        print("        ⚠️ JSON PARSING FAILED", flush=True)
        print("        CLEANED TEXT WAS:", flush=True)
        print(cleaned, flush=True)

        result = EvalResult(
            verdict="uncertain",
            reasoning="Invalid or non-JSON response from Gemini"
        )

    print(
        f"        📥 Gemini responded in {elapsed:.2f}s "
        f"| verdict={result.verdict} | conf={result.confidence:.2f}",
        flush=True,
    )

    return result


# ----------------------------
# EVALUATION
# ----------------------------
def evaluate_consistency(
    question: str,
    retrieved_chunks: List[str],
    backstory_facts: str,
    character: str = "",
) -> Dict:

    print(f"\n      🔍 Evaluating question:", flush=True)
    print(f"         Q: {question[:120]}", flush=True)

    if not retrieved_chunks:
        print("      ⚠️ No retrieved chunks → UNCERTAIN", flush=True)
        return {
            "verdict": "uncertain",
            "answer": "NOT_MENTIONED",
            "consistent": False,
            "confidence": 0.0,
            "reasoning": "No story passages provided",
            "retrieved_chunks": [],
        }

    context = "\n\n".join(retrieved_chunks[:4])[:6000]

    print(
        f"      📚 Using {min(len(retrieved_chunks),4)} chunks "
        f"({len(context)} chars)",
        flush=True,
    )

    # ----------------------------
    # PROMPTS
    # ----------------------------
    system_prompt = """
You are a precise narrative consistency evaluator.

You MUST return ONLY a valid JSON object with this exact schema:

{
  "answer": string,
  "verdict": "consistent" | "contradict" | "uncertain",
  "confidence": number,
  "reasoning": string
}

Rules:
- Use ONLY explicit information from the story
- Do NOT infer or assume
- JSON ONLY
- If unsure, return:
  {
    "answer": "NOT_MENTIONED",
    "verdict": "uncertain",
    "confidence": 0.0,
    "reasoning": "Insufficient information"
  }
"""

    user_prompt = f"""
CHARACTER: {character if character else "Not specified"}
QUESTION: {question}
BACKSTORY CLAIM: {backstory_facts}

STORY PASSAGES:
{context}

Evaluate consistency strictly.
Return JSON ONLY.
"""

    result = call_gemini_api(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    verdict = result.verdict.lower()
    if verdict not in {"consistent", "contradict", "uncertain"}:
        verdict = "uncertain"

    print(
        f"      ✅ Evaluation result: {verdict.upper()} "
        f"(conf={result.confidence:.2f})",
        flush=True,
    )
    print(f"         ↳ Answer: {result.answer[:120]}", flush=True)

    return {
        "verdict": verdict,
        "answer": result.answer,
        "consistent": verdict == "consistent",
        "confidence": float(result.confidence),
        "reasoning": result.reasoning,
        "retrieved_chunks": retrieved_chunks,
    }


# ----------------------------
# FINAL DECISION RULE
# ----------------------------
def apply_decision_rule(evaluations: List[Dict]) -> Dict:

    print(f"\n   🧮 Applying final decision rule...", flush=True)

    c = sum(e["verdict"] == "consistent" for e in evaluations)
    x = sum(e["verdict"] == "contradict" for e in evaluations)
    u = sum(e["verdict"] == "uncertain" for e in evaluations)

    print(
        f"      Signals → consistent={c}, contradict={x}, uncertain={u}",
        flush=True,
    )

    total = len(evaluations)

    if x > 0:
        print("      🚨 CONTRADICTION detected → FINAL=CONTRADICT", flush=True)
        return {
            "verdict": "contradict",
            "verdict_reason": f"{x} contradictions found",
            "confidence": round(min(0.7 + x / total * 0.3, 1.0), 2),
        }

    print("      🟢 No contradictions → FINAL=CONSISTENT", flush=True)
    return {
        "verdict": "consistent",
        "verdict_reason": "No contradictions detected",
        "confidence": round(0.6 + c / max(total, 1) * 0.3, 2),
    }
