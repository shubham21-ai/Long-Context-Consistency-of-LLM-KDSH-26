# How Rationale is Generated in results.csv

## Overview

The `Rationale` column in `results.csv` contains AI-generated explanations for the final verdict (CONSISTENT/INCONSISTENT). These rationales are extracted from the evaluation results in `test_results.json`.

## Process Flow

1. **Question Generation**: Multiple RAG-friendly questions are generated from the character backstory
2. **RAG Retrieval**: Each question is used to retrieve relevant story chunks
3. **Evaluation**: Each question-chunk pair is evaluated for consistency
4. **Decision Rule**: Final verdict is determined (CONSISTENT if no contradictions, INCONSISTENT if any contradiction)
5. **Rationale Selection**: Rationale is selected from the evaluation reasonings based on the final verdict

## Rationale Selection Logic

The rationale is selected from evaluation reasonings with the following priority:

### For INCONSISTENT Verdicts (Prediction=0):
- **Uses**: First INCONSISTENT evaluation's reasoning
- **Why**: The first contradiction explains why the backstory is inconsistent
- **Example**: "The backstory states X, but the story passages show Y instead..."

### For CONSISTENT Verdicts (Prediction=1):

**Priority 1: CONSISTENT Evaluation Reasoning**
- If 1 CONSISTENT evaluation exists: Uses that reasoning
- If 2+ CONSISTENT evaluations exist: Combines first 2 reasonings with space separator
- **Why**: These explain how the story supports the backstory

**Priority 2: UNCERTAIN Evaluation Reasoning (Fallback)**
- If no CONSISTENT evaluations exist: Uses first UNCERTAIN evaluation's reasoning
- **Why**: Explains why the verdict is CONSISTENT despite uncertainty (no contradictions found)
- **Common case**: "The provided story passages do not mention the character X, therefore there is no information..."

## Examples

### Example 1: ID 30 (CONSISTENT with 1 CONSISTENT evaluation)
- **5 evaluations**: 1 CONSISTENT, 4 UNCERTAIN
- **Rationale selected**: The single CONSISTENT evaluation's reasoning
- **Rationale**: "The backstory states that Tom Ayrton/Ben Joyce became second mate within three years thanks to his 'sailing skill'. The passages confirm he was a 'sailor of the Britannia' and a 'famous quartermaster on board the Britannia'."

### Example 2: ID 42 (CONSISTENT with 1 CONSISTENT evaluation)
- **3 evaluations**: 1 CONSISTENT, 2 UNCERTAIN
- **Rationale selected**: The single CONSISTENT evaluation's reasoning
- **Rationale**: "The story passages explicitly confirm that Tom Ayrton was 'engaged as quartermaster' on the 'brig Britannia of Glasgow,' supported by a contract. This directly confirms he accepted a berth on the Britannia."

### Example 3: ID 3 (CONSISTENT with only UNCERTAIN evaluations)
- **4 evaluations**: 0 CONSISTENT, 0 INCONSISTENT, 4 UNCERTAIN
- **Rationale selected**: First UNCERTAIN evaluation's reasoning (fallback)
- **Rationale**: "The provided story passages do not mention the character Jacques Paganel at all. Therefore, there is no information regarding his mother's death or any immediate events involving him following such a death."

### Example 4: ID 56 (INCONSISTENT)
- **Evaluations**: Contains INCONSISTENT evaluations
- **Rationale selected**: First INCONSISTENT evaluation's reasoning
- **Rationale**: "The backstory context for the question states that Thalcave predicted an 'earthquake'. However, the story passages describe Thalcave predicting a 'pampero' (a violent wind/storm) based on his observation of the sky."

## Truncation

Rationales are truncated to approximately **250 characters** to fit the submission format:
- Truncation happens at sentence boundaries when possible
- If no sentence boundary found, truncates at 247 characters and adds "..."
- This ensures the rationale fits the CSV format while preserving readability

## Source

All rationales come from the `evaluations` array in `test_results.json`, specifically from the `reasoning` field of each evaluation result.

