# Framework Optimization Summary

## Improvements Made

### 1. Question Generation Optimization ✅

**Changes:**
- **Reduced from 2 questions per claim → 1 question per claim** (50% reduction in API calls)
- **Improved prompt** for RAG-friendly, pinpoint-accurate questions
- **Lower temperature** (0.3 → 0.1) for more consistent questions
- **Reduced token limit** (1024 → 256) since we only need 1 question
- **Optimized JSON parsing** (faster, simpler extraction)

**Benefits:**
- ⚡ 50% fewer API calls = faster execution
- 🎯 More focused questions = better RAG retrieval
- 💰 Lower token usage = cost savings

### 2. Performance Optimizations ✅

**Reduced Verbose Prints:**
- Removed unnecessary progress prints during question generation
- Removed verbose RAG query prints
- Removed detailed evaluation progress prints
- Kept only essential error messages and final results

**Benefits:**
- ⚡ Faster execution (less I/O overhead)
- 📊 Cleaner output (focus on results, not progress)

### 3. Rate Limiting ✅

**Added exponential backoff:**
- Rate limit retries now use exponential backoff (0.5s, 1s, 1.5s)
- Prevents API throttling issues

### 4. RAG-Friendly Question Design ✅

**New Question Generation Strategy:**
- **Priority-based selection**: Time > Location > Action > Relationship
- **Keyword optimization**: Includes character names and specific terms
- **Single focus**: One question tests one specific claim
- **Pinpoint accuracy**: Targets exact factual claims

**Example Improvements:**
- ❌ Old: "What does the story say about X? When did X happen?"
- ✅ New: "When did [character] [specific action]?" (if time mentioned)

## Performance Impact

### Before Optimization:
- ~2 questions per claim
- ~10-15 questions per test case (5-7 claims)
- ~15-20 API calls per test case
- Execution time: ~2-3 minutes per test case

### After Optimization:
- 1 question per claim
- ~5-7 questions per test case (5-7 claims)
- ~8-12 API calls per test case (40% reduction)
- Estimated execution time: ~1-1.5 minutes per test case (40-50% faster)

## Question Quality Improvements

### RAG-Friendly Features:
1. **Character names included** for better keyword matching
2. **Specific actions/events** (not abstract concepts)
3. **Concrete terms** (places, times, actions) that appear in story text
4. **Priority-based selection** ensures most critical aspects are tested first

### Pinpoint Accuracy:
- Questions target ONE specific factual claim
- No multi-part questions (easier to evaluate)
- Clear yes/no/factual answers expected

## Recommendations

1. **Monitor question quality**: Check if 1 question per claim is sufficient
2. **Adjust if needed**: Can increase to 1.5 questions per claim if coverage is insufficient
3. **Hybrid search**: Use `--hybrid-search` for better retrieval accuracy
4. **Batch processing**: Consider batching multiple test cases if needed

## Files Modified

1. `questions/question_generator.py` - Optimized question generation
2. `test_framework.py` - Reduced verbose prints
3. `evaluator.py` - Simplified decision rule (already done)

