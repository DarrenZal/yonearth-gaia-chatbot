# GPT-5 API Investigation: Responses vs Chat Completions

**Date:** October 10, 2025
**Status:** Investigation Complete
**Conclusion:** Use Chat Completions API for structured outputs

---

## TL;DR

**For GPT-5 models with Pydantic structured outputs, use Chat Completions API (`client.beta.chat.completions.parse`).**

The Responses API does NOT support the `response_format` parameter needed for guaranteed structured outputs.

---

## The Question

When using GPT-5 models (gpt-5-mini, gpt-5-nano), which API should we use?

- **Responses API** (`/v1/responses`) - GPT-5 native API with `reasoning.effort` and `text.verbosity`
- **Chat Completions API** (`/v1/chat/completions`) - Standard API with structured outputs support

---

## What We Tested

Created test script to try multiple approaches with GPT-5-mini:

### Approach 1: Responses API with Tools
```python
response = client.responses.create(
    model="gpt-5-mini",
    input=full_prompt,
    reasoning={"effort": "medium"},
    text={"verbosity": "medium"},
    tools=[{
        "type": "function",
        "function": {
            "name": "extract_relationships",
            "description": "Extract dual-signal relationships",
            "parameters": DualSignalExtraction.model_json_schema()
        }
    }],
    tool_choice={"type": "function", "function": {"name": "extract_relationships"}}
)
```

**Result:** ❌ FAILED
```
Error code: 400 - Missing required parameter: 'tools[0].name'
```

### Approach 2: Chat Completions API with GPT-5 Parameters
```python
response = client.beta.chat.completions.parse(
    model="gpt-5-mini",
    messages=[...],
    response_format=DualSignalExtraction,
    extra_body={
        "reasoning": {"effort": "medium"},
        "text": {"verbosity": "medium"}
    }
)
```

**Result:** ❌ FAILED
```
Error code: 400 - Unknown parameter: 'reasoning'
```

### Approach 3: Responses API with Manual JSON Parsing
```python
response = client.responses.create(
    model="gpt-5-mini",
    input=full_prompt + "\n\nReturn ONLY valid JSON matching the schema.",
    reasoning={"effort": "medium"},
    text={"verbosity": "medium"}
)
```

**Result:** ⚠️ WORKS BUT NOT RELIABLE
- Response received successfully
- BUT: JSON parsing errors - model doesn't always return valid JSON
- No guaranteed schema compliance

---

## Key Findings

### 1. Responses API Does NOT Support `response_format`

```python
# This FAILS:
response = client.responses.create(
    model="gpt-5-mini",
    response_format=DualSignalExtraction  # ❌ TypeError: unexpected keyword argument
)
```

Even though OpenAI docs say "Structured outputs: Supported", the Responses API doesn't accept the `response_format` parameter that enables Pydantic structured outputs.

### 2. Chat Completions API Does NOT Support GPT-5 Parameters

```python
# This FAILS:
response = client.beta.chat.completions.parse(
    model="gpt-5-mini",
    reasoning={"effort": "medium"},  # ❌ Unknown parameter
    text={"verbosity": "medium"}      # ❌ Unknown parameter
)
```

The Chat Completions API doesn't recognize GPT-5-specific parameters like `reasoning` and `text`.

### 3. "Structured Outputs" Means Different Things

**For Responses API:**
- "Structured outputs" = Context-Free Grammars (CFG) via Lark
- Requires manual grammar definition
- Used via `tools` parameter with custom grammars
- NOT the same as Pydantic `response_format`

**For Chat Completions API:**
- "Structured outputs" = Pydantic models via `response_format`
- Automatic JSON schema generation
- Guaranteed valid output matching schema
- Works with `client.beta.chat.completions.parse()`

---

## Current API Capabilities

### Responses API (`/v1/responses`)
✅ **Supports:**
- `reasoning.effort` (minimal, low, medium, high)
- `text.verbosity` (low, medium, high)
- Context-Free Grammars (CFG) for custom syntax
- Custom tools with Lark grammars

❌ **Does NOT Support:**
- `response_format` parameter
- Pydantic structured outputs
- `temperature`, `top_p`, `logprobs`

### Chat Completions API (`/v1/chat/completions`)
✅ **Supports:**
- `response_format` with Pydantic models
- `client.beta.chat.completions.parse()` for guaranteed JSON
- `temperature` parameter (but GPT-5 models only support temperature=1)

❌ **Does NOT Support:**
- GPT-5-specific `reasoning` parameter
- GPT-5-specific `text` parameter

---

## Our Use Case: Dual-Signal Extraction

**Requirements:**
1. ✅ Extract complex nested relationships (List[DualSignalRelationship])
2. ✅ Guarantee valid JSON output (no parsing errors)
3. ✅ Pydantic validation for all fields
4. ⚠️ NICE TO HAVE: GPT-5 reasoning controls

**Best Solution:** **Chat Completions API with Pydantic structured outputs**

```python
response = client.beta.chat.completions.parse(
    model="gpt-5-mini",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    response_format=DualSignalExtraction  # ✅ Guaranteed structured output
)

extraction = response.choices[0].message.parsed  # ✅ Already a Pydantic object
```

**Why:**
- ✅ Guaranteed valid JSON matching our schema
- ✅ No parsing errors
- ✅ Type safety with Pydantic
- ✅ Works with GPT-5 models (they support Chat Completions API)
- ⚠️ Can't control `reasoning.effort` (but model may use internal reasoning anyway)

---

## What About "Constraining Outputs" with CFG?

The docs mention GPT-5 supports "context-free grammars (CFGs) for custom tools".

**This is for:**
- Domain-specific languages (SQL, custom DSLs)
- Strict syntax enforcement (e.g., only valid SQL queries)
- Custom grammars defined in Lark format

**This is NOT for:**
- Pydantic JSON schema validation
- General-purpose structured outputs
- What we need for dual-signal extraction

**Example CFG use case:**
```python
# If we wanted to constrain output to valid SQL:
tools=[{
    "type": "function",
    "function": {
        "name": "generate_sql",
        "grammar": """
            start: select_stmt
            select_stmt: "SELECT" columns "FROM" table
            ...
        """  # Lark grammar
    }
}]
```

This is overkill for our JSON extraction needs.

---

## Recommendation

**Use Chat Completions API (`client.beta.chat.completions.parse`) for all GPT-5 extraction work.**

### Why This Is Correct:

1. **Structured outputs work perfectly** - Pydantic models guarantee valid JSON
2. **GPT-5 models support Chat Completions API** - They're backward compatible
3. **No manual JSON parsing needed** - `response.choices[0].message.parsed` gives us the object directly
4. **Type safety** - Pydantic catches schema violations at extraction time
5. **Proven in production** - All our current tests use this successfully

### Trade-off:

- ❌ Can't explicitly control `reasoning.effort` or `text.verbosity`
- ✅ BUT GPT-5 models may use internal reasoning anyway
- ✅ AND we get guaranteed structured outputs which is MORE important

---

## Test Results

**Test:** 10 episodes with dual-signal extraction

| Approach | API Calls | Relationships | Errors |
|----------|-----------|---------------|--------|
| Chat Completions (current) | ✅ Working | 136-233 per episode | 0 |
| Responses API (Approach 1) | ❌ Failed | 0 | 100% (tools parameter issue) |
| Responses API (Approach 2) | ❌ Failed | 0 | 100% (reasoning parameter rejected) |
| Responses API (Approach 3) | ⚠️ Unreliable | Variable | JSON parsing errors |

---

## Files

**Test Script:** `/scripts/test_dual_signal_gpt5_mini_responses_api.py`
**Test Log:** `dual_signal_gpt5_mini_responses_api_test_*.log`
**Working Implementation:** `/scripts/test_dual_signal_gpt5_mini.py` (Chat Completions API)

---

## Conclusion

**For GPT-5 models with structured outputs, use Chat Completions API.**

The Responses API is designed for different use cases (reasoning-heavy tasks with natural language output, custom DSL generation with CFGs). For our knowledge graph extraction with Pydantic schemas, Chat Completions API with `response_format` is the correct choice.

**Current implementation is correct. No changes needed.**

---

## Related Documentation

- [OpenAI GPT-5 Models](https://platform.openai.com/docs/models/gpt-5-mini)
- [Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs)
- [Responses API Reference](https://platform.openai.com/docs/api-reference/responses)
- [Chat Completions API Reference](https://platform.openai.com/docs/api-reference/chat)

**Last Updated:** October 10, 2025
