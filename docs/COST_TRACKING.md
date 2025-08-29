# Cost Tracking Documentation

The YonEarth Gaia Chatbot includes comprehensive cost tracking for all API usage, providing transparency into the costs associated with each response.

## Overview

The cost tracking system calculates and displays costs for:
- **OpenAI LLM usage** (GPT-3.5-turbo, GPT-4, etc.)
- **OpenAI Embeddings** (text-embedding-3-small/large)
- **ElevenLabs Voice Generation** (character-based pricing)

## How It Works

### Cost Calculation

1. **Token Counting**: Uses OpenAI's `tiktoken` library for accurate token counting
2. **Character Counting**: Counts characters for ElevenLabs voice generation
3. **Real-time Calculation**: Costs are calculated during response generation
4. **Aggregate Display**: Shows total cost and breakdown by service

### Pricing Sources

#### OpenAI Pricing (as of August 2024)
- **GPT-3.5-turbo**: $0.50/1M input tokens, $1.50/1M output tokens
- **GPT-4**: $30/1M input tokens, $60/1M output tokens
- **GPT-4o**: $5/1M input tokens, $15/1M output tokens
- **GPT-4o-mini**: $0.15/1M input tokens, $0.60/1M output tokens
- **Embeddings**: $0.02/1M tokens (small), $0.13/1M tokens (large)

Source: https://openai.com/api/pricing/

#### ElevenLabs Pricing
Default pricing is based on the Creator plan:
- **$22/month for 110,000 characters**
- **Effective rate**: $0.20 per 1,000 characters

**Important**: ElevenLabs doesn't return cost information via API. The system calculates costs based on character count and your configured rate.

### Configuration

You can customize the ElevenLabs pricing to match your subscription:

```bash
# In .env file
ELEVENLABS_COST_PER_1K_CHARS=0.20  # Adjust based on your plan
```

To calculate your rate:
1. Check your monthly plan cost
2. Check your monthly character allowance
3. Calculate: (Plan Cost / Character Allowance) * 1000

Example calculations:
- **Creator Plan**: $22 / 110,000 × 1000 = $0.20 per 1K chars
- **Pro Plan**: $99 / 500,000 × 1000 = $0.198 per 1K chars
- **Scale Plan**: $330 / 2,000,000 × 1000 = $0.165 per 1K chars

## User Interface

### Cost Display Location
The cost breakdown appears:
- After the feedback section in each response
- Only for Gaia's responses (not errors or user messages)
- With a collapsible detail view

### UI Components

1. **Total Cost**: Prominently displayed (e.g., "$0.0234")
2. **Service Breakdown**: Shows each API service used
3. **Usage Details**: Token counts, character counts
4. **Toggle Button**: Show/hide detailed breakdown

### Example Display
```
Cost Breakdown                                    $0.0234
─────────────────────────────────────────────────────────
[Show details]

When expanded:
┌─────────────────────────────────────────────────────────┐
│ OpenAI LLM      gpt-3.5-turbo   874 in / 205 out   $0.0007 │
│ OpenAI Embeddings  text-embedding-3-small  220 tokens  $0.0000 │
│ ElevenLabs Voice   eleven_multilingual_v2  780 chars   $0.0156 │
└─────────────────────────────────────────────────────────┘
```

## API Response Format

The cost breakdown is included in all chat API responses:

```json
{
  "response": "Gaia's response text...",
  "sources": [...],
  "cost_breakdown": {
    "summary": "$0.0234",
    "details": [
      {
        "service": "OpenAI LLM",
        "model": "gpt-3.5-turbo",
        "usage": "874 in / 205 out tokens",
        "cost": "$0.0007"
      },
      {
        "service": "OpenAI Embeddings",
        "model": "text-embedding-3-small",
        "usage": "220 tokens",
        "cost": "$0.0000"
      },
      {
        "service": "ElevenLabs Voice",
        "model": "eleven_multilingual_v2",
        "usage": "780 characters",
        "cost": "$0.0156"
      }
    ]
  }
}
```

## Cost Optimization Tips

1. **Reduce Token Usage**:
   - Use concise prompts
   - Limit max_citations parameter
   - Use GPT-3.5-turbo for most queries

2. **Voice Optimization**:
   - Disable voice for testing
   - Voice costs are typically the highest component
   - Consider selective voice enablement

3. **Model Selection**:
   - GPT-3.5-turbo: Best value for most queries
   - GPT-4: Use only for complex reasoning
   - GPT-4o-mini: Good balance of cost and capability

## Limitations

1. **Estimates Only**: Displayed costs are estimates based on:
   - Published pricing tiers
   - Character/token counting
   - May not reflect volume discounts or special pricing

2. **No Historical Tracking**: Current implementation shows per-response costs only
   - No daily/monthly aggregation
   - No cost history storage

3. **API Limitations**: 
   - ElevenLabs doesn't provide cost via API
   - OpenAI costs are calculated, not returned by API

## Future Enhancements

Potential improvements for cost tracking:
- Historical cost tracking and analytics
- Daily/monthly cost summaries
- Cost alerts and budgets
- Export cost data for accounting
- Real-time quota monitoring
- Cost optimization suggestions

## Troubleshooting

### Incorrect ElevenLabs Costs
If voice costs seem wrong:
1. Check your ElevenLabs plan details
2. Calculate your actual rate per 1K characters
3. Update `ELEVENLABS_COST_PER_1K_CHARS` in .env

### Missing Cost Data
If cost breakdown doesn't appear:
1. Check browser console for errors
2. Ensure you're using a supported chat endpoint
3. Verify the backend is returning cost_breakdown in response

### Zero Costs
Some services may show $0.0000:
- Embedding costs are very low and may round to zero
- This is normal for small token counts