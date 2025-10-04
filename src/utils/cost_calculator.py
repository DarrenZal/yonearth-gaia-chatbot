"""
Cost calculator for API usage tracking

Note: Pricing information sources:
- OpenAI: https://openai.com/api/pricing/ (as of August 2024)
- ElevenLabs: Based on user's plan ($11/month for 200K characters, then $0.15 per 1K additional)

These are estimates and actual costs may vary based on:
- Your specific pricing tier
- Volume discounts
- API changes
- Regional pricing differences
"""
import os
import tiktoken
from typing import Dict, Optional, Tuple

# Pricing as of August 2024 (per 1K tokens for OpenAI, per character for ElevenLabs)
OPENAI_PRICING = {
    "gpt-3.5-turbo": {
        "input": 0.0005,  # $0.50 per 1M input tokens
        "output": 0.0015  # $1.50 per 1M output tokens
    },
    "gpt-4": {
        "input": 0.03,    # $30 per 1M input tokens
        "output": 0.06    # $60 per 1M output tokens
    },
    "gpt-4o": {
        "input": 0.005,   # $5 per 1M input tokens
        "output": 0.015   # $15 per 1M output tokens
    },
    "gpt-4o-mini": {
        "input": 0.00015,  # $0.15 per 1M input tokens
        "output": 0.0006   # $0.60 per 1M output tokens
    },
    "text-embedding-3-small": {
        "input": 0.00002,  # $0.02 per 1M tokens
        "output": 0
    },
    "text-embedding-3-large": {
        "input": 0.00013,  # $0.13 per 1M tokens
        "output": 0
    }
}

# ElevenLabs pricing (per 1K characters)
# Based on user's plan: $11/month for 200,000 characters, then $0.15 per 1K additional
# Marginal cost calculation: $0.15 per 1K characters for usage beyond monthly quota
# Note: This represents the additional cost per request, not the subscription cost
ELEVENLABS_PRICING = {
    "eleven_multilingual_v2": 0.15,  # $0.15 per 1K characters (marginal cost)
    "eleven_monolingual_v1": 0.15    # $0.15 per 1K characters (marginal cost)
}


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens for a given text and model"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except:
        # Fallback to cl100k_base encoding
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))


def calculate_openai_cost(
    model: str,
    input_text: str,
    output_text: str,
    is_embedding: bool = False
) -> Dict[str, float]:
    """Calculate OpenAI API costs"""
    
    # Default to gpt-3.5-turbo pricing if model not found
    pricing = OPENAI_PRICING.get(model, OPENAI_PRICING["gpt-3.5-turbo"])
    
    # Count tokens
    input_tokens = count_tokens(input_text, model)
    output_tokens = count_tokens(output_text, model) if not is_embedding else 0
    
    # Calculate costs (pricing is per 1K tokens)
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total_cost, 6)
    }


def calculate_elevenlabs_cost(
    text: str,
    model: str = "eleven_multilingual_v2"
) -> Dict[str, float]:
    """Calculate ElevenLabs TTS costs"""
    
    # Get pricing from environment or use defaults
    # Allow override via ELEVENLABS_COST_PER_1K_CHARS environment variable
    default_pricing = ELEVENLABS_PRICING.get(model, ELEVENLABS_PRICING["eleven_multilingual_v2"])
    pricing_per_1k = float(os.getenv('ELEVENLABS_COST_PER_1K_CHARS', str(default_pricing)))
    
    # Count characters
    character_count = len(text)
    
    # Calculate cost (pricing is per 1K characters)
    cost = (character_count / 1000) * pricing_per_1k
    
    return {
        "model": model,
        "characters": character_count,
        "cost": round(cost, 6),
        "rate_per_1k": pricing_per_1k
    }


def calculate_total_cost(
    llm_model: str,
    prompt: str,
    response: str,
    embedding_texts: Optional[list] = None,
    voice_text: Optional[str] = None,
    voice_model: str = "eleven_multilingual_v2"
) -> Dict[str, any]:
    """Calculate total cost for a chat interaction"""
    
    costs = {
        "llm": {},
        "embeddings": {},
        "voice": {},
        "total": 0
    }
    
    # LLM cost
    llm_cost = calculate_openai_cost(llm_model, prompt, response)
    costs["llm"] = llm_cost
    costs["total"] += llm_cost["total_cost"]
    
    # Embedding costs (if any)
    if embedding_texts:
        total_embedding_tokens = 0
        for text in embedding_texts:
            tokens = count_tokens(text, "text-embedding-3-small")
            total_embedding_tokens += tokens
        
        embedding_cost = (total_embedding_tokens / 1000) * OPENAI_PRICING["text-embedding-3-small"]["input"]
        costs["embeddings"] = {
            "model": "text-embedding-3-small",
            "tokens": total_embedding_tokens,
            "cost": round(embedding_cost, 6)
        }
        costs["total"] += embedding_cost
    
    # Voice cost (if enabled)
    if voice_text:
        voice_cost = calculate_elevenlabs_cost(voice_text, voice_model)
        costs["voice"] = voice_cost
        costs["total"] += voice_cost["cost"]
    
    costs["total"] = round(costs["total"], 6)
    
    return costs


def format_cost_breakdown(costs: Dict[str, any]) -> Dict[str, any]:
    """Format costs for display"""
    breakdown = {
        "summary": f"${costs['total']:.4f}",
        "details": []
    }
    
    # LLM details
    if costs.get("llm"):
        llm = costs["llm"]
        breakdown["details"].append({
            "service": "OpenAI LLM",
            "model": llm["model"],
            "usage": f"{llm['input_tokens']} in / {llm['output_tokens']} out tokens",
            "cost": f"${llm['total_cost']:.4f}"
        })
    
    # Embedding details
    if costs.get("embeddings") and costs["embeddings"].get("tokens", 0) > 0:
        emb = costs["embeddings"]
        breakdown["details"].append({
            "service": "OpenAI Embeddings",
            "model": emb["model"],
            "usage": f"{emb['tokens']} tokens",
            "cost": f"${emb['cost']:.4f}"
        })
    
    # Voice details
    if costs.get("voice") and costs["voice"].get("characters", 0) > 0:
        voice = costs["voice"]
        breakdown["details"].append({
            "service": "ElevenLabs Voice",
            "model": voice["model"],
            "usage": f"{voice['characters']} characters",
            "cost": f"${voice['cost']:.4f}"
        })
    
    return breakdown