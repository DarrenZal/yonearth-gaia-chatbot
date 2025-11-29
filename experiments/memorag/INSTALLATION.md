# MemoRAG Installation Guide

## ⚠️ Platform Considerations

MemoRAG has dependencies (particularly `deepspeed`) that are challenging to install on macOS. The recommended approach is to run this experiment on a Linux server with GPU support.

## Installation Options

### Option 1: Linux/Server Installation (Recommended)

On a Linux machine with GPU:

```bash
# Install CUDA and PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install MemoRAG and dependencies
pip install memorag pdfplumber

# Or install from requirements
pip install -r experiments/memorag/requirements.txt
```

### Option 2: macOS Installation (Limited)

On macOS, you can use MemoRAG in API mode (no local model):

```bash
# Install without deepspeed
pip install pdfplumber transformers accelerate

# Then modify build_memory.py to use --api flag
```

**Note**: This requires an API key and won't run the model locally.

### Option 3: Docker (Cross-Platform)

Use the provided Dockerfile:

```bash
cd experiments/memorag
docker build -t memorag-experiment .
docker run -v $(pwd)/data:/app/data -v $(pwd)/indices:/app/indices memorag-experiment
```

## Troubleshooting

### DeepSpeed Installation Errors on macOS

If you see errors like `ModuleNotFoundError: No module named 'cpuinfo'`, this is because DeepSpeed is designed for Linux/CUDA environments. Solutions:

1. **Run on Linux server**: The recommended approach
2. **Use API mode**: Set `--api` flag to use cloud-hosted models
3. **Use alternative**: Try a simpler RAG implementation with FAISS + sentence-transformers

### GPU Memory Issues

MemoRAG models can be large (7B+ parameters). If you encounter OOM errors:

```bash
# Use smaller model variant
python build_memory.py --model memorag-qwen2-1.5b-inst

# Or increase context chunking
python build_memory.py --chunk-size 1000
```

## Verification

Test your installation:

```bash
python -c "from memorag import MemoRAG; print('✅ MemoRAG installed successfully')"
```

## Next Steps

Once installed, proceed to:
1. Run `build_memory.py` to create the index
2. Use `query_memory.py` to test queries
