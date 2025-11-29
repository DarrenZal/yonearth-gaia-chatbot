# MemoRAG Server Deployment Guide

This guide explains how to run the MemoRAG experiment on the production server (152.53.37.180).

## Prerequisites

- SSH access to the server: `ssh claudeuser@152.53.37.180`
- The repository already exists at: `/home/claudeuser/yonearth-gaia-chatbot/`

## Deployment Steps

### 1. Push Experiment Branch from Local

On your local machine:

```bash
# Commit all changes
git add experiments/memorag
git commit -m "Add MemoRAG experiment setup for Our Biggest Deal"

# Push branch to remote
git push origin experiment/memorag-integration
```

### 2. SSH to Server

```bash
ssh claudeuser@152.53.37.180
cd /home/claudeuser/yonearth-gaia-chatbot
```

### 3. Pull Experiment Branch

```bash
# Fetch latest changes
git fetch origin

# Checkout experiment branch
git checkout experiment/memorag-integration
git pull origin experiment/memorag-integration
```

### 4. Install Dependencies

```bash
# Install MemoRAG and dependencies
pip install -r experiments/memorag/requirements.txt

# Or install individually
pip install memorag pdfplumber torch transformers

# Verify installation
python3 -c "from memorag import MemoRAG; print('✅ MemoRAG installed')"
```

### 5. Build Memory Index

```bash
# Build the index (this will take 10-30 minutes depending on hardware)
python3 experiments/memorag/scripts/build_memory.py

# Monitor progress
# The script will show:
# - Text extraction progress
# - Chapter detection
# - Memory building status
# - Final index size
```

**Expected Output:**
```
============================================================
🚀 MemoRAG Memory Builder - Our Biggest Deal
============================================================
📖 Extracting text from: /home/claudeuser/yonearth-gaia-chatbot/data/books/OurBiggestDeal/our_biggest_deal.pdf
   ✅ Extracted XXX,XXX characters from XXX pages
   ✅ Split into 7 chapters
🧠 Initializing MemoRAG with model: memorag-qwen2-7b-inst
   ✅ Pipeline initialized
📝 Memorizing book content (XXX,XXX characters)...
   This may take several minutes depending on hardware...
   ✅ Memory index built successfully!
💾 Saving pipeline to: experiments/memorag/indices/memorag_pipeline.pkl
   ✅ Pipeline saved!
   Size: XX.XX MB
```

### 6. Test with Sample Query

```bash
# Test query about Planetary Prosperity
python3 experiments/memorag/scripts/query_memory.py \
  "Trace the narrative arc of how the author describes 'Planetary Prosperity' across the chapters."
```

### 7. Interactive Mode

For exploratory queries:

```bash
python3 experiments/memorag/scripts/query_memory.py --interactive
```

Commands in interactive mode:
- `/context <n>` - Set context length
- `/topk <n>` - Set number of retrieved passages
- `quit` or `exit` - Exit interactive mode

## Performance Tuning

### GPU Acceleration

If the server has GPU:

```bash
# Check GPU availability
nvidia-smi

# MemoRAG will automatically use GPU if available
```

### Memory Constraints

If running out of memory:

```bash
# Use smaller model
python3 experiments/memorag/scripts/build_memory.py --model memorag-qwen2-1.5b-inst

# Or adjust batch size in the script
```

### Cache Directory

By default, models are cached in `experiments/memorag/indices/model_cache/`. To use a different location:

```bash
python3 experiments/memorag/scripts/build_memory.py --cache-dir /path/to/cache
```

## Comparison Testing

### Compare with ACE Pipeline

To compare MemoRAG with the existing ACE knowledge graph:

1. **ACE Query** (existing system):
   ```bash
   # Query the GraphRAG endpoint
   curl -X POST http://localhost:8000/api/graphrag/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What is Planetary Prosperity?"}'
   ```

2. **MemoRAG Query** (experimental):
   ```bash
   python3 experiments/memorag/scripts/query_memory.py \
     "What is Planetary Prosperity?"
   ```

3. **Compare Results**:
   - Response quality
   - Citation accuracy
   - Context relevance
   - Speed

### Metrics to Track

Create a comparison log in `experiments/memorag/results/`:

```bash
mkdir -p experiments/memorag/results

# Log queries and responses
echo "Query: What is Planetary Prosperity?" >> experiments/memorag/results/comparison.log
python3 experiments/memorag/scripts/query_memory.py \
  "What is Planetary Prosperity?" >> experiments/memorag/results/comparison.log 2>&1
```

## Troubleshooting

### Error: "Pipeline not found"

```bash
# Make sure you built the index first
python3 experiments/memorag/scripts/build_memory.py
```

### Error: "CUDA out of memory"

```bash
# Use smaller model or CPU mode
export CUDA_VISIBLE_DEVICES=""  # Force CPU
python3 experiments/memorag/scripts/build_memory.py
```

### Error: "ModuleNotFoundError: memorag"

```bash
# Reinstall dependencies
pip install --upgrade memorag transformers torch
```

## Maintenance

### Rebuilding Index

If the book is updated:

```bash
# Remove old index
rm experiments/memorag/indices/memorag_pipeline.pkl

# Rebuild
python3 experiments/memorag/scripts/build_memory.py
```

### Disk Space

The index and models require significant disk space:
- Model cache: ~4-8 GB
- Memory index: ~100-500 MB

Check available space:

```bash
df -h
du -sh experiments/memorag/indices/
```

## Next Steps

After successful deployment:

1. Run systematic comparison tests
2. Document response quality differences
3. Consider hybrid approach (ACE + MemoRAG)
4. Evaluate for production deployment

## Support

For issues, check:
- MemoRAG documentation: https://github.com/qhjqhj00/MemoRAG
- PyTorch installation: https://pytorch.org/get-started/locally/
- Server logs: Check `experiments/memorag/results/`
