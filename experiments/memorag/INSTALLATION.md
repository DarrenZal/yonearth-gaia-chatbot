# MemoRAG Installation Guide

This guide covers installation for two distinct deployment scenarios:

| Scenario | Hardware | Purpose |
|----------|----------|---------|
| **Serving** | CPU-only Linux server | Load and query pre-built memory indices |
| **Building** | GPU server (24GB+ VRAM) | Create memory indices from documents |

## Prerequisites

- Python 3.10+ (tested with 3.12)
- At least 16GB RAM for serving, 24GB+ GPU VRAM for building
- Linux recommended (macOS has limited support)

---

## Quick Start (CPU Serving)

For most users who just need to **query** pre-built indices:

```bash
# 1. Create virtual environment
cd /path/to/yonearth-gaia-chatbot
python3 -m venv venv-memorag
source venv-memorag/bin/activate

# 2. Install CPU-only PyTorch (MUST be first)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Install FAISS for CPU
pip install faiss-cpu

# 4. Install remaining dependencies
pip install -r experiments/memorag/requirements.txt

# 5. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import faiss; print(f'FAISS version: {faiss.__version__}')"
python -c "from memorag import MemoRAG; print('MemoRAG: OK')"
```

---

## Detailed Installation Paths

### Path A: CPU-Only Serving Environment

Use this for production servers that only need to **load and query** pre-built memory indices.

#### Step 1: Create Virtual Environment

```bash
cd /path/to/yonearth-gaia-chatbot
python3 -m venv venv-memorag
source venv-memorag/bin/activate
```

> **Why a separate venv?** MemoRAG has specific PyTorch/FAISS requirements that may conflict with the main chatbot dependencies.

#### Step 2: Install CPU-Only PyTorch

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

> **Important:** Install PyTorch BEFORE other dependencies to ensure the CPU version is used.

#### Step 3: Install FAISS-CPU

```bash
pip install faiss-cpu
```

#### Step 4: Install Remaining Dependencies

```bash
pip install -r experiments/memorag/requirements.txt
```

#### Step 5: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import faiss; print(f'FAISS: {faiss.__version__}')"
python -c "from memorag import MemoRAG; print('MemoRAG: OK')"
```

Expected output:
```
PyTorch: 2.x.x+cpu, CUDA: False
FAISS: 1.x.x
MemoRAG: OK
```

#### CPU Serving Notes

- Memory indices (`memory.bin`, `index.bin`) must be built on a GPU machine first
- Loading indices on CPU is slower but works correctly
- `faiss-cpu` can load indices built with `faiss-gpu`

---

### Path B: GPU Building Environment

Use this for machines with NVIDIA GPUs that will **create** memory indices.

#### Step 1: Check CUDA Version

```bash
nvidia-smi
# Look for "CUDA Version: X.X" in the output
```

#### Step 2: Create Virtual Environment

```bash
cd /path/to/yonearth-gaia-chatbot
python3 -m venv venv-memorag-gpu
source venv-memorag-gpu/bin/activate
```

#### Step 3: Install GPU-Enabled PyTorch

Choose the command matching your CUDA version:

**CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 12.4:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### Step 4: Install FAISS-GPU

```bash
pip install faiss-gpu
```

#### Step 5: Install Remaining Dependencies

```bash
pip install -r experiments/memorag/requirements.txt
```

#### Step 6: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
python -c "import faiss; print(f'FAISS GPU support: {faiss.get_num_gpus()}')"
python -c "from memorag import MemoRAG; print('MemoRAG: OK')"
```

Expected output:
```
PyTorch: 2.x.x+cu121, CUDA: True, Device: NVIDIA RTX 4090
FAISS GPU support: 1
MemoRAG: OK
```

---

### Path C: macOS (Limited Support)

MemoRAG has dependencies (particularly `deepspeed`) that are challenging on macOS. Use API mode instead:

```bash
python3 -m venv venv-memorag
source venv-memorag/bin/activate

# Install without GPU dependencies
pip install torch torchvision torchaudio
pip install faiss-cpu
pip install -r experiments/memorag/requirements.txt
```

**Note:** You'll need to use `--no-openai` flag disabled and provide an `OPENAI_API_KEY` for generation, as local model inference is limited on macOS.

---

### Path D: Docker (Cross-Platform)

```bash
cd experiments/memorag
docker build -t memorag-experiment .
docker run -v $(pwd)/data:/app/data -v $(pwd)/indices:/app/indices memorag-experiment
```

---

## Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI API (for hybrid generation)
OPENAI_API_KEY=sk-your-key-here

# Hugging Face (for gated models, optional)
HF_TOKEN=hf_your-token-here

# Model cache directory (optional)
TRANSFORMERS_CACHE=/path/to/cache
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'faiss'"

Install either `faiss-cpu` or `faiss-gpu`:
```bash
pip install faiss-cpu  # For CPU-only
# OR
pip install faiss-gpu  # For GPU
```

### "ImportError: cannot import name 'StandardGpuResources' from 'faiss'"

You have `faiss-cpu` installed but code is trying to use GPU features. Either:
1. Install `faiss-gpu` instead, OR
2. Ensure MemoRAG falls back to CPU mode

### Both faiss-cpu and faiss-gpu installed

This causes conflicts. Remove both and reinstall one:
```bash
pip uninstall faiss-cpu faiss-gpu -y
pip install faiss-cpu  # OR faiss-gpu
```

### CUDA out of memory

MemoRAG requires substantial GPU memory (24GB+ recommended). Options:
1. Use `load_in_4bit=True` for quantized models
2. Reduce batch size in memorization
3. Use a smaller model: `--model memorag-qwen2-1.5b-inst`

### Flash Attention errors

If you see errors about flash attention, the scripts already disable it:
```python
MemoRAG(..., enable_flash_attn=False)
```

### DeepSpeed errors on macOS

DeepSpeed is designed for Linux/CUDA. On macOS:
1. Use API mode with OpenAI for generation
2. Or run on a Linux server/VM

---

## Usage

### Build Memory Index (GPU required)

```bash
source venv-memorag-gpu/bin/activate
python experiments/memorag/scripts/build_memory.py --pdf /path/to/book.pdf
```

### Query the Memory (CPU or GPU)

```bash
source venv-memorag/bin/activate
python experiments/memorag/scripts/query_memory.py "What is the main thesis of the book?"
```

---

## Version Compatibility

| Component | CPU Version | GPU Version |
|-----------|-------------|-------------|
| Python | 3.10+ | 3.10+ |
| PyTorch | 2.0+ (cpu) | 2.0+ (cu118/cu121/cu124) |
| FAISS | faiss-cpu | faiss-gpu |
| MemoRAG | 0.1.5+ | 0.1.5+ |
| CUDA | N/A | 11.8+ |
