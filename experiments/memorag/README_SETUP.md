# MemoRAG Experiment Setup

This document provides detailed installation instructions for the MemoRAG experiment. The system has two distinct deployment configurations:

- **Serving (CPU-only)**: For running inference on CPU-only Linux servers
- **Building (GPU)**: For creating memory indices on GPU-equipped machines

## Prerequisites

- Python 3.10+ (tested with 3.12)
- Virtual environment (recommended)
- At least 16GB RAM for serving, 24GB+ GPU VRAM for building

## Directory Structure

```
experiments/memorag/
├── README_SETUP.md          # This file
├── requirements-memorag.txt # Python dependencies
├── indices/                 # Generated memory indices (gitignored)
│   └── .gitkeep
└── scripts/                 # MemoRAG scripts (to be added)
```

---

## Path A: Serving Environment (CPU-Only)

Use this path for production servers that only need to **load and query** pre-built memory indices.

### Step 1: Create Virtual Environment

```bash
cd /home/gaia/yonearth-gaia-chatbot
python3 -m venv venv-memorag-cpu
source venv-memorag-cpu/bin/activate
```

### Step 2: Install CPU-Only PyTorch

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Install FAISS-CPU

```bash
pip install faiss-cpu
```

### Step 4: Install Remaining Dependencies

```bash
pip install -r experiments/memorag/requirements-memorag.txt
```

### Step 5: Verify Installation

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

### CPU Serving Notes

- The memory indices (`memory.bin`, `index.bin`) must be built on a GPU machine first
- Loading indices on CPU is slower but works correctly
- For inference, `faiss-cpu` can load indices built with `faiss-gpu`

---

## Path B: Building Environment (GPU)

Use this path for machines with NVIDIA GPUs that will **create** memory indices.

### Step 1: Check CUDA Version

```bash
nvidia-smi
# Look for "CUDA Version: X.X" in the output
```

### Step 2: Create Virtual Environment

```bash
cd /home/gaia/yonearth-gaia-chatbot
python3 -m venv venv-memorag-gpu
source venv-memorag-gpu/bin/activate
```

### Step 3: Install GPU-Enabled PyTorch

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

### Step 4: Install FAISS-GPU

```bash
pip install faiss-gpu
```

### Step 5: Install Remaining Dependencies

```bash
pip install -r experiments/memorag/requirements-memorag.txt
```

### Step 6: Verify Installation

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

## Troubleshooting

### "ModuleNotFoundError: No module named 'faiss'"

You need to install either `faiss-cpu` or `faiss-gpu`:
```bash
pip install faiss-cpu  # For CPU-only
# OR
pip install faiss-gpu  # For GPU
```

### "ImportError: cannot import name 'StandardGpuResources' from 'faiss'"

You have `faiss-cpu` installed but the code is trying to use GPU features. Either:
1. Install `faiss-gpu` instead, OR
2. Ensure the MemoRAG code falls back to CPU mode

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
3. Use a smaller model

### Flash Attention errors

If you see errors about flash attention, disable it:
```python
MemoRAG(..., enable_flash_attn=False)
```

---

## Memory Index Management

### Building Indices (GPU)

```python
from memorag import MemoRAG

# Initialize MemoRAG
rag = MemoRAG(
    mem_model_name_or_path="TommyChien/memorag-qwen2-7b-inst",
    ret_model_name_or_path="BAAI/bge-m3",
    load_in_4bit=True
)

# Memorize your context
context = "Your long document text here..."
rag.memorize(
    context,
    save_dir="experiments/memorag/indices/my_index",
    print_stats=True
)
```

### Loading Indices (CPU or GPU)

```python
from memorag import MemoRAG

rag = MemoRAG(
    mem_model_name_or_path="TommyChien/memorag-qwen2-7b-inst",
    ret_model_name_or_path="BAAI/bge-m3",
    load_in_4bit=True
)

# Load pre-built index
rag.load("experiments/memorag/indices/my_index")

# Query
answer = rag(query="What is the main topic?", task_type="memorag")
print(answer)
```

---

## Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI API (for custom generation)
OPENAI_API_KEY=sk-your-key-here

# Hugging Face (for gated models)
HF_TOKEN=hf_your-token-here

# Cache directory (optional)
TRANSFORMERS_CACHE=/path/to/cache
```

---

## Version Compatibility Matrix

| Component | CPU Version | GPU Version |
|-----------|-------------|-------------|
| PyTorch | 2.0+ (cpu) | 2.0+ (cu118/cu121/cu124) |
| FAISS | faiss-cpu | faiss-gpu |
| MemoRAG | 0.1.0+ | 0.1.0+ |
| Python | 3.10+ | 3.10+ |
| CUDA | N/A | 11.8+ |
