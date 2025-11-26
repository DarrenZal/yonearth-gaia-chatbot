# GraphRAG Production Safeguards

**Last Updated**: November 22, 2025

## Overview

The GraphRAG generation script (`scripts/build_proper_graphrag.py`) has been hardened with production-grade safeguards to prevent data loss and ensure reliable long-running execution.

## Implemented Safeguards

### 1. ✅ Run Under a Keeper

**Problem**: SSH disconnects kill the process
**Solutions Provided**:

#### Option A: nohup Wrapper Script (Recommended)
```bash
export OPENAI_API_KEY=your-key-here
./scripts/run_graphrag_safe.sh
```

Features:
- Survives SSH disconnects via `nohup`
- Auto-generates timestamped log files
- Displays PID and monitoring commands
- Validates environment variables before start

#### Option B: systemd Service
```bash
# Copy service file
sudo cp scripts/graphrag-builder.service /etc/systemd/system/

# Edit to add your OpenAI API key
sudo systemctl edit graphrag-builder.service

# Start service
sudo systemctl start graphrag-builder

# Monitor
sudo journalctl -u graphrag-builder -f
```

Features:
- Full systemd integration
- Automatic restart on failure (optional)
- Resource limits (22GB memory, 200% CPU)
- Persistent logging

### 2. ✅ Retry + Backoff

**Problem**: Transient API errors waste hours of work
**Solution**: Enhanced retry decorator with max retries

```python
@retry_with_backoff(max_retries=5, initial_delay=1.0)
def generate_community_summary(...):
    # Handles 429 (rate limit) and 5xx (server errors)
    # Exponential backoff: 1s → 2s → 4s → 8s → 16s
    # Logs each attempt with detailed error messages
```

Features:
- **5 retries** before giving up (configurable via `MAX_LLM_RETRIES`)
- **Exponential backoff**: 1s, 2s, 4s, 8s, 16s delays
- **Detailed logging**: Shows attempt number and error details
- **Failed community tracking**: Logs failed communities to `checkpoints/failed_communities.json` for later retry
- **Fallback summaries**: Uses simple placeholder if all retries exhausted

### 3. ✅ Checkpoint Cadence with fsync

**Problem**: Crashes lose all progress
**Solution**: Frequent checkpoints with forced disk writes

```python
# Save every 50 communities
if total_processed % SUMMARY_CHECKPOINT_INTERVAL == 0:
    with summary_checkpoint.open('w') as f:
        json.dump(all_summaries, f, indent=2)
        f.flush()
        os.fsync(f.fileno())  # Force write to disk
    print(f"  💾 Checkpoint saved ({total_processed} communities)")
```

Features:
- **Checkpoint every 50 communities** (configurable)
- **fsync()** ensures data hits disk before continuing
- **Progressive saves**: Each level's summaries saved incrementally
- **Final fsync**: Ensures complete data persistence at end

### 4. ✅ Per-Level Resume Flags

**Problem**: Mid-level crashes require full restart
**Solution**: Track completed IDs per community

The checkpoint system now stores:
```json
{
  "2": {
    "cluster_0_0": {"title": "...", "summary": "..."},
    "cluster_0_1": {"title": "...", "summary": "..."}
  },
  "1": {...},
  "0": {...}
}
```

On resume:
- **Skips already-completed communities** (checks for `'title'` and `'summary'` keys)
- **Continues from last checkpoint** within each level
- **Preserves progress** across restarts

### 5. ✅ Progress Logging (Heartbeat)

**Problem**: Long-running processes appear stalled
**Solution**: Background thread logs progress every 5 minutes

```python
def heartbeat_logger():
    """Logs every 300 seconds with current status"""
```

Output example:
```
⏱️  HEARTBEAT [14:35:22]
   Elapsed: 2.3h | Level: 0 | Community: cluster_0_423
   Total processed: 3,892 | Memory: 4.23 GB
```

Features:
- **5-minute intervals** (configurable via `HEARTBEAT_INTERVAL`)
- **Non-blocking**: Runs in background daemon thread
- **Detailed state**: Shows level, community ID, total count, memory
- **Automatic cleanup**: Stops on script exit

### 6. ✅ Memory Guardrail

**Problem**: OOM killer terminates process without warning
**Solution**: Periodic memory checks with configurable limit

```python
def check_memory():
    """Check every 100 communities, bail if >20GB"""
    mem_gb = psutil.Process().memory_info().rss / (1024 ** 3)
    if mem_gb > MEMORY_LIMIT_GB:
        print(f"❌ MEMORY LIMIT EXCEEDED: {mem_gb:.2f} GB")
        sys.exit(1)
```

Features:
- **Checks every 100 communities** (configurable via `MEMORY_CHECK_INTERVAL`)
- **20GB default limit** (configurable via `MEMORY_LIMIT_GB`)
- **Graceful shutdown**: Stops heartbeat and releases lockfile before exit
- **Resumable**: Restart picks up from last checkpoint

### 7. ✅ Single-Run Lock

**Problem**: Concurrent runs corrupt checkpoints
**Solution**: PID-based lockfile with stale detection

```python
def acquire_lockfile():
    """Creates lockfile with current PID"""
    # Checks if existing process is still running
    # Removes stale locks automatically
```

Features:
- **PID verification**: Checks if locked process still exists
- **Stale lock removal**: Auto-cleans up dead processes
- **Clear error messages**: Shows PID and lockfile path if blocked
- **Automatic cleanup**: Released in `finally` block

Location: `/home/claudeuser/yonearth-gaia-chatbot/data/graphrag_hierarchy/graphrag.lock`

### 8. ✅ Validation on Resume

**Problem**: Script runs without required checkpoints, wastes time
**Solution**: Pre-flight validation before starting

```python
def validate_checkpoints():
    """Verify embeddings and Leiden checkpoints exist"""
```

Validates:
- ✅ `checkpoints/embeddings.npy` (102 MB expected)
- ✅ `checkpoints/leiden_hierarchies.json` (1.9 MB expected)

If missing:
```
❌ CHECKPOINT VALIDATION FAILED:
   Missing embeddings checkpoint: /path/to/embeddings.npy

Run the full pipeline first to generate checkpoints.
```

## Usage Examples

### First-Time Run (Generate All Checkpoints)

```bash
# This will fail with validation error since checkpoints don't exist yet
# Remove validation for first run:
export OPENAI_API_KEY=your-key-here
export AUTO_CONFIRM_LLM=yes

# Run WITHOUT validation (comment out in main() or use flag)
python3 scripts/build_proper_graphrag.py
```

**Note**: For first run, you need to comment out `validate_checkpoints()` in the `main()` function temporarily, or add a `--skip-validation` flag.

### Resume from Checkpoint

```bash
# Just run the script - it will:
# 1. Acquire lockfile
# 2. Validate checkpoints exist
# 3. Load embeddings from checkpoint
# 4. Load Leiden hierarchies from checkpoint
# 5. Resume summarization from last saved state

export OPENAI_API_KEY=your-key-here
./scripts/run_graphrag_safe.sh
```

### Monitor Progress

```bash
# Watch the log in real-time
tail -f logs/graphrag_build_20251122_*.log

# Check memory usage
ps aux | grep build_proper_graphrag

# View heartbeat logs
grep "HEARTBEAT" logs/graphrag_build_*.log | tail -5

# Check checkpoint status
ls -lh data/graphrag_hierarchy/checkpoints/
```

### Handle Failed Communities

If some communities fail after max retries:

```bash
# Check failed communities log
cat data/graphrag_hierarchy/checkpoints/failed_communities.json

# Example output:
[
  {
    "level": 0,
    "id": "cluster_0_423",
    "error": "Rate limit exceeded after 5 retries"
  }
]
```

Failed communities get fallback summaries like:
- Title: `"Cluster cluster_0_423"`
- Summary: `"A community of 47 entities."`

You can manually retry these later or regenerate with adjusted retry settings.

## Configuration

All safeguards are configurable via constants at the top of `build_proper_graphrag.py`:

```python
# Summary parameters
LLM_RATE_DELAY = 0.05  # seconds between requests
SUMMARY_CHECKPOINT_INTERVAL = 50  # Save every N communities
MAX_LLM_RETRIES = 5  # Max retries before skipping

# Production safeguards
HEARTBEAT_INTERVAL = 300  # Progress log every 5 minutes
MEMORY_CHECK_INTERVAL = 100  # Log memory every N communities
MEMORY_LIMIT_GB = 20  # Bail if memory exceeds this
```

## Performance Impact

**Overhead from safeguards**:
- **Checkpointing**: ~0.1s every 50 communities (negligible)
- **Heartbeat logging**: No measurable impact (background thread)
- **Memory checks**: <0.01s every 100 communities
- **Lockfile**: <0.1s total (only at start/end)
- **fsync**: <0.1s per checkpoint (ensures durability)

**Total overhead**: <1% of runtime

## Recovery Scenarios

### Scenario 1: SSH Disconnect During Level 2 (1,000/2,129 complete)

**What happens**:
1. Process killed if not using nohup/systemd ❌
2. If using keeper: Process continues ✅

**Recovery**:
```bash
# Check if still running
ps aux | grep build_proper_graphrag

# If stopped, restart (resumes from last checkpoint)
./scripts/run_graphrag_safe.sh
```

**Result**: Resumes at community 1,000, skips completed summaries

### Scenario 2: Out of Memory at Community 5,000

**What happens**:
1. Memory check detects >20GB usage
2. Script logs error and exits gracefully
3. Heartbeat stopped, lockfile released
4. Checkpoint saved up to community 5,000

**Recovery**:
```bash
# Increase memory limit or restart on lower-load system
# Edit MEMORY_LIMIT_GB in script if needed
./scripts/run_graphrag_safe.sh
```

**Result**: Resumes at community 5,000

### Scenario 3: API Rate Limit Errors

**What happens**:
1. Retry logic kicks in (5 attempts)
2. Exponential backoff: 1s, 2s, 4s, 8s, 16s
3. If all retries fail: Logs to `failed_communities.json`
4. Uses fallback summary and continues

**Recovery**: No action needed - script handles automatically

### Scenario 4: Concurrent Run Attempt

**What happens**:
```
❌ ERROR: Another instance is running (PID 12345)
   Lockfile: /path/to/graphrag.lock
   If this is stale, remove: rm /path/to/graphrag.lock
```

**Recovery**:
```bash
# Check if process is really running
ps -fp 12345

# If dead, remove stale lock
rm data/graphrag_hierarchy/graphrag.lock

# Restart
./scripts/run_graphrag_safe.sh
```

## Checkpoint File Structure

```
data/graphrag_hierarchy/checkpoints/
├── embeddings.npy                # 102 MB - entity embeddings
├── leiden_hierarchies.json       # 1.9 MB - community structure
├── summaries_progress.json       # Growing - partial summaries
└── failed_communities.json       # Optional - retry log
```

## Future Improvements

Potential enhancements (not yet implemented):

1. **Batch API for 50% savings**: Use OpenAI Batch API (24hr turnaround)
2. **GPT-5 nano**: Switch when available (50-67% cost reduction)
3. **Prometheus metrics**: Export progress/memory to monitoring
4. **Slack/email alerts**: Notify on completion or errors
5. **Automatic retry of failed communities**: Background job to retry logged failures
6. **Rate limit detection**: Preemptively slow down before hitting limits

## Troubleshooting

### "Missing embeddings checkpoint" Error

**Cause**: First-time run or checkpoints were deleted
**Solution**: Comment out `validate_checkpoints()` for initial run

### Process Dies Without Logs

**Cause**: OOM killer terminated it
**Solution**: Check dmesg for OOM messages, reduce `MEMORY_LIMIT_GB`

### Stale Lockfile Won't Clear

**Cause**: Manual kill left lockfile
**Solution**: `rm data/graphrag_hierarchy/graphrag.lock`

### Progress Seems Stuck

**Cause**: API slowness or long summary generation
**Solution**: Check heartbeat logs - script is likely still progressing

## Cost Analysis

**With all safeguards enabled**:
- Estimated cost: $1.34 for 6,398 communities (unchanged)
- Runtime: 4-5 hours (10-15% faster due to reduced delay)
- Checkpoint overhead: <$0.01 (negligible)

**Recovery cost savings**:
- Without safeguards: $1.34 × restart attempts
- With safeguards: $0.00 (resumes from checkpoint)

**ROI**: Pays for itself after 1 crash/restart event

## Support

For issues or questions:
- Check logs: `logs/graphrag_build_*.log`
- Review checkpoints: `data/graphrag_hierarchy/checkpoints/`
- Inspect lockfile: `data/graphrag_hierarchy/graphrag.lock`
- Check failed communities: `checkpoints/failed_communities.json`
