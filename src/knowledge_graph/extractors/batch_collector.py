"""
Batch API Collector for Knowledge Graph Extraction

Collects extraction requests and manages OpenAI Batch API lifecycle.
Handles file rotation when JSONL exceeds 90MB or 5,000 requests.

The OpenAI Batch API has a 100MB file size limit, so we create multiple
files when needed and submit them as separate batch jobs.
"""

import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class BatchRequest:
    """A single request in the batch JSONL file"""
    custom_id: str              # parent_chunk.id for tracking
    method: str = "POST"
    url: str = "/v1/chat/completions"
    body: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchStatus:
    """Status of a batch job"""
    batch_id: str
    status: str  # "validating", "in_progress", "completed", "failed", "expired", "cancelled"
    completed: int = 0
    failed: int = 0
    total: int = 0

    @property
    def is_done(self) -> bool:
        """Check if batch has finished processing (success or failure)"""
        return self.status in ["completed", "failed", "expired", "cancelled"]

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage"""
        if self.total == 0:
            return 0.0
        return (self.completed + self.failed) / self.total * 100


class BatchCollector:
    """
    Collects extraction requests and manages OpenAI Batch API lifecycle.

    Handles file rotation: If JSONL exceeds ~90MB or ~5,000 requests,
    automatically creates new file (batch_part_2.jsonl, etc.)

    Usage:
        collector = BatchCollector(output_dir=Path("data/batch_jobs"))

        # Add requests
        for chunk in parent_chunks:
            collector.add_extraction_request(chunk, system_prompt, user_template)

        # Finalize and submit
        jsonl_files = collector.finalize()
        batch_ids = collector.submit_all_batches()

        # Poll for status
        statuses = collector.poll_all_batches()

        # Download when complete
        results = collector.download_all_results()
    """

    MAX_FILE_SIZE_BYTES = 90 * 1024 * 1024  # 90MB (safety margin under 100MB limit)
    MAX_REQUESTS_PER_FILE = 5000

    # JSON schema for extraction response
    # NOTE: OpenAI structured outputs require ALL properties in 'required' array
    # when additionalProperties: false is set
    EXTRACTION_SCHEMA = {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string"},
                        "description": {"type": "string"},
                        "aliases": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["name", "type", "description", "aliases"],
                    "additionalProperties": False
                }
            },
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "predicate": {"type": "string"},
                        "target": {"type": "string"}
                    },
                    "required": ["source", "predicate", "target"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["entities", "relationships"],
        "additionalProperties": False
    }

    def __init__(
        self,
        output_dir: Path,
        model: Optional[str] = None,
        extraction_prompt: Optional[str] = None
    ):
        """
        Initialize the batch collector.

        Args:
            output_dir: Directory to store JSONL files and state
            model: OpenAI model to use (default from GRAPH_EXTRACTION_MODEL env)
            extraction_prompt: Custom extraction prompt (optional)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = model or os.getenv("GRAPH_EXTRACTION_MODEL", "gpt-5.1")
        self.extraction_prompt = extraction_prompt

        self.requests: List[BatchRequest] = []
        self.current_file_size = 0
        self.file_parts: List[Path] = []
        self.batch_ids: List[str] = []

        # Lazy load OpenAI client
        self._client = None

    @property
    def client(self):
        """Lazy load OpenAI client"""
        if self._client is None:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable required")
            self._client = OpenAI(api_key=api_key)
        return self._client

    def add_extraction_request(
        self,
        parent_chunk,  # ParentChunk from chunking module
        system_prompt: str,
        user_prompt_template: str
    ) -> None:
        """
        Add a parent chunk to the batch queue.

        Args:
            parent_chunk: ParentChunk object with id and content
            system_prompt: System message for the LLM
            user_prompt_template: User prompt with {text} placeholder
        """
        # Format user prompt with chunk content
        user_content = user_prompt_template.format(text=parent_chunk.content)

        request = BatchRequest(
            custom_id=parent_chunk.id,
            body={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction_response",
                        "strict": True,
                        "schema": self.EXTRACTION_SCHEMA
                    }
                },
                "temperature": 0.1
            }
        )

        # Estimate size of this request
        request_json = json.dumps(asdict(request))
        request_size = len(request_json.encode('utf-8'))

        # Check if we need to rotate files
        if (self.current_file_size + request_size > self.MAX_FILE_SIZE_BYTES or
                len(self.requests) >= self.MAX_REQUESTS_PER_FILE):
            self._rotate_file()

        self.requests.append(request)
        self.current_file_size += request_size

    def _rotate_file(self) -> None:
        """Close current file and start a new one"""
        if self.requests:
            part_num = len(self.file_parts) + 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"extraction_batch_part_{part_num}_{timestamp}.jsonl"
            path = self._write_jsonl_internal(filename)
            self.file_parts.append(path)
            self.requests = []
            self.current_file_size = 0

    def _write_jsonl_internal(self, filename: str) -> Path:
        """Write current requests to JSONL file"""
        path = self.output_dir / filename
        with open(path, 'w', encoding='utf-8') as f:
            for req in self.requests:
                f.write(json.dumps(asdict(req)) + '\n')
        return path

    def finalize(self) -> List[Path]:
        """
        Finalize all pending requests and return list of JSONL files.

        Call this after adding all requests to flush any remaining
        requests to a file.

        Returns:
            List of paths to JSONL files ready for submission
        """
        if self.requests:
            self._rotate_file()
        return self.file_parts

    def get_request_count(self) -> int:
        """Get total number of requests across all files"""
        count = len(self.requests)  # Pending in current buffer

        # Count requests in already-written files
        for path in self.file_parts:
            with open(path, 'r') as f:
                count += sum(1 for _ in f)

        return count

    def submit_batch(self, jsonl_path: Path) -> str:
        """
        Upload file and create batch job.

        Args:
            jsonl_path: Path to JSONL file to submit

        Returns:
            Batch ID for tracking
        """
        # Upload file
        with open(jsonl_path, 'rb') as f:
            file_response = self.client.files.create(file=f, purpose="batch")

        # Create batch job
        batch_response = self.client.batches.create(
            input_file_id=file_response.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "source": "yonearth_kg_extraction",
                "file": jsonl_path.name
            }
        )

        return batch_response.id

    def submit_all_batches(self) -> List[str]:
        """
        Submit all JSONL files as batch jobs.

        Returns:
            List of batch IDs
        """
        batch_ids = []
        for jsonl_path in self.file_parts:
            batch_id = self.submit_batch(jsonl_path)
            batch_ids.append(batch_id)
            print(f"Submitted batch: {batch_id} from {jsonl_path.name}")

        self.batch_ids = batch_ids
        return batch_ids

    def poll_batch(self, batch_id: str) -> BatchStatus:
        """
        Check status of a batch job.

        Args:
            batch_id: ID of batch to check

        Returns:
            BatchStatus with current state
        """
        batch = self.client.batches.retrieve(batch_id)

        completed = 0
        failed = 0
        total = 0

        if batch.request_counts:
            completed = batch.request_counts.completed or 0
            failed = batch.request_counts.failed or 0
            total = batch.request_counts.total or 0

        return BatchStatus(
            batch_id=batch.id,
            status=batch.status,
            completed=completed,
            failed=failed,
            total=total
        )

    def poll_all_batches(self) -> Dict[str, BatchStatus]:
        """
        Check status of all submitted batches.

        Returns:
            Dict mapping batch_id to BatchStatus
        """
        return {bid: self.poll_batch(bid) for bid in self.batch_ids}

    def download_results(self, batch_id: str) -> List[Dict]:
        """
        Download completed batch results.

        Args:
            batch_id: ID of completed batch

        Returns:
            List of result dictionaries from the batch

        Raises:
            ValueError: If batch not completed or no output file
        """
        batch = self.client.batches.retrieve(batch_id)

        if batch.status != "completed":
            raise ValueError(f"Batch {batch_id} not completed. Status: {batch.status}")

        if not batch.output_file_id:
            raise ValueError(f"Batch {batch_id} has no output file")

        # Download output file
        content = self.client.files.content(batch.output_file_id)

        results = []
        for line in content.text.split('\n'):
            if line.strip():
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line: {line[:100]}...")

        return results

    def download_all_results(self) -> Dict[str, List[Dict]]:
        """
        Download results from all completed batches.

        Returns:
            Dict mapping batch_id to list of results
        """
        all_results = {}

        for batch_id in self.batch_ids:
            status = self.poll_batch(batch_id)
            if status.status == "completed":
                try:
                    all_results[batch_id] = self.download_results(batch_id)
                except Exception as e:
                    print(f"Error downloading batch {batch_id}: {e}")
            else:
                print(f"Skipping batch {batch_id} - status: {status.status}")

        return all_results

    def save_state(self, state_path: Optional[Path] = None) -> Path:
        """
        Save collector state for resuming later.

        Args:
            state_path: Path to save state (default: output_dir/batch_state.json)

        Returns:
            Path to saved state file
        """
        if state_path is None:
            state_path = self.output_dir / "batch_state.json"

        state = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "batch_ids": self.batch_ids,
            "file_parts": [str(p) for p in self.file_parts],
            "total_requests": self.get_request_count()
        }

        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

        return state_path

    @classmethod
    def load_state(cls, state_path: Path) -> 'BatchCollector':
        """
        Load collector from saved state.

        Args:
            state_path: Path to saved state file

        Returns:
            BatchCollector with restored state
        """
        with open(state_path) as f:
            state = json.load(f)

        output_dir = state_path.parent
        collector = cls(
            output_dir=output_dir,
            model=state.get("model")
        )

        collector.batch_ids = state.get("batch_ids", [])
        collector.file_parts = [Path(p) for p in state.get("file_parts", [])]

        return collector
