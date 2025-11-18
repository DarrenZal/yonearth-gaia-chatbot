#!/usr/bin/env python3
"""
Locust Performance Testing for YonEarth Gaia Chatbot

This script tests the performance of the chat API with various query types:
- Simple queries (70%): Basic questions with keyword search
- Complex queries (20%): Multi-hop reasoning requiring context
- Graph-heavy queries (10%): Entity-rich queries requiring graph traversal

Success criteria:
- p95 response time < 2.5s
- Error rate < 1%
"""

from locust import HttpUser, task, between, events
import json
import random
import time
from datetime import datetime

# Sample queries for different categories
SIMPLE_QUERIES = [
    "What is regenerative agriculture?",
    "Tell me about composting",
    "What is biochar?",
    "How can I reduce my carbon footprint?",
    "What are soil building parties?",
    "Explain permaculture principles",
    "What is sustainable living?",
    "How do I start a garden?",
    "What is climate change?",
    "Tell me about renewable energy"
]

COMPLEX_QUERIES = [
    "How does regenerative agriculture relate to climate change mitigation, and what role does biochar play in carbon sequestration?",
    "Explain the connection between soil health, human health, and planetary health in the context of the microbiome",
    "What are the economic benefits of transitioning to renewable energy, and how does this relate to community resilience?",
    "Describe the relationship between indigenous wisdom and modern sustainability practices",
    "How do permaculture principles apply to urban environments and food security?"
]

GRAPH_HEAVY_QUERIES = [
    "Who are the main experts on regenerative agriculture mentioned across episodes, and what organizations do they represent?",
    "What are all the different practices and techniques related to soil health discussed in the podcast?",
    "List the key concepts that connect climate change, agriculture, and human health",
    "What books and resources are recommended for learning about permaculture and sustainable living?",
    "Which episodes feature discussions about both renewable energy and community development?"
]

class ChatUser(HttpUser):
    """Simulated user for chat API testing"""

    # Wait between 1 and 3 seconds between requests
    wait_time = between(1, 3)

    def on_start(self):
        """Initialize user session"""
        self.session_id = f"perf_test_{int(time.time())}_{random.randint(1000, 9999)}"
        self.message_count = 0
        self.search_method = random.choice(["original", "bm25", "both"])
        self.personality = random.choice(["warm_mother", "wise_guide", "earth_activist"])

    @task(70)
    def simple_query(self):
        """Test with simple queries (70% of traffic)"""
        query = random.choice(SIMPLE_QUERIES)
        self.send_chat_message(query, "simple")

    @task(20)
    def complex_query(self):
        """Test with complex multi-hop queries (20% of traffic)"""
        query = random.choice(COMPLEX_QUERIES)
        self.send_chat_message(query, "complex")

    @task(10)
    def graph_heavy_query(self):
        """Test with graph-intensive queries (10% of traffic)"""
        query = random.choice(GRAPH_HEAVY_QUERIES)
        self.send_chat_message(query, "graph_heavy")

    def send_chat_message(self, message, query_type):
        """Send a chat message to the API"""

        self.message_count += 1

        # Prepare request payload
        payload = {
            "message": message,
            "session_id": self.session_id,
            "k": 5,  # Number of context chunks to retrieve
            "search_method": self.search_method,
            "personality": self.personality,
            "enable_graph": True,  # Enable knowledge graph retrieval
            "message_number": self.message_count
        }

        # Choose endpoint based on search method
        if self.search_method == "bm25":
            endpoint = "/api/bm25/chat"
        elif self.search_method == "both":
            endpoint = "/api/compare"
        else:
            endpoint = "/api/chat"

        # Send request with custom name for metrics
        with self.client.post(
            endpoint,
            json=payload,
            name=f"{endpoint}_{query_type}",
            catch_response=True
        ) as response:
            try:
                # Check response status
                if response.status_code != 200:
                    response.failure(f"Got status code {response.status_code}")
                    return

                # Parse response
                data = response.json()

                # Validate response structure
                if self.search_method == "both":
                    # Compare endpoint returns both results
                    if "original" not in data or "bm25" not in data:
                        response.failure("Missing comparison results")
                        return
                else:
                    # Single method endpoints
                    if "response" not in data:
                        response.failure("Missing response field")
                        return

                    # Check for valid response content
                    response_text = data.get("response", "")
                    if len(response_text) < 50:
                        response.failure("Response too short")
                        return

                    # Check for references
                    if "references" not in data:
                        response.failure("Missing references")
                        return

                # Check response time
                if response.elapsed.total_seconds() > 2.5:
                    response.failure(f"Response took {response.elapsed.total_seconds():.2f}s (> 2.5s)")
                else:
                    response.success()

            except json.JSONDecodeError:
                response.failure("Invalid JSON response")
            except Exception as e:
                response.failure(f"Unexpected error: {str(e)}")

    @task(5)
    def health_check(self):
        """Perform health check"""
        with self.client.get("/health", name="/health_check", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure("Unhealthy status")
            else:
                response.failure(f"Health check failed with status {response.status_code}")

    @task(3)
    def get_recommendations(self):
        """Test recommendations endpoint"""

        # Generate some topics based on previous "conversations"
        topics = random.sample([
            "regenerative agriculture",
            "composting",
            "biochar",
            "climate change",
            "soil health",
            "permaculture"
        ], k=random.randint(1, 3))

        payload = {
            "topics": topics,
            "episode_ids": [random.randint(0, 172) for _ in range(random.randint(1, 3))]
        }

        with self.client.post(
            "/api/recommendations",
            json=payload,
            name="/api/recommendations",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "episodes" in data and len(data["episodes"]) > 0:
                    response.success()
                else:
                    response.failure("No recommendations returned")
            else:
                response.failure(f"Recommendations failed with status {response.status_code}")


# Event handlers for custom statistics
@events.init_command_line_parser.add_listener
def add_custom_arguments(parser):
    """Add custom command line arguments"""
    parser.add_argument(
        '--target-rps',
        type=int,
        default=10,
        help='Target requests per second'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Test duration in seconds'
    )

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Log test start"""
    print("=" * 60)
    print("Starting YonEarth Gaia Chatbot Performance Test")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Host: {environment.host}")
    print("=" * 60)

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Log test completion and summary"""
    print("\n" + "=" * 60)
    print("Performance Test Summary")
    print("=" * 60)

    stats = environment.stats

    # Calculate key metrics
    total_requests = stats.total.num_requests
    total_failures = stats.total.num_failures
    error_rate = (total_failures / total_requests * 100) if total_requests > 0 else 0

    print(f"Total requests: {total_requests}")
    print(f"Total failures: {total_failures}")
    print(f"Error rate: {error_rate:.2f}%")
    print(f"Median response time: {stats.total.median_response_time}ms")
    print(f"95th percentile: {stats.total.get_response_time_percentile(0.95)}ms")
    print(f"99th percentile: {stats.total.get_response_time_percentile(0.99)}ms")

    # Check success criteria
    p95 = stats.total.get_response_time_percentile(0.95)
    success = True

    if p95 > 2500:  # 2.5 seconds in milliseconds
        print(f"\n❌ FAILED: p95 response time ({p95}ms) exceeds 2500ms threshold")
        success = False
    else:
        print(f"\n✅ PASSED: p95 response time ({p95}ms) within threshold")

    if error_rate > 1:
        print(f"❌ FAILED: Error rate ({error_rate:.2f}%) exceeds 1% threshold")
        success = False
    else:
        print(f"✅ PASSED: Error rate ({error_rate:.2f}%) within threshold")

    if success:
        print("\n🎉 All performance criteria met!")
    else:
        print("\n⚠️ Performance criteria not met - optimization needed")

    print("=" * 60)


# Custom request statistics
class CustomRequestStats:
    """Track custom metrics for requests"""

    def __init__(self):
        self.graph_queries_count = 0
        self.graph_queries_time = 0
        self.simple_queries_count = 0
        self.simple_queries_time = 0
        self.complex_queries_count = 0
        self.complex_queries_time = 0

    def log_request(self, query_type, response_time):
        """Log request metrics by type"""
        if query_type == "graph_heavy":
            self.graph_queries_count += 1
            self.graph_queries_time += response_time
        elif query_type == "simple":
            self.simple_queries_count += 1
            self.simple_queries_time += response_time
        elif query_type == "complex":
            self.complex_queries_count += 1
            self.complex_queries_time += response_time

    def get_average_times(self):
        """Calculate average response times by query type"""
        results = {}

        if self.simple_queries_count > 0:
            results["simple"] = self.simple_queries_time / self.simple_queries_count

        if self.complex_queries_count > 0:
            results["complex"] = self.complex_queries_time / self.complex_queries_count

        if self.graph_queries_count > 0:
            results["graph_heavy"] = self.graph_queries_time / self.graph_queries_count

        return results


# Initialize custom stats
custom_stats = CustomRequestStats()


if __name__ == "__main__":
    print("""
    YonEarth Gaia Chatbot Performance Testing
    ==========================================

    To run the performance test:

    1. Install Locust:
       pip install locust

    2. Start the test with web UI:
       locust -f tests/locustfile.py --host http://localhost:8000

       Then open http://localhost:8089 in your browser

    3. Or run headless test:
       locust -f tests/locustfile.py --host http://localhost:8000 \\
              --headless --users 10 --spawn-rate 2 --run-time 60s

    4. For production testing:
       locust -f tests/locustfile.py --host http://152.53.194.214 \\
              --headless --users 50 --spawn-rate 5 --run-time 300s

    Success Criteria:
    - p95 response time < 2.5 seconds
    - Error rate < 1%
    """)