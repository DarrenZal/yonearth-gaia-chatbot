#!/usr/bin/env python3
"""
Simple HTTP server to serve web interface and provide working API
"""
import sys
import os
import json
import logging
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/root/yonearth-gaia-chatbot/.env')

sys.path.append('/root/yonearth-gaia-chatbot')

from src.rag.bm25_chain import BM25RAGChain
from src.voice.elevenlabs_client import ElevenLabsVoiceClient
from src.config import settings
from src.utils.cost_calculator import calculate_elevenlabs_cost, format_cost_breakdown

class GaiaHTTPHandler(SimpleHTTPRequestHandler):
    # Class-level initialization
    bm25_chain = None
    openai_client = None
    voice_client = None
    _initialized = False
    
    @classmethod
    def initialize_services(cls):
        """Initialize all services once"""
        if cls._initialized:
            return
            
        print("Initializing services...")
        
        # Initialize BM25 chain
        print("Initializing BM25 chain...")
        cls.bm25_chain = BM25RAGChain()
        cls.bm25_chain.initialize()
        print("BM25 chain ready!")
        
        # Initialize OpenAI client
        cls.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        print("OpenAI client ready!")
        
        # Initialize voice client
        # Temporarily hardcode the API key since env loading is having issues
        elevenlabs_key = os.getenv('ELEVENLABS_API_KEY') or 'sk_878861eb5bf57da30b761fdf70e6438d9cc80e59938ac71c'
        print(f"[INIT] ElevenLabs API Key: {elevenlabs_key[:10] + '...' if elevenlabs_key else 'None'}")
        
        if elevenlabs_key:
            try:
                cls.voice_client = ElevenLabsVoiceClient(
                    api_key=elevenlabs_key,
                    voice_id=os.getenv('ELEVENLABS_VOICE_ID', 'YcVr5DmTjJ2cEVwNiuhU'),
                    model_id=os.getenv('ELEVENLABS_MODEL_ID', 'eleven_multilingual_v2')
                )
                print("Voice client initialized successfully!")
            except Exception as e:
                print(f"Voice client initialization failed: {e}")
                import traceback
                traceback.print_exc()
                cls.voice_client = None
        else:
            print("No ElevenLabs API key found")
            cls.voice_client = None
            
        cls._initialized = True
    
    def __init__(self, *args, **kwargs):
        # Initialize services on first request
        self.__class__.initialize_services()
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path == '/':
            self.path = '/web/index.html'
        elif self.path.startswith('/web/'):
            pass  # Keep the path as is
        elif self.path == '/health' or self.path == '/api/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy"}).encode())
            return
        elif self.path == '/api/voice/test':
            # Test voice endpoint
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            test_result = {
                "voice_client_exists": self.__class__.voice_client is not None,
                "elevenlabs_key_set": bool(os.getenv('ELEVENLABS_API_KEY')),
                "voice_id": os.getenv('ELEVENLABS_VOICE_ID', 'Not set')
            }
            
            if self.__class__.voice_client:
                try:
                    audio = self.__class__.voice_client.generate_speech_base64("Test")
                    test_result["test_generation"] = "Success" if audio else "Failed"
                    test_result["audio_length"] = len(audio) if audio else 0
                except Exception as e:
                    test_result["test_generation"] = f"Error: {str(e)}"
            
            self.wfile.write(json.dumps(test_result).encode())
            return
        else:
            self.path = '/web' + self.path
        
        super().do_GET()
    
    def do_POST(self):
        if self.path in ['/bm25/chat', '/chat', '/api/chat', '/api/bm25/chat']:
            self.handle_chat()
        elif self.path in ['/chat/compare', '/api/chat/compare']:
            self.handle_model_comparison()
        elif self.path in ['/conversation-recommendations', '/api/conversation-recommendations']:
            self.handle_conversation_recommendations()
        elif self.path in ['/feedback', '/api/feedback']:
            self.handle_feedback()
        else:
            self.send_error(404)
    
    def handle_chat(self):
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            # Extract parameters
            message = request_data.get('message', '')
            search_method = request_data.get('search_method', 'semantic')
            max_citations = request_data.get('max_citations', request_data.get('max_references', 3))
            k = request_data.get('k', max(10, max_citations * 2))  # Retrieve more docs to ensure enough unique episodes
            model = request_data.get('model')  # New: model selection
            # Handle both personality and gaia_personality
            personality = request_data.get('personality') or request_data.get('gaia_personality', 'warm_mother')
            custom_prompt = request_data.get('custom_prompt')
            category_threshold = request_data.get('category_threshold', 0.7)
            
            print(f"Chat request: {message[:50]}... (method: {search_method}, model: {model}, max_citations: {max_citations})")
            
            # Use BM25 chain and pass model parameter through kwargs
            result = self.__class__.bm25_chain.chat(
                message=message,
                search_method=search_method,
                k=k,
                include_sources=True,
                custom_prompt=custom_prompt,
                model_name=model,
                personality_variant=personality,
                max_citations=max_citations,
                category_threshold=category_threshold
            )
            
            # Format sources for web interface
            sources = []
            for source in result.get('sources', []):
                # Create source in format web interface expects
                formatted_source = {
                    'episode_id': source.get('episode_id', 'Unknown'),
                    'episode_number': source.get('episode_number', 'Unknown'),
                    'title': source.get('title', 'Unknown Title'),
                    'guest_name': source.get('guest_name', 'Unknown Guest'),
                    'url': source.get('url', ''),
                    'content_preview': source.get('content_preview', ''),
                    'content_type': source.get('content_type', 'episode')
                }
                
                # For book content, the bm25_chain already provides correct formatting
                # Just pass through the correctly formatted values
                if source.get('content_type') == 'book':
                    # bm25_chain.py already formats these correctly:
                    # - title: f"Chapter {chapter_int}" 
                    # - guest_name: author (without "Author:" prefix)
                    # - episode_number: f"Book: {book_title}"
                    # So we don't need to override them
                    pass
                
                sources.append(formatted_source)
            
            # Generate voice if requested and available
            audio_data = None
            voice_cost = None
            enable_voice = request_data.get('enable_voice', False)
            print(f"Voice requested: {enable_voice}, Voice client available: {self.__class__.voice_client is not None}")
            
            if enable_voice and self.__class__.voice_client:
                try:
                    response_text = result.get('response', '')
                    print(f"Generating voice for text length: {len(response_text)}")
                    processed_text = self.__class__.voice_client.preprocess_text_for_speech(response_text)
                    audio_data = self.__class__.voice_client.generate_speech_base64(processed_text)
                    print(f"Generated voice audio: {len(audio_data) if audio_data else 0} chars")
                    
                    # Calculate voice cost
                    if audio_data:
                        voice_cost = calculate_elevenlabs_cost(
                            processed_text, 
                            model=os.getenv('ELEVENLABS_MODEL_ID', 'eleven_multilingual_v2')
                        )
                except Exception as e:
                    print(f"Voice generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue without voice
            else:
                if enable_voice:
                    print("Voice requested but client not available")
            
            # Get cost breakdown from result (or create empty one if not present)
            cost_breakdown = result.get('cost_breakdown', {'summary': '$0.0000', 'details': []})
            
            # Add voice cost if applicable
            if voice_cost:
                # Update the cost breakdown with voice cost
                if 'details' in cost_breakdown:
                    cost_breakdown['details'].append({
                        'service': 'ElevenLabs Voice',
                        'model': voice_cost['model'],
                        'usage': f"{voice_cost['characters']} characters",
                        'cost': f"${voice_cost['cost']:.4f}"
                    })
                # Update total
                current_total = float(cost_breakdown['summary'].replace('$', ''))
                new_total = current_total + voice_cost['cost']
                cost_breakdown['summary'] = f"${new_total:.4f}"
            
            # Create response
            response_data = {
                'response': result.get('response', ''),
                'sources': sources,
                'episode_references': result.get('episode_references', []),
                'search_method_used': result.get('search_method_used', search_method),
                'model_used': model or 'gpt-3.5-turbo',  # Include which model was used
                'cost_breakdown': cost_breakdown,
                'success': True
            }
            
            # Add audio data if generated
            if audio_data:
                response_data['audio_data'] = audio_data
                print(f"Audio data added to response: {len(audio_data)} chars")
            else:
                print("No audio data to add to response")
            
            print(f"Returning {len(sources)} sources")
            if sources:
                print(f"First source: {sources[0].get('title', 'NO_TITLE')}")
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode())
            
        except Exception as e:
            print(f"Error handling chat: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error_response = {'error': str(e), 'success': False}
            self.wfile.write(json.dumps(error_response).encode())
    
    def handle_model_comparison(self):
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            # Extract parameters
            message = request_data.get('message', '')
            personality = request_data.get('personality', 'warm_mother')
            custom_prompt = request_data.get('custom_prompt')
            max_references = request_data.get('max_references', 3)
            
            print(f"Model comparison request: {message[:50]}... (max_references: {max_references})")
            
            # Define models to compare
            models = ['gpt-3.5-turbo', 'gpt-4o-mini', 'gpt-4']
            results = {}
            
            # Get response from each model
            import time
            start_time = time.time()
            
            for model in models:
                try:
                    # Create a Gaia instance with the specific model
                    from src.character.gaia import GaiaCharacter
                    gaia_instance = GaiaCharacter(
                        personality_variant=personality,
                        model_name=model
                    )
                    
                    # Get search results using existing BM25 chain
                    search_result = self.bm25_chain._retrieve_documents(
                        query=message,
                        search_method='hybrid',
                        k=max(5, max_references)  # Retrieve at least 5 docs to have enough for max_references
                    )
                    
                    # Generate response with specific model
                    response = gaia_instance.generate_response(
                        user_input=message,
                        retrieved_docs=search_result,  # Already just documents
                        custom_prompt=custom_prompt,
                        max_references=max_references
                    )
                    
                    # Format citations
                    citations = []
                    for doc in search_result:
                        metadata = getattr(doc, 'metadata', {})
                        citation = {
                            'episode_number': metadata.get('episode_number', 'Unknown'),
                            'title': metadata.get('title', 'Unknown Title'),
                            'guest_name': metadata.get('guest_name', 'Unknown Guest'),
                            'url': metadata.get('url', ''),
                            'relevance': 'High'
                        }
                        citations.append(citation)
                    
                    # Format result for this model
                    results[model] = {
                        'response': response.get('response', ''),
                        'personality': personality,
                        'citations': citations,
                        'context_used': len(search_result),
                        'session_id': request_data.get('session_id'),
                        'retrieval_count': len(search_result),
                        'processing_time': 0,
                        'model_used': model
                    }
                    
                except Exception as e:
                    print(f"Error with model {model}: {e}")
                    results[model] = {
                        'response': f"Error with {model}: {str(e)}",
                        'personality': personality,
                        'citations': [],
                        'context_used': 0,
                        'session_id': request_data.get('session_id'),
                        'retrieval_count': 0,
                        'processing_time': 0,
                        'model_used': model
                    }
            
            processing_time = time.time() - start_time
            
            # Create response
            response_data = {
                'comparison': True,
                'modelComparison': True,
                'models': results,
                'processing_time': processing_time
            }
            
            print(f"Model comparison completed in {processing_time:.2f}s")
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode())
            
        except Exception as e:
            print(f"Error handling model comparison: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error_response = {'error': str(e), 'success': False}
            self.wfile.write(json.dumps(error_response).encode())
    
    def handle_conversation_recommendations(self):
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            # Extract parameters
            conversation_history = request_data.get('conversation_history', [])
            max_recommendations = request_data.get('max_recommendations', 4)
            
            print(f"Conversation recommendations request: {len(conversation_history)} messages")
            
            # Use LLM to intelligently analyze conversation and recommend content
            recommended_citations, extracted_topics = self.get_llm_recommendations(
                conversation_history, max_recommendations
            )
            
            # Create response
            response_data = {
                'recommendations': recommended_citations,
                'conversation_topics': extracted_topics[:5],  # Limit topics
                'total_found': len(recommended_citations)  # Use recommended_citations length instead
            }
            
            print(f"Returning {len(recommended_citations)} recommendations with {len(extracted_topics)} topics")
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode())
            
        except Exception as e:
            print(f"Error handling conversation recommendations: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            error_response = {'error': str(e), 'success': False}
            self.wfile.write(json.dumps(error_response).encode())
    
    def get_llm_recommendations(self, conversation_history, max_recommendations):
        """Use LLM to intelligently analyze conversation and recommend relevant content"""
        unique_citations = []  # Initialize early to avoid NameError in exception handler
        
        try:
            # Collect all available citations from the conversation
            all_citations = []
            conversation_summary = []
            
            for message in conversation_history:
                role = message.get('role', '')
                content = message.get('content', '')
                conversation_summary.append(f"{role.title()}: {content}")
                
                if message.get('citations'):
                    all_citations.extend(message['citations'])
            
            # Remove duplicates from citations
            unique_citations = []
            seen_keys = set()
            for citation in all_citations:
                key = f"{citation.get('episode_number', '')}:{citation.get('title', '')}"
                if key not in seen_keys and key != ':':
                    seen_keys.add(key)
                    unique_citations.append(citation)
            
            if not unique_citations:
                return [], []
            
            # Create LLM prompt for intelligent recommendation
            conversation_text = "\n".join(conversation_summary[-6:])  # Last 6 messages for context
            
            # Format available content for LLM to choose from
            content_options = []
            for i, citation in enumerate(unique_citations):
                episode_num = citation.get('episode_number', 'Unknown')
                title = citation.get('title', 'Unknown')
                guest = citation.get('guest_name', 'Unknown')
                
                if episode_num.startswith('Book:'):
                    content_options.append(f"{i+1}. {title} (Book by {guest})")
                else:
                    content_options.append(f"{i+1}. Episode {episode_num}: {title} (with {guest})")
            
            prompt = f"""You are an AI assistant helping users discover relevant content from the YonEarth Community Podcast and related books. 

Analyze this conversation between a user and Gaia (Earth's spirit):

{conversation_text}

Based on the conversation flow and the user's most recent interests, select the {max_recommendations} most relevant pieces of content from the following options:

{chr(10).join(content_options)}

Requirements:
1. Prioritize content that matches the user's MOST RECENT questions and interests
2. If the conversation has evolved to new topics, focus on those newer topics
3. Aim for diversity - mix episodes and books when possible, but prioritize relevance
4. Consider the natural progression of the conversation

Respond with ONLY a JSON object in this format:
{{
    "selected_indices": [1, 3, 5],
    "topics": ["topic1", "topic2", "topic3"],
    "reasoning": "Brief explanation of why these recommendations are most relevant to the current conversation"
}}

The selected_indices should be the numbers (1-based) of the most relevant content pieces."""

            # Call OpenAI API (with error handling for missing API key)
            if not os.getenv('OPENAI_API_KEY'):
                print("No OpenAI API key found, using smart fallback algorithm")
                return self.smart_fallback_recommendations(unique_citations, conversation_history, max_recommendations)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            
            # Parse LLM response
            llm_response = response.choices[0].message.content.strip()
            
            try:
                recommendation_data = json.loads(llm_response)
                selected_indices = recommendation_data.get('selected_indices', [])
                topics = recommendation_data.get('topics', [])
                reasoning = recommendation_data.get('reasoning', '')
                
                print(f"LLM reasoning: {reasoning}")
                
                # Convert indices to actual citations (1-based to 0-based)
                recommended_citations = []
                for idx in selected_indices:
                    if 1 <= idx <= len(unique_citations):
                        recommended_citations.append(unique_citations[idx - 1])
                
                return recommended_citations[:max_recommendations], topics[:5]
                
            except json.JSONDecodeError:
                print(f"Failed to parse LLM response: {llm_response}")
                # Fallback: return most recent citations
                return unique_citations[-max_recommendations:] if unique_citations else [], ["AI analysis failed"]
        
        except Exception as e:
            print(f"Error in LLM recommendations: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: return most recent citations
            all_citations = []
            for message in conversation_history:
                if message.get('citations'):
                    all_citations.extend(message['citations'])
            
            # Simple deduplication and return recent ones
            unique_citations = []
            seen_keys = set()
            for citation in reversed(all_citations):
                key = f"{citation.get('episode_number', '')}:{citation.get('title', '')}"
                if key not in seen_keys and key != ':':
                    seen_keys.add(key)
                    unique_citations.append(citation)
            
            return unique_citations[:max_recommendations], ["fallback"]
    
    def smart_fallback_recommendations(self, unique_citations, conversation_history, max_recommendations):
        """Smart algorithm for recommendations when OpenAI API is not available"""
        if not unique_citations:
            return [], []
        
        # Extract recent topics from conversation
        recent_keywords = set()
        topic_keywords = [
            'soil', 'biochar', 'compost', 'permaculture', 'regenerative', 'agriculture', 
            'sustainability', 'climate', 'water', 'energy', 'community', 'carbon', 
            'farming', 'garden', 'food', 'ecosystem', 'forest', 'organic'
        ]
        
        # Focus on last 3 messages for recency
        recent_messages = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
        for message in recent_messages:
            content = message.get('content', '').lower()
            for keyword in topic_keywords:
                if keyword in content:
                    recent_keywords.add(keyword)
        
        # Score citations based on recency and relevance
        citation_scores = []
        for i, citation in enumerate(unique_citations):
            score = 0
            citation_text = f"{citation.get('title', '')} {citation.get('guest_name', '')}".lower()
            
            # Reverse order bonus (more recent = higher score)
            score += (len(unique_citations) - i) * 2
            
            # Topic relevance bonus
            for keyword in recent_keywords:
                if keyword in citation_text:
                    score += 3
            
            # Prefer episodes over books for diversity
            if not citation.get('episode_number', '').startswith('Book:'):
                score += 1
            
            citation_scores.append((citation, score))
        
        # Sort by score and take top recommendations
        citation_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Ensure diversity: max 2 books
        recommended_citations = []
        book_count = 0
        
        for citation, score in citation_scores:
            is_book = citation.get('episode_number', '').startswith('Book:')
            
            if is_book and book_count >= 2:
                continue
                
            recommended_citations.append(citation)
            if is_book:
                book_count += 1
                
            if len(recommended_citations) >= max_recommendations:
                break
        
        return recommended_citations, list(recent_keywords)[:5]
    
    def handle_feedback(self):
        """Handle feedback submission from web interface"""
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            feedback_data = json.loads(post_data.decode('utf-8'))
            
            print(f"Received feedback: {feedback_data.get('type')} for message {feedback_data.get('messageId')}")
            
            # Save to JSON file
            from datetime import datetime
            
            feedback_dir = "data/feedback"
            os.makedirs(feedback_dir, exist_ok=True)
            
            # Create filename with date
            filename = f"{feedback_dir}/feedback_{datetime.now().strftime('%Y-%m-%d')}.json"
            
            # Read existing feedback or create new list
            feedback_list = []
            if os.path.exists(filename):
                try:
                    with open(filename, 'r') as f:
                        feedback_list = json.load(f)
                except:
                    feedback_list = []
            
            # Append new feedback
            feedback_list.append(feedback_data)
            
            # Save updated feedback
            with open(filename, 'w') as f:
                json.dump(feedback_list, f, indent=2)
            
            print(f"Feedback saved to {filename}")
            
            # Create response
            response_data = {
                'success': True,
                'message': 'Thank you for your feedback! It helps us improve Gaia\'s responses.'
            }
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode())
            
        except Exception as e:
            print(f"Error handling feedback: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            error_response = {'error': str(e), 'success': False}
            self.wfile.write(json.dumps(error_response).encode())
    
    def do_OPTIONS(self):
        # Handle CORS preflight
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

if __name__ == '__main__':
    os.chdir('/root/yonearth-gaia-chatbot')
    port = 80  # Use port 80 for public access
    server = HTTPServer(('0.0.0.0', port), GaiaHTTPHandler)
    print(f"Starting server on port {port}")
    print("Web interface will be available at http://152.53.194.214/")
    server.serve_forever()