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

sys.path.append('/root/yonearth-gaia-chatbot')

from src.rag.bm25_chain import BM25RAGChain

class GaiaHTTPHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Initialize BM25 chain once
        if not hasattr(GaiaHTTPHandler, 'bm25_chain'):
            print("Initializing BM25 chain...")
            GaiaHTTPHandler.bm25_chain = BM25RAGChain()
            GaiaHTTPHandler.bm25_chain.initialize()
            print("BM25 chain ready!")
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
        else:
            self.path = '/web' + self.path
        
        super().do_GET()
    
    def do_POST(self):
        if self.path in ['/bm25/chat', '/chat', '/api/chat', '/api/bm25/chat']:
            self.handle_chat()
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
            k = request_data.get('k', 3)
            
            print(f"Chat request: {message[:50]}... (method: {search_method})")
            
            # Get response from BM25 chain
            result = self.bm25_chain.chat(
                message=message,
                search_method=search_method,
                k=k,
                include_sources=True
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
                
                # For book content, enhance the display
                if source.get('content_type') == 'book':
                    book_title = source.get('book_title', 'Unknown Book')
                    author = source.get('author', 'Unknown Author')
                    chapter_num = source.get('chapter_number', 'Unknown')
                    
                    formatted_source.update({
                        'episode_number': f"Book: {book_title}",
                        'guest_name': f"Author: {author}",
                        'title': source.get('title', f"{book_title} - Chapter {chapter_num}")
                    })
                
                sources.append(formatted_source)
            
            # Create response
            response_data = {
                'response': result.get('response', ''),
                'sources': sources,
                'episode_references': result.get('episode_references', []),
                'search_method_used': result.get('search_method_used', search_method),
                'success': True
            }
            
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