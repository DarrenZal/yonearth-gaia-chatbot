"""
Hybrid retriever combining keyword-based and semantic search
"""
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from langchain_core.documents import Document
import re

from .keyword_indexer import KeywordIndexer
from .vectorstore import YonEarthVectorStore
from ..config import settings

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever combining keyword frequency and semantic search"""
    
    def __init__(
        self, 
        vectorstore: YonEarthVectorStore,
        keyword_indexer: Optional[KeywordIndexer] = None,
        keyword_weight: float = 0.4,
        semantic_weight: float = 0.6
    ):
        self.vectorstore = vectorstore
        self.keyword_indexer = keyword_indexer or KeywordIndexer()
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        
        # Load or build keyword index
        if not self.keyword_indexer.load_index():
            logger.info("Building keyword index...")
            self.keyword_indexer.build_index_from_episodes_dir()
    
    def _extract_keywords_from_query(self, query: str) -> List[str]:
        """Extract important keywords from user query"""
        # Simple keyword extraction - could be enhanced with NER, etc.
        keywords = []
        
        # Remove common question words and extract meaningful terms
        query_clean = re.sub(r'\b(what|how|when|where|why|is|are|can|do|does|tell|me|about)\b', '', query.lower())
        
        # Split and filter
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query_clean)
        
        # Remove very common words that aren't in our index's stopwords
        common_remove = {'podcast', 'episode', 'guest', 'talk', 'discuss', 'conversation'}
        keywords = [w for w in words if w not in common_remove]
        
        # If no meaningful keywords found, use original query words
        if not keywords:
            keywords = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
            
        return keywords[:5]  # Limit to top 5 keywords
    
    def _keyword_search(
        self, 
        keywords: List[str], 
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Perform keyword-based search"""
        return self.keyword_indexer.search_by_keywords(
            keywords=keywords,
            top_k=top_k,
            min_frequency=1
        )
    
    def _semantic_search(
        self, 
        query: str, 
        top_k: int = 20,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Perform semantic search"""
        return self.vectorstore.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=filter_dict
        )
    
    def _load_episode_transcript(self, episode_id: str) -> Optional[str]:
        """Load full transcript for an episode"""
        try:
            import json
            from pathlib import Path
            
            # Construct episode file path
            episode_file = settings.episodes_dir / f"{episode_id}.json"
            
            if not episode_file.exists():
                return None
                
            with open(episode_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            transcript = data.get('full_transcript', '')
            if transcript == 'NO_TRANSCRIPT_AVAILABLE':
                return None
                
            return transcript
        except Exception as e:
            logger.error(f"Error loading transcript for {episode_id}: {e}")
            return None
    
    def _create_document_from_episode(
        self, 
        episode_result: Dict[str, Any], 
        keywords: List[str],
        context_window: int = 500
    ) -> Optional[Document]:
        """Create a document from episode search result with relevant context"""
        episode_id = episode_result['episode_id']
        transcript = self._load_episode_transcript(episode_id)
        
        if not transcript:
            return None
            
        # Find the best context window around keywords
        best_context = self._find_keyword_context(transcript, keywords, context_window)
        
        # Create document with metadata
        metadata = {
            'episode_id': episode_id,
            'episode_number': episode_result['episode_number'],
            'title': episode_result['title'],
            'url': episode_result['url'],
            'keyword_score': episode_result['score'],
            'keyword_frequencies': episode_result['keyword_frequencies'],
            'matching_keywords': episode_result['matching_keywords'],
            'source': 'keyword_search'
        }
        
        return Document(
            page_content=best_context,
            metadata=metadata
        )
    
    def _find_keyword_context(
        self, 
        transcript: str, 
        keywords: List[str], 
        context_window: int = 500
    ) -> str:
        """Find the best context window around keywords in transcript"""
        if not keywords:
            return transcript[:context_window]
            
        # Process keywords for matching
        processed_keywords = []
        for keyword in keywords:
            # Add both stemmed and original versions
            processed = self.keyword_indexer._preprocess_text(keyword)
            processed_keywords.extend(processed)
            processed_keywords.append(keyword.lower())
            
        # Find keyword positions in transcript
        positions = []
        transcript_lower = transcript.lower()
        
        for keyword in processed_keywords:
            # Find all occurrences
            start = 0
            while True:
                pos = transcript_lower.find(keyword, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1
                
        if not positions:
            # No keywords found, return beginning
            return transcript[:context_window]
            
        # Find the position with the highest keyword density
        best_start = 0
        best_score = 0
        
        # Try different starting positions around keyword locations
        for pos in positions:
            start = max(0, pos - context_window // 2)
            end = min(len(transcript), start + context_window)
            window = transcript[start:end]
            
            # Score this window by keyword frequency
            window_lower = window.lower()
            score = sum(window_lower.count(kw) for kw in processed_keywords)
            
            if score > best_score:
                best_score = score
                best_start = start
                
        # Extract the best context window
        end = min(len(transcript), best_start + context_window)
        context = transcript[best_start:end]
        
        # Clean up context boundaries (try to end at sentence boundaries)
        if not context.endswith('.'):
            last_period = context.rfind('.')
            if last_period > len(context) * 0.8:  # Only if we don't lose too much text
                context = context[:last_period + 1]
                
        return context
    
    def _merge_and_rank_results(
        self,
        keyword_results: List[Dict[str, Any]],
        semantic_results: List[Tuple[Document, float]],
        top_k: int = 10
    ) -> List[Tuple[Document, float]]:
        """Merge keyword and semantic results using weighted scoring"""
        merged_results = {}
        
        # Add keyword results
        for idx, result in enumerate(keyword_results):
            episode_id = result['episode_id']
            
            # Create document from episode
            doc = self._create_document_from_episode(result, [])
            if doc:
                # Score based on keyword ranking (higher is better)
                keyword_score = result['score']
                # Normalize keyword score (inverse rank + frequency score)
                normalized_keyword_score = keyword_score + (len(keyword_results) - idx) / len(keyword_results)
                
                merged_results[episode_id] = {
                    'document': doc,
                    'keyword_score': normalized_keyword_score,
                    'semantic_score': 0.0
                }
        
        # Add semantic results
        for doc, score in semantic_results:
            metadata = getattr(doc, 'metadata', {})
            episode_id = metadata.get('episode_id') or metadata.get('episode_number', 'unknown')
            
            # Ensure episode_id is a string
            episode_id = str(episode_id)
            
            if episode_id in merged_results:
                # Update existing result
                merged_results[episode_id]['semantic_score'] = float(score)
            else:
                # Add new result
                merged_results[episode_id] = {
                    'document': doc,
                    'keyword_score': 0.0,
                    'semantic_score': float(score)
                }
        
        # Calculate final weighted scores
        final_results = []
        for episode_id, data in merged_results.items():
            keyword_score = data['keyword_score']
            semantic_score = data['semantic_score']
            
            # Weighted combination
            final_score = (
                self.keyword_weight * keyword_score + 
                self.semantic_weight * semantic_score
            )
            
            # Update document metadata with scores
            doc = data['document']
            if hasattr(doc, 'metadata'):
                doc.metadata.update({
                    'final_score': final_score,
                    'keyword_component': keyword_score,
                    'semantic_component': semantic_score
                })
            
            final_results.append((doc, final_score))
        
        # Sort by final score (higher is better)
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        return final_results[:top_k]
    
    def retrieve(
        self,
        query: str,
        k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Perform hybrid retrieval combining keyword and semantic search"""
        logger.info(f"Performing hybrid search for: {query[:50]}...")
        
        # Extract keywords from query
        keywords = self._extract_keywords_from_query(query)
        logger.info(f"Extracted keywords: {keywords}")
        
        # Perform keyword search
        keyword_results = self._keyword_search(keywords, top_k=k*2)
        logger.info(f"Keyword search found {len(keyword_results)} results")
        
        # Perform semantic search
        semantic_results = self._semantic_search(query, top_k=k*2, filter_dict=filter_dict)
        logger.info(f"Semantic search found {len(semantic_results)} results")
        
        # Merge and rank results
        final_results = self._merge_and_rank_results(keyword_results, semantic_results, top_k=k)
        logger.info(f"Final hybrid results: {len(final_results)} documents")
        
        return final_results
    
    def get_keyword_stats(self) -> Dict[str, Any]:
        """Get keyword indexer statistics"""
        return self.keyword_indexer.get_stats()


def main():
    """Test hybrid retriever functionality"""
    import logging
    from .vectorstore import create_vectorstore
    
    logging.basicConfig(level=logging.INFO)
    
    # Create components
    logger.info("Setting up hybrid retriever...")
    vectorstore = create_vectorstore()
    retriever = HybridRetriever(vectorstore)
    
    # Test biochar query
    query = "what is biochar"
    results = retriever.retrieve(query, k=5)
    
    print(f"\nHybrid search results for '{query}':")
    print(f"Found {len(results)} results\n")
    
    for i, (doc, score) in enumerate(results, 1):
        metadata = getattr(doc, 'metadata', {})
        print(f"{i}. Episode {metadata.get('episode_number', 'Unknown')}: {metadata.get('title', 'Unknown Title')}")
        print(f"   Final Score: {score:.4f}")
        print(f"   Keyword Score: {metadata.get('keyword_component', 0):.4f}")
        print(f"   Semantic Score: {metadata.get('semantic_component', 0):.4f}")
        print(f"   Matching Keywords: {metadata.get('matching_keywords', [])}")
        print(f"   Content Preview: {doc.page_content[:100]}...")
        print()
    
    # Show keyword index stats
    stats = retriever.get_keyword_stats()
    print(f"\nKeyword Index Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()