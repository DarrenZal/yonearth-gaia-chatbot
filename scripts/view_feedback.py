#!/usr/bin/env python3
"""
Script to view feedback data saved by the chatbot
"""
import json
import os
from datetime import datetime
from typing import List, Dict

def load_feedback_files(feedback_dir: str = "data/feedback") -> List[Dict]:
    """Load all feedback files and combine them"""
    all_feedback = []
    
    if not os.path.exists(feedback_dir):
        print(f"No feedback directory found at {feedback_dir}")
        return all_feedback
    
    # Get all JSON files in the feedback directory
    feedback_files = sorted([f for f in os.listdir(feedback_dir) if f.endswith('.json')])
    
    for filename in feedback_files:
        filepath = os.path.join(feedback_dir, filename)
        try:
            with open(filepath, 'r') as f:
                feedback_list = json.load(f)
                all_feedback.extend(feedback_list)
                print(f"Loaded {len(feedback_list)} feedback entries from {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return all_feedback

def print_feedback_summary(feedback_list: List[Dict]):
    """Print summary statistics of feedback"""
    if not feedback_list:
        print("\nNo feedback found.")
        return
    
    print(f"\n=== FEEDBACK SUMMARY ===")
    print(f"Total feedback entries: {len(feedback_list)}")
    
    # Count by type
    type_counts = {}
    for fb in feedback_list:
        fb_type = fb.get('type', 'unknown')
        type_counts[fb_type] = type_counts.get(fb_type, 0) + 1
    
    print("\nFeedback by type:")
    for fb_type, count in type_counts.items():
        print(f"  {fb_type}: {count}")
    
    # Average relevance rating
    ratings = [fb.get('relevanceRating') for fb in feedback_list if fb.get('relevanceRating')]
    if ratings:
        avg_rating = sum(ratings) / len(ratings)
        print(f"\nAverage relevance rating: {avg_rating:.2f}/5 ({len(ratings)} ratings)")
    
    # Episodes correct percentage
    episodes_correct = [fb.get('episodesCorrect') for fb in feedback_list if fb.get('episodesCorrect') is not None]
    if episodes_correct:
        correct_pct = sum(episodes_correct) / len(episodes_correct) * 100
        print(f"Episodes/books correct: {correct_pct:.1f}% ({len(episodes_correct)} responses)")
    
    # RAG type distribution
    rag_counts = {}
    for fb in feedback_list:
        rag_type = fb.get('ragType', 'unknown')
        rag_counts[rag_type] = rag_counts.get(rag_type, 0) + 1
    
    print("\nFeedback by RAG type:")
    for rag_type, count in rag_counts.items():
        print(f"  {rag_type}: {count}")

def print_detailed_feedback(feedback_list: List[Dict], max_entries: int = 10):
    """Print detailed feedback entries"""
    detailed = [fb for fb in feedback_list if fb.get('detailedFeedback')]
    
    if not detailed:
        print("\nNo detailed feedback found.")
        return
    
    print(f"\n=== DETAILED FEEDBACK (showing {min(max_entries, len(detailed))} of {len(detailed)}) ===")
    
    for i, fb in enumerate(detailed[:max_entries]):
        print(f"\n--- Entry {i+1} ---")
        print(f"Date: {fb.get('timestamp', 'Unknown')}")
        print(f"Type: {fb.get('type', 'Unknown')}")
        print(f"Query: {fb.get('query', 'No query')[:100]}...")
        print(f"Relevance: {fb.get('relevanceRating', 'Not rated')}/5")
        print(f"Episodes Correct: {fb.get('episodesCorrect', 'Not specified')}")
        print(f"RAG Type: {fb.get('ragType', 'Unknown')}")
        print(f"Feedback: {fb.get('detailedFeedback', 'No feedback')}")

def main():
    """Main function"""
    print("YonEarth Gaia Chatbot - Feedback Viewer")
    print("="*40)
    
    feedback_list = load_feedback_files()
    
    if feedback_list:
        print_feedback_summary(feedback_list)
        print_detailed_feedback(feedback_list)
        
        # Save combined feedback for analysis
        output_file = f"data/feedback/combined_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(feedback_list, f, indent=2)
        print(f"\nCombined feedback saved to: {output_file}")
    else:
        print("\nNo feedback data found.")

if __name__ == "__main__":
    main()