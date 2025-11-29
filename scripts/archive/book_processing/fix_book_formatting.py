#!/usr/bin/env python3
"""
Script to identify and fix book formatting issues in bm25_chain.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def analyze_book_metadata_fields():
    """Analyze what fields are actually available in book metadata"""
    
    print("ANALYSIS: Book Metadata Fields")
    print("=" * 80)
    
    print("\nFrom Pinecone inspection, book entries contain:")
    print("- author: 'Aaron William Perry'")
    print("- book_title: 'VIRIDITAS: THE GREAT HEALING'") 
    print("- chapter_number: 4.0, 2.0, etc.")
    print("- chapter_title: '| Aaron William Perry' (problematic)")
    print("- content_type: 'book'")
    print("- Other fields: category, description, isbn, page_start, page_end, etc.")
    
    print("\nCurrent bm25_chain.py formatting tries to access:")
    print("- title: NOT FOUND (should use chapter info)")
    print("- guest_name: NOT FOUND (should use 'author')")
    
    print("\nCorrect mapping should be:")
    print("- title → f'Chapter {chapter_number}' (from chapter_number)")
    print("- guest_name → author")
    print("- episode_number → f'Book: {book_title}'")
    
    print("\nBUT there's also a chapter title issue:")
    print("- chapter_title contains '| Aaron William Perry' instead of actual chapter names")
    print("- This suggests a problem in the PDF chapter detection regex")
    
    return True

def suggest_fixes():
    """Suggest the fixes needed"""
    
    print("\n" + "=" * 80)
    print("SUGGESTED FIXES")
    print("=" * 80)
    
    print("\n1. IMMEDIATE FIX (bm25_chain.py):")
    print("   The formatting logic is actually CORRECT!")
    print("   Lines 260-261 already do:")
    print("   'title': f'Chapter {chapter_int}'")
    print("   'guest_name': author")
    print("   ")
    print("   The issue might be that chapter_int is None or invalid")
    
    print("\n2. DEEPER ISSUE (chapter detection):")
    print("   The chapter_title field contains '| Aaron William Perry'")
    print("   This suggests the regex is matching page headers, not actual chapters")
    
    print("\n3. DEBUGGING STEPS:")
    print("   - Check if chapter_number is properly converted to int")
    print("   - Verify chapter detection regex in book_processor.py")
    print("   - Test with a simple chapter title override")
    
    return True

if __name__ == "__main__":
    analyze_book_metadata_fields()
    suggest_fixes()
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Check the actual chapter conversion logic in bm25_chain.py")
    print("2. Test with a simple API call to see the exact response")
    print("3. If needed, fix the chapter detection regex")
    print("4. Consider adding fallback chapter titles")