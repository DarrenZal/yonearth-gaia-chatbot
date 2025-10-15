#!/bin/bash
# Run V12 extraction using V12 prompts with V11.2.2 pipeline

echo "================================================================================"
echo "🚀 V12 KNOWLEDGE GRAPH EXTRACTION"
echo "================================================================================"
echo ""
echo "V12 Improvements:"
echo "  ✅ Enhanced pronoun resolution (possessive pronouns)"
echo "  ✅ V12 extraction prompts (prevent vague entities, pronouns at source)"
echo "  ✅ V12 evaluation prompts (entity quality checks, philosophical detection)"
echo "  ✅ Expanded praise quote detection"
echo "  ✅ V12 predicate normalization (is-X-for patterns, absolute moderation)"
echo "  ✅ Generic is-a filtering"
echo "  ✅ Recommendation filtering (incidental vs core thesis)"
echo ""
echo "Expected Quality: <4.5% error rate (V11.2.2 was 7.86%)"
echo ""

# Create V12 extraction script by copying V11.2.2 and updating prompt paths
V11_SCRIPT="/home/claudeuser/yonearth-gaia-chatbot/scripts/extract_kg_v11_2_2_book.py"
V12_SCRIPT="/home/claudeuser/yonearth-gaia-chatbot/scripts/extract_kg_v12_book.py"

if [ ! -f "$V12_SCRIPT" ]; then
    echo "📝 Creating V12 extraction script..."
    cp "$V11_SCRIPT" "$V12_SCRIPT"
    
    # Update prompts to v12
    sed -i 's/pass1_extraction_v11\.txt/pass1_extraction_v12.txt/g' "$V12_SCRIPT"
    sed -i 's/pass2_evaluation_v11\.txt/pass2_evaluation_v12.txt/g' "$V12_SCRIPT"
    
    # Update version references
    sed -i 's/V11\.2\.2/V12/g' "$V12_SCRIPT"
    sed -i 's/v11\.2\.2/v12/g' "$V12_SCRIPT"
    sed -i 's/v11_2_2/v12/g' "$V12_SCRIPT"
    
    echo "✅ Created extract_kg_v12_book.py"
    echo ""
fi

# Run V12 extraction
echo "================================================================================"
echo "Starting V12 Extraction..."
echo "================================================================================"
echo ""
echo "This will take approximately 30-40 minutes."
echo ""
echo "Progress will be logged to: kg_extraction_book_v12_*.log"
echo ""

cd /home/claudeuser/yonearth-gaia-chatbot
python3 scripts/extract_kg_v12_book.py

EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ V12 EXTRACTION COMPLETE"
else
    echo "❌ V12 EXTRACTION FAILED (exit code: $EXIT_CODE)"
fi
echo "================================================================================"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "💡 NEXT STEPS:"
    echo ""
    echo "1. Run Reflector on V12 to analyze quality:"
    echo "   python3 scripts/run_reflector_generic.py"
    echo ""
    echo "2. Compare quality metrics:"
    echo "   V11.2.2: 7.86% error rate (70 issues, B+ grade)"
    echo "   V12: Target <4.5% error rate (A- grade)"
    echo ""
    echo "3. If V12 achieves <4.5% error rate:"
    echo "   - V12 becomes new baseline"
    echo "   - Can proceed to next ACE cycle (V13)"
    echo ""
fi

exit $EXIT_CODE
