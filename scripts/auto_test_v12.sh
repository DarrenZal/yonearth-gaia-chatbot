#!/bin/bash
# Automated V12 Testing - Tests V12 improvements automatically

echo "================================================================================"
echo "🤖 AUTOMATED V12 TESTING SYSTEM"
echo "================================================================================"
echo ""
echo "This script will:"
echo "1. ✅ Verify all V12 files are in place"
echo "2. 🧪 Test V12 on problematic chunks from V11.2.2 (fast validation)"
echo "3. 📊 Generate comparison report"
echo "4. 💡 Recommend next steps based on results"
echo ""

# Step 1: Verify V12 files
echo "================================================================================"
echo "STEP 1: Verifying V12 Files"
echo "================================================================================"
echo ""

V12_PASS1="/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/prompts/pass1_extraction_v12.txt"
V12_PASS2="/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/prompts/pass2_evaluation_v12.txt"
V12_NORMALIZER="/home/claudeuser/yonearth-gaia-chatbot/src/knowledge_graph/postprocessing/universal/predicate_normalizer.py"
V12_FILTER="/home/claudeuser/yonearth-gaia-chatbot/src/knowledge_graph/postprocessing/universal/generic_isa_filter.py"

ALL_PRESENT=true

if [ -f "$V12_PASS1" ]; then
    echo "✅ pass1_extraction_v12.txt"
else
    echo "❌ pass1_extraction_v12.txt NOT FOUND"
    ALL_PRESENT=false
fi

if [ -f "$V12_PASS2" ]; then
    echo "✅ pass2_evaluation_v12.txt"
else
    echo "❌ pass2_evaluation_v12.txt NOT FOUND"
    ALL_PRESENT=false
fi

if [ -f "$V12_NORMALIZER" ]; then
    if grep -q "V12 NEW" "$V12_NORMALIZER"; then
        echo "✅ predicate_normalizer.py (V12 enhanced)"
    else
        echo "⚠️  predicate_normalizer.py (V12 changes not detected)"
    fi
else
    echo "❌ predicate_normalizer.py NOT FOUND"
    ALL_PRESENT=false
fi

if [ -f "$V12_FILTER" ]; then
    echo "✅ generic_isa_filter.py (new module)"
else
    echo "❌ generic_isa_filter.py NOT FOUND"
    ALL_PRESENT=false
fi

echo ""

if [ "$ALL_PRESENT" = false ]; then
    echo "❌ Some V12 files are missing. Cannot proceed with testing."
    exit 1
fi

# Step 2: Run targeted test
echo "================================================================================"
echo "STEP 2: Testing V12 on Problematic Chunks (Fast Validation)"
echo "================================================================================"
echo ""
echo "This will re-extract the ~70 problematic chunks from V11.2.2 using V12 prompts."
echo "Expected time: 3-5 minutes"
echo ""

python3 /home/claudeuser/yonearth-gaia-chatbot/scripts/test_v12_on_issues.py

TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ Targeted test failed with exit code $TEST_EXIT_CODE"
    exit 1
fi

echo ""

# Step 3: Analyze results
echo "================================================================================"
echo "STEP 3: Analysis & Recommendations"
echo "================================================================================"
echo ""

# Check if test results exist
RESULTS_FILE="/home/claudeuser/yonearth-gaia-chatbot/kg_extraction_playbook/analysis_reports/v12_targeted_test.json"

if [ ! -f "$RESULTS_FILE" ]; then
    echo "⚠️  Results file not found: $RESULTS_FILE"
    echo ""
    echo "💡 RECOMMENDATION:"
    echo "   - Check test script logs for errors"
    echo "   - Verify V11.2.2 reflection analysis exists"
    echo ""
    exit 1
fi

# Parse results
CHUNKS_TESTED=$(python3 -c "
import json
with open('$RESULTS_FILE') as f:
    data = json.load(f)
    print(data['test_metadata']['chunks_tested'])
" 2>/dev/null)

V11_ISSUES=$(python3 -c "
import json
with open('$RESULTS_FILE') as f:
    data = json.load(f)
    print(data['test_metadata']['v11_issues_total'])
" 2>/dev/null)

echo "📊 TARGETED TEST RESULTS:"
echo "   - Problematic chunks from V11.2.2: $V11_ISSUES issues"
echo "   - Unique chunks tested: $CHUNKS_TESTED"
echo ""

if [ "$CHUNKS_TESTED" = "0" ]; then
    echo "⚠️  No chunks were tested. This likely means:"
    echo "   - V11.2.2 reflection analysis has different structure than expected"
    echo "   - Test script needs to be updated for current JSON format"
    echo ""
    echo "💡 RECOMMENDATION:"
    echo "   - Proceed directly to full V12 extraction"
    echo "   - Run: bash scripts/run_v12_extraction.sh"
    echo ""
else
    echo "✅ Targeted test completed successfully!"
    echo ""
    echo "💡 NEXT STEPS:"
    echo ""
    echo "Option A (Recommended): Full V12 Extraction"
    echo "   bash scripts/run_v12_extraction.sh"
    echo ""
    echo "Option B: Review targeted test results first"
    echo "   cat $RESULTS_FILE | python3 -m json.tool | less"
    echo ""
fi

echo "================================================================================"
