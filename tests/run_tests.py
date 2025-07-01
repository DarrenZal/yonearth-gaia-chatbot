#!/usr/bin/env python3
"""
Test runner script for YonEarth Gaia Chatbot
"""
import sys
import subprocess
from pathlib import Path

def run_tests():
    """Run the complete test suite"""
    project_root = Path(__file__).parent.parent
    
    print("🧪 Running YonEarth Gaia Chatbot Test Suite")
    print("=" * 50)
    
    # Basic pytest command
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--color=yes",  # Colored output
        f"{project_root}/tests/",
    ]
    
    # Add coverage if available
    try:
        import coverage
        cmd.extend([
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ])
        print("📊 Coverage reporting enabled")
    except ImportError:
        print("ℹ️  Coverage not available (install: pip install pytest-cov)")
    
    print(f"🏃 Running command: {' '.join(cmd)}")
    print()
    
    # Run tests
    result = subprocess.run(cmd, cwd=project_root)
    
    if result.returncode == 0:
        print("\n✅ All tests passed!")
        
        # Show coverage report location if available
        coverage_file = project_root / "htmlcov" / "index.html"
        if coverage_file.exists():
            print(f"📊 Coverage report: file://{coverage_file}")
    else:
        print("\n❌ Some tests failed!")
        
    return result.returncode

if __name__ == "__main__":
    sys.exit(run_tests())