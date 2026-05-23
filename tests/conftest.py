"""
MedQA-Hindi Test Configuration
"""
import sys
from pathlib import Path

# Add project root to path for both local and CI environments
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))