"""Global pytest fixtures and configuration."""

import sys
from pathlib import Path

# Ensure repo root is on sys.path so tests can import `core` directly
REPO_ROOT = Path(__file__).parent.parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
