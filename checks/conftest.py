"""Shared pytest setup: put the project root on sys.path and make it the
cwd, so tests can `import book`/`import ingest`/`import replay` directly and
config.yaml's relative data paths resolve regardless of where pytest was
invoked from."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
