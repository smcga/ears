"""Compatibility launcher for the ears MCP server over stdio."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ears.server import main  # noqa: E402

if __name__ == "__main__":
    main()
