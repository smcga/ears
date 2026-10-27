"""Launcher for the ears MCP server (stdio). Point Claude at: .venv\\Scripts\\python.exe ears_server.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ears.server import main  # noqa: E402

if __name__ == "__main__":
    main()
