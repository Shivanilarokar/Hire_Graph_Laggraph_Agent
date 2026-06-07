"""HireGraph CLI entry point.

Run on a fresh clone with:  ``uv run python main.py``
Compiles the graph, writes graph_out/graph.png, runs the four canned
scenarios, and prints the scoreboard.
"""

from hiregraph.cli import main

if __name__ == "__main__":
    main()
