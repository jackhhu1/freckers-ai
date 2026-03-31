# 🐸 Freckers AI

An AI game-playing agent for **Freckers** - a two-player board game played on an 8×8 grid where frogs must race to the opponent's home row using moves and jumps across lily pads.

---

## 🎮 Game Overview

Freckers is a turn-based strategy game:
- **RED** frogs start on row 0 and race to row 7
- **BLUE** frogs start on row 7 and race to row 0
- Frogs can **step** onto adjacent lily pads or **jump** over other frogs onto lily pads
- Players can also **grow** new lily pads around all their frogs
- The first player to get all their frogs to the opponent's home row wins

---

## 🧠 Agent Design

The main agent lives in `agent/program.py` and is the final submission agent. It uses:

### Search Algorithm
- **Minimax with Alpha-Beta Pruning** - explores game trees up to a configurable depth while cutting off branches that can't affect the result
- **Time-limited search** - respects a 5-second per-move limit, cutting off early if needed
- **Adaptive depth** - increases search depth to 4 plies when 4+ frogs have reached the goal row, for stronger endgame play
- **BFS endgame finisher** - when only one frog remains unfinished, a BFS finds the shortest path to the goal row

### Heuristic Evaluation
The board evaluation function combines several weighted signals:

| Feature | Description | Weight |
|---|---|---|
| Goal saturation | Number of frogs on the goal row | +1000 |
| Chebyshev distance | Distance differential to goal (self vs opponent) | ×5.0 |
| Mobility | Move count advantage | ×0.5 |
| Threat avoidance | Frogs threatened by opponent jumps | −5.0 each |
| Concentration | Inverse average pairwise distance (frogs together = better jump chains) | ×10.0 |
| Jump potential | Jump opportunity count advantage | ×1.0 |
| Grow potential | Number of empty cells adjacent to frogs | ×0.3 |

### Board Representation
- `LightweightBoard` - a dictionary-based board state that tracks frog positions and lily pads
- Supports `clone()` for efficient search-tree branching
- Move generation handles single-step moves and multi-hop jump sequences via DFS

---

## 📁 Project Structure

```
freckers-ai/
├── agent/                  # Final submission agent
│   ├── program.py          # Main agent (Minimax + alpha-beta)
│   ├── V21.py              # Development version of the agent
│   ├── V6–V17*.py          # Earlier agent iterations
│   ├── greedy.py           # Greedy baseline agent
│   ├── randomAgent.py      # Random baseline agent
│   └── __init__.py
├── agent2/                 # Secondary agent for internal benchmarking
├── referee/                # Official game referee (provided by subject)
├── results/                # Benchmark CSV results
├── benchmark_freckers.py   # Script to run multiple games and log results
└── team.py                 # Team metadata
```

---

## 🚀 Running the Agent

Requires **Python 3.12+**.

### Play a game (agent vs agent)

```bash
python -m referee agent agent2
```

### Run benchmarks

Runs multiple games between two agents and saves results to a CSV:

```bash
python benchmark_freckers.py
```

Configure `NUM_GAMES`, `AGENT1`, `AGENT2`, and `RESULTS_FILE` at the top of the script.

---

## 📊 Development History

The agent went through 21+ iterations, progressively adding and tuning:

| Version | Key Changes |
|---|---|
| V3–V5 | Basic Minimax, win condition detection |
| V6–V9 | Alpha-beta pruning, improved move generation |
| V10–V12 | Heuristic tuning (distance, mobility) |
| V14–V15 | Euclidean distance experiments |
| V17 | Heuristic weight tuning pass |
| V21 | Chebyshev distance, concentration bonus, threat detection, adaptive depth, BFS endgame finisher |

---

## 👤 Author

**Jack Hu**
