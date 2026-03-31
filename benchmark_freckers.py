import subprocess
import re
import csv
from pathlib import Path
from datetime import datetime


NUM_GAMES = 10
AGENT1 = "agent"
AGENT2 = "agent2"
RESULTS_FILE = f"results/V21_V17.csv"

WIN_RE = re.compile(r"game over, winner is (\w+)", re.IGNORECASE)
THINK_RE = re.compile(r"\[agent-(RED|BLUE)\] ThinkTime: ([0-9.]+)")
TURN_RE = re.compile(r"(RED|BLUE) to play \(turn (\d+)\)", re.IGNORECASE)

def run_game(red_agent, blue_agent):
    proc = subprocess.run(
        ["python", "-m", "referee", red_agent, blue_agent],
        capture_output=True,
        text=True
    )
    output = proc.stderr + "\n" + proc.stdout

    # Extract winner
    win_match = WIN_RE.search(output)
    winner_color = win_match.group(1).upper() if win_match else "UNKNOWN"
    winner = red_agent if winner_color == "RED" else blue_agent if winner_color == "BLUE" else "draw"

    # Extract total moves (based on turn announcements)
    turn_numbers = [int(t[1]) for t in TURN_RE.findall(output)]
    total_moves = max(turn_numbers) if turn_numbers else 0

    # Extract think times
    think_times = {"RED": [], "BLUE": []}
    for color, time_str in THINK_RE.findall(output):
        think_times[color].append(float(time_str))

    avg_red = round(sum(think_times["RED"]) / len(think_times["RED"]), 4) if think_times["RED"] else None
    avg_blue = round(sum(think_times["BLUE"]) / len(think_times["BLUE"]), 4) if think_times["BLUE"] else None

    return {
        "red": red_agent,
        "blue": blue_agent,
        "winner": winner,
        "raw_winner": winner_color,
        "avg_think_time_red": avg_red,
        "avg_think_time_blue": avg_blue,
        "total_moves": total_moves
    }

def main():
    Path("results").mkdir(exist_ok=True)
    results = []

    for i in range(NUM_GAMES):
        red, blue = (AGENT1, AGENT2) if i % 2 == 0 else (AGENT2, AGENT1)
        print(f"Running game {i+1}: RED={red}, BLUE={blue}")
        result = run_game(red, blue)
        result["game"] = i + 1
        results.append(result)

    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nBenchmark complete! Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()