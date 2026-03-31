# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Freckers Agent with Optimized Board Representation and Minimax Search

from referee.game import PlayerColor, Coord, Direction, Action, MoveAction, GrowAction
from referee.game.board import CellState
import random
import time
from typing import Optional
import math

BOARD_SIZE = 8
SEARCH_TIME_LIMIT = 1.5  # seconds per move
MAX_DEPTH = 3

VALID_DIRECTIONS = {
    PlayerColor.RED: {
        Direction.Right, Direction.Left, Direction.Down,
        Direction.DownLeft, Direction.DownRight
    },
    PlayerColor.BLUE: {
        Direction.Right, Direction.Left, Direction.Up,
        Direction.UpLeft, Direction.UpRight
    }
}

class LightweightBoard:
    def __init__(self):
        self.board: dict[Coord, CellState] = {
            Coord(r, c): CellState(None) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
        }
        self.frogs = {
            PlayerColor.RED: set(),
            PlayerColor.BLUE: set()
        }
        self._initialize_board()

    def _initialize_board(self):
        for c in range(1, BOARD_SIZE - 1):
            red_coord = Coord(0, c)
            blue_coord = Coord(7, c)
            self.board[red_coord] = CellState(PlayerColor.RED)
            self.board[Coord(1, c)] = CellState("LilyPad")
            self.frogs[PlayerColor.RED].add(red_coord)

            self.board[blue_coord] = CellState(PlayerColor.BLUE)
            self.board[Coord(6, c)] = CellState("LilyPad")
            self.frogs[PlayerColor.BLUE].add(blue_coord)

    def clone(self) -> 'LightweightBoard':
        new_board = LightweightBoard()
        new_board.board = self.board.copy()
        new_board.frogs = {
            PlayerColor.RED: set(self.frogs[PlayerColor.RED]),
            PlayerColor.BLUE: set(self.frogs[PlayerColor.BLUE])
        }
        return new_board

    def is_valid_coord(self, coord: Coord) -> bool:
        return 0 <= coord.r < BOARD_SIZE and 0 <= coord.c < BOARD_SIZE

    def is_lilypad(self, coord: Coord) -> bool:
        return self.is_valid_coord(coord) and self.board[coord].state == "LilyPad"

    def is_empty(self, coord: Coord) -> bool:
        return self.is_valid_coord(coord) and self.board[coord].state is None

    def is_win(self, color: PlayerColor) -> bool:
        goal_row = BOARD_SIZE - 1 if color == PlayerColor.RED else 0
        return all(pos.r == goal_row for pos in self.frogs[color])

    def apply_action(self, color: PlayerColor, action: Action):
        if isinstance(action, MoveAction):
            self._apply_move(color, action)
        elif isinstance(action, GrowAction):
            self._apply_grow(color)

    def _apply_move(self, color: PlayerColor, action: MoveAction):
        pos = action.coord
        if not self.is_valid_coord(pos) or pos not in self.frogs[color] or self.board[pos].state != color:
            return
        self.board[pos] = CellState(None)
        self.frogs[color].remove(pos)
        for direction in action.directions:
            mid = pos + direction
            dest = mid + direction if self.is_valid_coord(mid) and self.board[mid].state in [PlayerColor.RED, PlayerColor.BLUE] else pos + direction
            if not self.is_valid_coord(dest):
                return
            pos = dest
        if self.is_valid_coord(pos):
            self.frogs[color].add(pos)
            self.board[pos] = CellState(color)

    def _apply_grow(self, color: PlayerColor):
        for frog in self.frogs[color]:
            for direction in Direction:
                try:
                    adj = frog + direction
                    if self.is_valid_coord(adj) and self.is_empty(adj):
                        self.board[adj] = CellState("LilyPad")
                except ValueError:
                    continue

    def get_legal_moves(self, color: PlayerColor) -> list[Action]:
        moves: list[Action] = [GrowAction()]
        for frog in self.frogs[color]:
            moves.extend(self.generate_jumps(frog, color))
            for direction in VALID_DIRECTIONS[color]:
                try:
                    dest = frog + direction
                    if self.is_valid_coord(dest) and self.is_lilypad(dest):
                        moves.append(MoveAction(frog, (direction,)))
                except ValueError:
                    continue
        return moves

    def generate_jumps(self, start: Coord, color: PlayerColor) -> list[MoveAction]:
        visited = set()
        results = []
        stack = [(start, [])]
        directions = VALID_DIRECTIONS[color]
        while stack:
            curr, path = stack.pop()
            for d in directions:
                try:
                    mid = curr + d
                    dest = mid + d
                    if not (self.is_valid_coord(mid) and self.is_valid_coord(dest)):
                        continue
                    if self.board[mid].state in [PlayerColor.RED, PlayerColor.BLUE] and self.is_lilypad(dest) and dest not in visited:
                        visited.add(dest)
                        new_path = path + [d]
                        results.append(MoveAction(start, tuple(new_path)))
                        stack.append((dest, new_path))
                except ValueError:
                    continue
        return results

    def evaluate(self, color: PlayerColor) -> float:
        enemy = PlayerColor.RED if color == PlayerColor.BLUE else PlayerColor.BLUE
        # Sum of Manhattan distances to goal row: lower is better for us, worse for opponent
        self_dist = sum((7 - frog.r) if color == PlayerColor.RED else frog.r for frog in self.frogs[color])
        opp_dist = sum((7 - frog.r) if enemy == PlayerColor.RED else frog.r for frog in self.frogs[enemy])
        # Jump potential count
        jump_self = sum(len(self.generate_jumps(frog, color)) for frog in self.frogs[color])
        jump_opp = sum(len(self.generate_jumps(frog, enemy)) for frog in self.frogs[enemy])
        # Heuristic combination: prioritize reaching goal row
        return (opp_dist - self_dist) * 2.0 + (jump_self - jump_opp) * 1

class Agent:
    def __init__(self, color: PlayerColor, **referee: dict):
        self.color = color
        self.enemy = PlayerColor.RED if color == PlayerColor.BLUE else PlayerColor.BLUE
        self.board = LightweightBoard()

    def action(self, **referee: dict) -> Action:
        start = time.time()
        # immediate winning move
        for act in self.board.get_legal_moves(self.color):
            clone = self.board.clone()
            clone.apply_action(self.color, act)
            if clone.is_win(self.color):
                return act
        best_action = GrowAction()
        best_value = float("-inf")
        alpha, beta = float("-inf"), float("inf")
        for act in self.board.get_legal_moves(self.color):
            clone = self.board.clone()
            clone.apply_action(self.color, act)
            value = self.minimax(clone, MAX_DEPTH - 1, alpha, beta, False, start)
            if value > best_value:
                best_value = value
                best_action = act
            alpha = max(alpha, best_value)
            if time.time() - start > SEARCH_TIME_LIMIT:
                break
        return best_action

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        self.board.apply_action(color, action)

    def minimax(self, board: LightweightBoard, depth: int, alpha: float, beta: float,
                maximizing: bool, start_time: float) -> float:
        if depth == 0 or time.time() - start_time > SEARCH_TIME_LIMIT:
            return board.evaluate(self.color)
        current_player = self.color if maximizing else self.enemy
        actions = board.get_legal_moves(current_player)
        if maximizing:
            max_eval = float("-inf")
            for action in actions:
                clone = board.clone()
                clone.apply_action(current_player, action)
                eval = self.minimax(clone, depth - 1, alpha, beta, False, start_time)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float("inf")
            for action in actions:
                clone = board.clone()
                clone.apply_action(current_player, action)
                eval = self.minimax(clone, depth - 1, alpha, beta, True, start_time)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
