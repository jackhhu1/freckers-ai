# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Freckers Agent with Optimized Board Representation

from referee.game import PlayerColor, Coord, Direction, Action, MoveAction, GrowAction
from referee.game.board import CellState
import random
from typing import Optional

BOARD_SIZE = 8

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

DIRECTION_OFFSETS = {
    Direction.Up: (-1, 0),
    Direction.Down: (1, 0),
    Direction.Left: (0, -1),
    Direction.Right: (0, 1),
    Direction.UpLeft: (-1, -1),
    Direction.UpRight: (-1, 1),
    Direction.DownLeft: (1, -1),
    Direction.DownRight: (1, 1)
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

    def is_valid_coord(self, coord: Coord) -> bool:
        return 0 <= coord.r < BOARD_SIZE and 0 <= coord.c < BOARD_SIZE

    def is_lilypad(self, coord: Coord) -> bool:
        return self.board[coord].state == "LilyPad"

    def is_empty(self, coord: Coord) -> bool:
        return self.board[coord].state is None

    def apply_action(self, color: PlayerColor, action: Action):
        if isinstance(action, MoveAction):
            self._apply_move(color, action)
        elif isinstance(action, GrowAction):
            self._apply_grow(color)

    def _apply_move(self, color: PlayerColor, action: MoveAction):
        pos = action.coord
        self.board[pos] = CellState(None)
        self.frogs[color].remove(pos)

        for direction in action.directions:
            pos = pos + direction
            if self.board[pos].state in [PlayerColor.RED, PlayerColor.BLUE]:
                pos = pos + direction

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


class Agent:
    def __init__(self, color: PlayerColor, **referee: dict):
        self.color = color
        self.enemy = PlayerColor.RED if color == PlayerColor.BLUE else PlayerColor.BLUE
        self.board = LightweightBoard()

    def action(self, **referee: dict) -> Action:
        moves = self._get_legal_moves(self.color)
        if moves:
            return random.choice(moves)
        return GrowAction()

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        self.board.apply_action(color, action)

    def _get_legal_moves(self, color: PlayerColor) -> list[MoveAction]:
        legal_moves = [GrowAction()]
        for frog in self.board.frogs[color]:
            for direction in VALID_DIRECTIONS[color]:
                try:
                    dest = frog + direction
                    if self.board.is_valid_coord(dest) and self.board.is_lilypad(dest):
                        legal_moves.append(MoveAction(frog, direction))
                except ValueError:
                    continue
        return legal_moves
