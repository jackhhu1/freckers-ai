# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Freckers Agent with Optimized Board Representation and Minimax Search

from referee.game import PlayerColor, Coord, Direction, Action, MoveAction, GrowAction
from referee.game.board import CellState
import time
import math
from functools import lru_cache

BOARD_SIZE = 8
SEARCH_TIME_LIMIT = 5  # seconds per move
MAX_DEPTH = 3

VALID_DIRECTIONS = {
    PlayerColor.RED: {Direction.Right, Direction.Left, Direction.Down, Direction.DownLeft, Direction.DownRight},
    PlayerColor.BLUE: {Direction.Right, Direction.Left, Direction.Up, Direction.UpLeft, Direction.UpRight}
}

class LightweightBoard:
    def __init__(self):
        # Initialize empty board and frog positions
        self.board = {Coord(r, c): CellState(None) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)}
        self.frogs = {PlayerColor.RED: set(), PlayerColor.BLUE: set()}
        self._initialize_board()

    def _initialize_board(self):
        for c in range(1, BOARD_SIZE - 1):
            red = Coord(0, c)
            blue = Coord(BOARD_SIZE-1, c)
            self.board[red] = CellState(PlayerColor.RED)
            self.board[Coord(1, c)] = CellState('LilyPad')
            self.frogs[PlayerColor.RED].add(red)
            self.board[blue] = CellState(PlayerColor.BLUE)
            self.board[Coord(BOARD_SIZE-2, c)] = CellState('LilyPad')
            self.frogs[PlayerColor.BLUE].add(blue)

    def clone(self) -> 'LightweightBoard':
        nb = LightweightBoard()
        nb.board = self.board.copy()
        nb.frogs = {PlayerColor.RED: set(self.frogs[PlayerColor.RED]), PlayerColor.BLUE: set(self.frogs[PlayerColor.BLUE])}
        return nb

    def is_valid_coord(self, coord: Coord) -> bool:
        return 0 <= coord.r < BOARD_SIZE and 0 <= coord.c < BOARD_SIZE

    def is_lilypad(self, coord: Coord) -> bool:
        return self.is_valid_coord(coord) and self.board[coord].state == 'LilyPad'

    def is_empty(self, coord: Coord) -> bool:
        return self.is_valid_coord(coord) and self.board[coord].state is None

    def is_win(self, color: PlayerColor) -> bool:
        goal_row = BOARD_SIZE-1 if color == PlayerColor.RED else 0
        return all(pos.r == goal_row for pos in self.frogs[color])

    def apply_action(self, color: PlayerColor, action: Action):
        if isinstance(action, MoveAction):
            self._apply_move(color, action)
        else:
            self._apply_grow(color)

    def _apply_move(self, color: PlayerColor, action: MoveAction):
        pos = action.coord
        if pos not in self.frogs[color]: return
        self.board[pos] = CellState(None)
        self.frogs[color].remove(pos)
        for d in action.directions:
            try:
                mid = pos + d
            except ValueError:
                return
            if self.is_valid_coord(mid) and self.board[mid].state in (PlayerColor.RED, PlayerColor.BLUE):
                try:
                    dest = mid + d
                except ValueError:
                    return
            else:
                dest = mid
            if not self.is_valid_coord(dest): return
            pos = dest
        self.frogs[color].add(pos)
        self.board[pos] = CellState(color)

    def _apply_grow(self, color: PlayerColor):
        for f in list(self.frogs[color]):
            for d in Direction:
                try:
                    adj = f + d
                except ValueError:
                    continue
                if self.is_valid_coord(adj) and self.is_empty(adj):
                    self.board[adj] = CellState('LilyPad')

    def get_legal_moves(self, color: PlayerColor) -> list[Action]:
        moves = [GrowAction()]
        for f in self.frogs[color]:
            for d in VALID_DIRECTIONS[color]:
                try:
                    dest = f + d
                except ValueError:
                    continue
                if self.is_valid_coord(dest) and self.is_lilypad(dest):
                    moves.append(MoveAction(f, (d,)))
            moves.extend(self.generate_jumps(f, color))
        return moves

    def generate_jumps(self, start: Coord, color: PlayerColor) -> list[MoveAction]:
        visited, results = set(), []
        stack = [(start, [])]
        for curr, path in stack:
            for d in VALID_DIRECTIONS[color]:
                try:
                    mid = curr + d
                    dest = mid + d
                except ValueError:
                    continue
                if not self.is_valid_coord(mid) or not self.is_valid_coord(dest):
                    continue
                if self.board[mid].state in (PlayerColor.RED, PlayerColor.BLUE) and self.is_lilypad(dest) and dest not in visited:
                    visited.add(dest)
                    new_path = path + [d]
                    results.append(MoveAction(start, tuple(new_path)))
                    stack.append((dest, new_path))
        return results

    @lru_cache(None)
    def _endrow_lilypads(self, color: PlayerColor) -> tuple[Coord,...]:
        goal_row = BOARD_SIZE-1 if color == PlayerColor.RED else 0
        return tuple(Coord(goal_row, c) for c in range(BOARD_SIZE))

    def evaluate(self, color: PlayerColor) -> float:
        enemy = PlayerColor.RED if color == PlayerColor.BLUE else PlayerColor.BLUE
        # Euclidean distance to nearest end-row lilypad
        endpads = self._endrow_lilypads(color)
        def frog_euclid(frog):
            return min(math.dist((frog.r,frog.c),(pad.r,pad.c)) for pad in endpads)
        self_e = sum(frog_euclid(f) for f in self.frogs[color])
        opp_e = sum(frog_euclid(f) for f in self.frogs[enemy])
        # Jump potentials
        jump_self = sum(len(self.generate_jumps(f, color)) for f in self.frogs[color])
        jump_opp = sum(len(self.generate_jumps(f, enemy)) for f in self.frogs[enemy])
        # Combined heuristic
        return (opp_e - self_e) * 2.0 + (jump_self - jump_opp) * 0.5

class Agent:
    def __init__(self, color: PlayerColor, **referee: dict):
        self.color, self.enemy = color, (PlayerColor.RED if color==PlayerColor.BLUE else PlayerColor.BLUE)
        self.board = LightweightBoard()

    def action(self, **referee: dict) -> Action:
        # Immediate win or advance
        goal_row = BOARD_SIZE-1 if self.color==PlayerColor.RED else 0
        best_adv, adv_count = None, sum(1 for f in self.board.frogs[self.color] if f.r==goal_row)
        for mv in self.board.get_legal_moves(self.color):
            b2 = self.board.clone(); b2.apply_action(self.color,mv)
            newc = sum(1 for f in b2.frogs[self.color] if f.r==goal_row)
            if b2.is_win(self.color): return mv
            if newc>adv_count: best_adv,adv_count = mv,newc
        if best_adv: return best_adv
        # Minimax search
        start=time.time(); best, bv=GrowAction(),-math.inf; a,b=-math.inf,math.inf
        for mv in self.board.get_legal_moves(self.color):
            b2=self.board.clone(); b2.apply_action(self.color,mv)
            val=self.minimax(b2, MAX_DEPTH-1, a,b, False, start)
            if val>bv: bv, best = val, mv
            a = max(a,bv)
            if time.time()-start>SEARCH_TIME_LIMIT: break
        return best

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        self.board.apply_action(color, action)

    def minimax(self, board: LightweightBoard, depth:int, alpha:float, beta:float,
                maximizing:bool, start_time:float) -> float:
        if depth==0 or time.time()-start_time>SEARCH_TIME_LIMIT:
            return board.evaluate(self.color)
        player = self.color if maximizing else self.enemy
        moves = board.get_legal_moves(player)
        if maximizing:
            v=-math.inf
            for mv in moves:
                b2=board.clone(); b2.apply_action(player,mv)
                v2=self.minimax(b2,depth-1,alpha,beta,False,start_time)
                v=max(v,v2); alpha=max(alpha,v)
                if alpha>=beta: break
            return v
        else:
            v= math.inf
            for mv in moves:
                b2=board.clone(); b2.apply_action(player,mv)
                v2=self.minimax(b2,depth-1,alpha,beta,True,start_time)
                v=min(v,v2); beta=min(beta,v)
                if beta<=alpha: break
            return v
