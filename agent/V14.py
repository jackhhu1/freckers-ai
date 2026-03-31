# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Freckers Agent with Enhanced Heuristics and Optimized Search

from referee.game import PlayerColor, Coord, Direction, Action, MoveAction, GrowAction
from referee.game.board import CellState
import time
import math

BOARD_SIZE = 8
SEARCH_TIME_LIMIT = 5  # seconds per move
MAX_DEPTH = 3

VALID_DIRECTIONS = {
    PlayerColor.RED: {Direction.Right, Direction.Left, Direction.Down, Direction.DownLeft, Direction.DownRight},
    PlayerColor.BLUE: {Direction.Right, Direction.Left, Direction.Up, Direction.UpLeft, Direction.UpRight}
}

class LightweightBoard:
    def __init__(self):
        # Initialize board and frogs
        self.board = {Coord(r, c): CellState(None) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)}
        self.frogs = {PlayerColor.RED: set(), PlayerColor.BLUE: set()}
        self._initialize_board()

    def _initialize_board(self):
        for c in range(1, BOARD_SIZE - 1):
            red = Coord(0, c)
            blue = Coord(BOARD_SIZE - 1, c)
            self.board[red] = CellState(PlayerColor.RED)
            self.board[Coord(1, c)] = CellState("LilyPad")
            self.frogs[PlayerColor.RED].add(red)

            self.board[blue] = CellState(PlayerColor.BLUE)
            self.board[Coord(BOARD_SIZE - 2, c)] = CellState("LilyPad")
            self.frogs[PlayerColor.BLUE].add(blue)

    def clone(self) -> 'LightweightBoard':
        new = LightweightBoard()
        new.board = self.board.copy()
        new.frogs = {PlayerColor.RED: set(self.frogs[PlayerColor.RED]), PlayerColor.BLUE: set(self.frogs[PlayerColor.BLUE])}
        return new

    def is_valid(self, coord: Coord) -> bool:
        return 0 <= coord.r < BOARD_SIZE and 0 <= coord.c < BOARD_SIZE

    def is_lily(self, coord: Coord) -> bool:
        return self.is_valid(coord) and self.board[coord].state == "LilyPad"

    def apply_action(self, color: PlayerColor, action: Action):
        if isinstance(action, MoveAction): self._apply_move(color, action)
        else: self._apply_grow(color)

    def _apply_move(self, color: PlayerColor, action: MoveAction):
        src = action.coord
        if src not in self.frogs[color]: return
        # remove frog
        self.board[src] = CellState(None)
        self.frogs[color].remove(src)
        pos = src
        for d in action.directions:
            try:
                mid = pos + d
                # jump if occupied
                dest = mid + d if self.board[mid].state in (PlayerColor.RED, PlayerColor.BLUE) else mid
            except ValueError:
                return
            if not self.is_valid(dest): return
            pos = dest
        # place frog
        self.frogs[color].add(pos)
        self.board[pos] = CellState(color)

    def _apply_grow(self, color: PlayerColor):
        for f in list(self.frogs[color]):
            for d in Direction:
                try:
                    adj = f + d
                    if self.is_valid(adj) and self.board[adj].state is None:
                        self.board[adj] = CellState("LilyPad")
                except ValueError:
                    continue

    def get_legal_moves(self, color: PlayerColor) -> list[Action]:
        moves = [GrowAction()]
        for f in self.frogs[color]:
            for d in VALID_DIRECTIONS[color]:
                try:
                    dest = f + d
                    if self.is_valid(dest) and self.is_lily(dest): moves.append(MoveAction(f, (d,)))
                except ValueError:
                    continue
            # jumps
            stack = [(f, [])]
            visited = set()
            while stack:
                curr, path = stack.pop()
                for d in VALID_DIRECTIONS[color]:
                    try:
                        mid = curr + d; dest = mid + d
                    except ValueError:
                        continue
                    if self.is_valid(mid) and self.is_valid(dest) and self.board[mid].state in (PlayerColor.RED, PlayerColor.BLUE) and self.is_lily(dest) and dest not in visited:
                        visited.add(dest)
                        newp = path + [d]
                        moves.append(MoveAction(f, tuple(newp)))
                        stack.append((dest, newp))
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
                    if not (self.is_valid(mid) and self.is_valid(dest)):
                        continue
                    if self.board[mid].state in [PlayerColor.RED, PlayerColor.BLUE] and self.is_lily(dest) and dest not in visited:
                        visited.add(dest)
                        new_path = path + [d]
                        results.append(MoveAction(start, tuple(new_path)))
                        stack.append((dest, new_path))
                except ValueError:
                    continue
        return results

    def evaluate(self, color: PlayerColor) -> float:
        enemy = PlayerColor.RED if color == PlayerColor.BLUE else PlayerColor.BLUE
        # Goal saturation
        home_row = BOARD_SIZE -1 if color==PlayerColor.RED else 0
        home_count = sum(1 for f in self.frogs[color] if f.r==home_row)
        # Manhattan goal distance
        self_dist = sum((home_row - f.r) if color==PlayerColor.RED else (f.r - home_row) for f in self.frogs[color])
        opp_home = BOARD_SIZE-1 if enemy==PlayerColor.RED else 0
        opp_dist = sum((opp_home - f.r) if enemy==PlayerColor.RED else (f.r - opp_home) for f in self.frogs[enemy])
        # Mobility
        self_moves = len(self.get_legal_moves(color))
        opp_moves = len(self.get_legal_moves(enemy))
        # Threat detection
        threats = 0
        opp_jumps = [mv for f in self.frogs[enemy] for mv in self.generate_jumps(f, enemy)]
        threatened = {mv.coord for mv in opp_jumps}
        threats = sum(1 for f in self.frogs[color] if f in threatened)
        # Lily proximity
        # lilies = [pos for pos, cs in self.board.items() if cs.state=="LilyPad"]
        # def nearest_euclid(p): return min(math.hypot(p.r-l.r, p.c-l.c) for l in lilies) if lilies else BOARD_SIZE
        # self_euc = sum(nearest_euclid(f) for f in self.frogs[color])
        # opp_euc = sum(nearest_euclid(f) for f in self.frogs[enemy])
        # Concentration bonus: inverse avg pairwise distance
        def avg_pairdist(fl):
            if len(fl)<2: return 0
            pts=[(p.r,p.c) for p in fl]
            dsum=0;cnt=0
            for i in range(len(pts)):
                for j in range(i+1,len(pts)):
                    dsum+=math.hypot(pts[i][0]-pts[j][0],pts[i][1]-pts[j][1]); cnt+=1
            return (cnt>0 and (1/dsum*cnt) or 0)
        self_conc=avg_pairdist(self.frogs[color]); opp_conc=avg_pairdist(self.frogs[enemy])
        # Grow potential
        grow_pot=0
        for f in self.frogs[color]:
            for d in Direction:
                try:
                    adj=f+d
                    if self.is_valid(adj) and self.is_empty(adj): grow_pot+=1
                except: pass
        # Combine heuristic
        score = 0
        score += home_count * 100.0
        score += (opp_dist - self_dist) * 2.0
        score += (self_moves - opp_moves) * 0.5
        score -= threats * 5.0
        # score += (opp_euc - self_euc) * 0.2
        score += (self_conc - opp_conc) * 10.0
        score += grow_pot * 0.3
        return score

class Agent:
    def __init__(self, color: PlayerColor, **referee: dict):
        self.color=color; self.enemy = PlayerColor.RED if color==PlayerColor.BLUE else PlayerColor.BLUE
        self.board=LightweightBoard()
    def action(self, **referee: dict) -> Action:
        # immediate win
        # for act in self.board.get_legal_moves(self.color):
        #     b2=self.board.clone(); b2.apply_action(self.color,act)
        #     if b2.is_win(self.color): return act
        start=time.time(); best=GrowAction();bv=-math.inf;alph=-math.inf;bet=math.inf
        for act in self.board.get_legal_moves(self.color):
            b2=self.board.clone();b2.apply_action(self.color,act)
            v=self.minimax(b2,MAX_DEPTH-1,alph,bet,False,start)
            if v>bv: bv, best = v, act
            alph=max(alph,bv)
            if time.time()-start>SEARCH_TIME_LIMIT: break
        return best
    def update(self, color, action, **r): self.board.apply_action(color,action)
    def minimax(self,b,depth,alpha,beta,maxim,start):
        if depth==0 or time.time()-start>SEARCH_TIME_LIMIT: return b.evaluate(self.color)
        player=self.color if maxim else self.enemy
        moves=b.get_legal_moves(player)
        if maxim:
            v=-math.inf
            for m in moves:
                b2=b.clone(); b2.apply_action(player,m)
                v2=self.minimax(b2,depth-1,alpha,beta,False,start)
                v=max(v,v2); alpha=max(alpha,v)
                if beta<=alpha: break
            return v
        else:
            v=math.inf
            for m in moves:
                b2=b.clone(); b2.apply_action(player,m)
                v2=self.minimax(b2,depth-1,alpha,beta,True,start)
                v=min(v,v2); beta=min(beta,v)
                if beta<=alpha: break
            return v
