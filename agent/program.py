# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Freckers Agent with Enhanced Heuristics and Optimized Search
# Method: Minimax search with alpha-beta pruning, variable depth based on goal state

from referee.game import PlayerColor, Coord, Direction, Action, MoveAction, GrowAction
from referee.game.board import CellState
import time
import math
from collections import deque
import sys

BOARD_SIZE = 8
SEARCH_TIME_LIMIT = 5  # seconds per move
MAX_DEPTH = 3

VALID_DIRECTIONS = {
    PlayerColor.RED: {Direction.Right, Direction.Left, Direction.Down, Direction.DownLeft, Direction.DownRight},
    PlayerColor.BLUE: {Direction.Right, Direction.Left, Direction.Up, Direction.UpLeft, Direction.UpRight}
}

class LightweightBoard:
    """
    Lightweight representation of the Freckers board state to track
    frog positions and lily pads. Provides move generation, application,
    and evaluation functions for search.
    """
    def __init__(self):
        # Initialize board and frogs
        self.board = {Coord(r, c): CellState(None) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)}
        self.frogs = {PlayerColor.RED: set(), PlayerColor.BLUE: set()}
        self._initialize_board()

    def _initialize_board(self):
        """
        Place initial frogs and lily pads on the starting rows for RED and BLUE.
        Frogs occupy row 0 (RED) and row BOARD_SIZE-1 (BLUE), with lily pads below them.
        """
        for c in range(1, BOARD_SIZE - 1):
            red = Coord(0, c)
            blue = Coord(BOARD_SIZE - 1, c)
            # Red frog and lilies
            self.board[red] = CellState(PlayerColor.RED)
            self.board[Coord(1, c)] = CellState("LilyPad")
            self.frogs[PlayerColor.RED].add(red)
            # Blue frog and lilies
            self.board[blue] = CellState(PlayerColor.BLUE)
            self.board[Coord(BOARD_SIZE - 2, c)] = CellState("LilyPad")
            self.frogs[PlayerColor.BLUE].add(blue)

    def clone(self) -> 'LightweightBoard':
        """ Board state for search tree branching """
        new = LightweightBoard()
        new.board = self.board.copy()
        new.frogs = {PlayerColor.RED: set(self.frogs[PlayerColor.RED]), PlayerColor.BLUE: set(self.frogs[PlayerColor.BLUE])}
        return new

    def is_valid(self, coord: Coord) -> bool:
        """Check if the coordinate is within board bounds."""
        return 0 <= coord.r < BOARD_SIZE and 0 <= coord.c < BOARD_SIZE
    
    def is_win(self, color: PlayerColor) -> bool:
        """
        Determine if all frogs of the given color have reached the opponent's home row.
        """
        goal_row = BOARD_SIZE - 1 if color == PlayerColor.RED else 0
        return all(f.r == goal_row for f in self.frogs[color])

    def is_lily(self, coord: Coord) -> bool:
        """If lilypad, return true"""
        return self.is_valid(coord) and self.board[coord].state == "LilyPad"

    def apply_action(self, color: PlayerColor, action: Action):
        """
        Applies Move or Grow Action
        """
        if isinstance(action, MoveAction): self._apply_move(color, action)
        else: self._apply_grow(color)

    def _apply_move(self, color: PlayerColor, action: MoveAction):
        """
        Executes a sequence of jumps or single-step move:
        1. Remove frog from source cell
        2. Follow direction sequence, handling jumps over occupied cells
        3. Place frog at destination
        """
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
        """
        Grow lilies around each frog by one cell in all directions where empty.
        """
        for f in list(self.frogs[color]):
            for d in Direction:
                try:
                    adj = f + d
                    if self.is_valid(adj) and self.board[adj].state is None:
                        self.board[adj] = CellState("LilyPad")
                except ValueError:
                    continue

    def get_legal_moves(self, color: PlayerColor) -> list[Action]:
        """
        Generate all legal MoveAction and one GrowAction for the given color.
        Skips frogs that have already reached the goal row.
        """
        
        moves = []
        
        goal_row = BOARD_SIZE - 1 if color == PlayerColor.RED else 0

        for f in self.frogs[color]:
            # skip frogs that have already finished
            if f.r == goal_row:
                continue    
            # Jump sequences via DFS [Adapted from Part A]        
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
            for d in VALID_DIRECTIONS[color]:
                try:
                    dest = f + d
                    if self.is_valid(dest) and self.is_lily(dest): moves.append(MoveAction(f, (d,)))
                except ValueError:
                    continue
        # Include grow
        moves.append(GrowAction())
        return moves
    
    def generate_jumps(self, start: Coord, color: PlayerColor) -> list[MoveAction]:
        """
        Return all possible jump sequences starting from a single frog position.
        Adapted from Part A, used for threat and jump potential heuristics
        """
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
        # Goal saturation - COunt the frogs on goal row
        goal_row = BOARD_SIZE -1 if color==PlayerColor.RED else 0
        goal_count = sum(1 for f in self.frogs[color] if f.r==goal_row)
        # Chebyshev distance to goal row
        def cheb_to_goal(frog: Coord, goal_row: int) -> int:
            return min(
                max(abs(goal_row - frog.r), abs(col - frog.c))
                for col in range(BOARD_SIZE)
            )
        self_dist = sum(cheb_to_goal(f, goal_row) for f in self.frogs[color])

        opp_goal = BOARD_SIZE - 1 if enemy == PlayerColor.RED else 0
        opp_dist = sum(cheb_to_goal(f, opp_goal) for f in self.frogs[enemy])
        
        # Mobility - How many moves does it have
        self_moves = len(self.get_legal_moves(color))
        opp_moves = len(self.get_legal_moves(enemy))
        # Threat detection: Count trheats by enemy jumps
        threats = 0
        opp_jumps = [mv for f in self.frogs[enemy] for mv in self.generate_jumps(f, enemy)]
        threatened = {mv.coord for mv in opp_jumps}
        threats = sum(1 for f in self.frogs[color] if f in threatened)
        # Jump potential count
        jump_self = sum(len(self.generate_jumps(frog, color)) for frog in self.frogs[color])
        jump_opp = sum(len(self.generate_jumps(frog, enemy)) for frog in self.frogs[enemy])
        
        # Concentration bonus: inverse avg pairwise distance - together is better
        def avg_pairdist(fl):
            if len(fl)<2: return 0
            pts=[(p.r,p.c) for p in fl]
            dsum=0;cnt=0
            for i in range(len(pts)):
                for j in range(i+1,len(pts)):
                    dsum+=math.hypot(pts[i][0]-pts[j][0],pts[i][1]-pts[j][1]); cnt+=1
            return (cnt>0 and (1/dsum*cnt) or 0)
        self_conc=avg_pairdist(self.frogs[color]); opp_conc=avg_pairdist(self.frogs[enemy])
        # Grow potential: Empty adjacencies around frogs - increase board control
        grow_pot=0
        for f in self.frogs[color]:
            for d in Direction:
                try:
                    adj=f+d
                    if self.is_valid(adj) and self.is_empty(adj): grow_pot+=1
                except: pass
        score = 0
        # Combine heuristic and weights
        # if (goal_count <= 5):
        #    score -= self_dist + goal_count 
        score += goal_count * 1000.0
        score += (opp_dist - self_dist) * 5.0
        score += (self_moves - opp_moves) * 0.5
        score -= threats * 5.0
        # score += (opp_euc - self_euc) * 0.2
        score += (self_conc - opp_conc) * 10.0
        score += grow_pot * 0.3
        score += (jump_self - jump_opp) * 1.0
        return score

class Agent:
    def __init__(self, color: PlayerColor, **referee: dict):
        self.color=color; self.enemy = PlayerColor.RED if color==PlayerColor.BLUE else PlayerColor.BLUE
        self.board=LightweightBoard()
    
    def find_finish_path(self) -> list[MoveAction] | None:
        """
        BFS on just the one non-finished frog to find a shortest
        sequence of MoveActions that brings it to the goal row,
        ignoring opponent moves.
        """
        # identify the lone straggler frog
        goal_row = BOARD_SIZE - 1 if self.color == PlayerColor.RED else 0
        frogs = list(self.board.frogs[self.color])
        # filter out any already home
        stragglers = [f for f in frogs if f.r != goal_row]
        if len(stragglers) != 1:
            return None
        start_frog = stragglers[0]

        # BFS queue holds tuples: (board_state, frog_pos, action_sequence)
        Q = deque()
        Q.append((self.board.clone(), start_frog, []))
        visited = set([start_frog])

        while Q:
            b, pos, path = Q.popleft()
            # if that frog is now on the goal row, we’re done
            if pos.r == goal_row:
                return path

            # only generate moves for that one frog
            for act in b.get_legal_moves(self.color):
                if not isinstance(act, MoveAction) or act.coord != start_frog:
                    continue
                b2 = b.clone()
                b2.apply_action(self.color, act)
                # compute new position of that frog
                new_pos = next(iter(b2.frogs[self.color] - (self.board.frogs[self.color] - {start_frog})))
                if new_pos in visited:
                    continue
                visited.add(new_pos)
                Q.append((b2, new_pos, path + [act]))
        return None
    
    
    
    def action(self, **referee: dict) -> Action:
        """
        Decide on the next action:
        1. Immediate win check
        2. Adjust search depth based on goal_count
        3. Run Minimax with alpha-beta until time limit
        """
        start_time = time.perf_counter()
        # immediate win
        for act in self.board.get_legal_moves(self.color):
            b2=self.board.clone(); b2.apply_action(self.color,act)
            if b2.is_win(self.color): 
                return act
            
        # search depth change based on goal_count

        goal_row = BOARD_SIZE - 1 if self.color == PlayerColor.RED else 0
        goal_count = sum(1 for f in self.board.frogs[self.color] if f.r == goal_row)

        # Set search depth based on goal_count
        if goal_count >= 4:
            depth_limit = 4
        elif goal_count == 5:
            goal_row = BOARD_SIZE - 1 if self.color == PlayerColor.RED else 0
            goal_count = sum(1 for f in self.board.frogs[self.color] if f.r == goal_row)
            finish_seq = self.find_finish_path()
            if finish_seq:
                # return the very first MoveAction in that BFS path
                return finish_seq[0]
        else:
            depth_limit = MAX_DEPTH

        # Else, minimax search with alpha-beta
        start=time.time()
        best=GrowAction()
        bv=-math.inf
        alph=-math.inf
        bet=math.inf
        for act in self.board.get_legal_moves(self.color):
            # print("Action Test")
            # print(act)
            b2=self.board.clone();b2.apply_action(self.color,act)
            v=self.minimax(b2,depth_limit-1,alph,bet,False,start)
            if v>bv: bv, best = v, act
            alph=max(alph,bv)
            if time.time()-start>SEARCH_TIME_LIMIT: break
        end_time = time.perf_counter()
        think_time = end_time - start_time
        print(f"[agent-{self.color.name}] ThinkTime: {think_time:.4f}", file=sys.stderr)
        return best
    def update(self, color, action, **r): 
        self.board.apply_action(color,action)

    def minimax(self,b,depth,alpha,beta,maxim,start):
        """
        Standard recursive Minimax with alpha-beta pruning.
        Terminates at depth 0 or time limit and uses evaluate().
        """

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
