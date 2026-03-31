# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Freckers Agent with Corrected and Optimized Implementation

from referee.game import PlayerColor, Coord, Direction, Action, MoveAction, GrowAction
from referee.game.board import CellState
import time
import math

BOARD_SIZE = 8
SEARCH_TIME_LIMIT = 1.5  # seconds per move
BASE_DEPTH = 3           # base minimax depth
MOVE_ORDERING_WIDTH = 8  # consider top-K moves

VALID_DIRECTIONS = {
    PlayerColor.RED: {Direction.Right, Direction.Left, Direction.Down, Direction.DownLeft, Direction.DownRight},
    PlayerColor.BLUE:{Direction.Right, Direction.Left, Direction.Up, Direction.UpLeft, Direction.UpRight}
}

class LightweightBoard:
    __slots__ = ('board','frogs')
    def __init__(self):
        # Initialize empty board and frog positions
        self.board = {Coord(r,c): CellState(None) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)}
        self.frogs = {PlayerColor.RED:set(), PlayerColor.BLUE:set()}
        # Set initial frogs and lily pads
        for c in range(1, BOARD_SIZE-1):
            r0 = Coord(0,c); r1 = Coord(1,c)
            b7 = Coord(BOARD_SIZE-1,c); b6 = Coord(BOARD_SIZE-2,c)
            self.board[r0] = CellState(PlayerColor.RED)
            self.board[r1] = CellState('LilyPad')
            self.frogs[PlayerColor.RED].add(r0)
            self.board[b7] = CellState(PlayerColor.BLUE)
            self.board[b6] = CellState('LilyPad')
            self.frogs[PlayerColor.BLUE].add(b7)

    def clone(self):
        nb = object.__new__(LightweightBoard)
        nb.board = self.board.copy()
        nb.frogs = {col:set(pos) for col,pos in self.frogs.items()}
        return nb

    def is_valid(self,coord:Coord)->bool:
        return 0<=coord.r<BOARD_SIZE and 0<=coord.c<BOARD_SIZE

    def is_lily(self,coord:Coord)->bool:
        return self.is_valid(coord) and self.board[coord].state=='LilyPad'

    def apply_action(self,color:PlayerColor,action:Action):
        if isinstance(action,MoveAction): self._move(color,action)
        else: self._grow(color)

    def _move(self,color,action:MoveAction):
        src=action.coord
        if src not in self.frogs[color]: return
        # remove frog and underlying pad
        self.frogs[color].remove(src)
        self.board[src]=CellState(None)
        pos=src
        # apply each direction
        for d in action.directions:
            # step to mid
            try: mid=pos+d
            except ValueError: return
            # if occupied, jump
            if self.board[mid].state in (PlayerColor.RED,PlayerColor.BLUE):
                try: dest=mid+d
                except ValueError: return
            else:
                dest=mid
            if not self.is_valid(dest): return
            pos=dest
        # place frog
        self.frogs[color].add(pos)
        self.board[pos]=CellState(color)

    def _grow(self,color):
        for f in list(self.frogs[color]):
            for d in Direction:
                try: adj=f+d
                except ValueError: continue
                if self.is_valid(adj) and self.board[adj].state is None:
                    self.board[adj]=CellState('LilyPad')

    def legal_moves(self,color:PlayerColor)->list[Action]:
        moves=[GrowAction()]
        goal=(BOARD_SIZE-1 if color==PlayerColor.RED else 0)
        # jumps
        for f in self.frogs[color]:
            if f.r==goal: continue
            for d in VALID_DIRECTIONS[color]:
                try:
                    mid=f+d; dest=mid+d
                except ValueError: continue
                if self.board[mid].state in (PlayerColor.RED,PlayerColor.BLUE) and self.is_lily(dest):
                    moves.append(MoveAction(f,(d,)))
        # steps
        for f in self.frogs[color]:
            if f.r==goal: continue
            for d in VALID_DIRECTIONS[color]:
                try: dest=f+d
                except ValueError: continue
                if self.is_lily(dest): moves.append(MoveAction(f,(d,)))
        return moves

    def evaluate(self,color:PlayerColor)->float:
        # terminal wins
        me=self.frogs[color]; opp=self.frogs[PlayerColor.RED if color==PlayerColor.BLUE else PlayerColor.BLUE]
        if all(f.r==BOARD_SIZE-1 for f in me): return 1e6
        if all(f.r==0 for f in opp): return -1e6
        # distance sum
        sd=sum((BOARD_SIZE-1-f.r) if color==PlayerColor.RED else f.r for f in me)
        od=sum((BOARD_SIZE-1-f.r) if (PlayerColor.RED if color==PlayerColor.BLUE else PlayerColor.BLUE)==PlayerColor.RED else f.r for f in opp)
        # jump counts
        jp=sum(self._jump_count(f,color) for f in me)
        op=sum(self._jump_count(f,PlayerColor.RED if color==PlayerColor.BLUE else PlayerColor.BLUE) for f in opp)
        return (od-sd)*2.0 + (jp-op)*1.0

    def _jump_count(self,f,color):
        cnt=0
        for d in VALID_DIRECTIONS[color]:
            try: mid=f+d; dest=mid+d
            except ValueError: continue
            if self.board[mid].state in (PlayerColor.RED,PlayerColor.BLUE) and self.is_lily(dest): cnt+=1
        return cnt

class Agent:
    def __init__(self,color:PlayerColor,**referee):
        self.color=color
        self.enemy=PlayerColor.RED if color==PlayerColor.BLUE else PlayerColor.BLUE
        self.board=LightweightBoard()

    def action(self,**referee)->Action:
        start=time.time()
        # immediate win
        for m in self.board.legal_moves(self.color):
            b2=self.board.clone(); b2.apply_action(self.color,m)
            if all(f.r==(BOARD_SIZE-1) for f in b2.frogs[self.color]): return m
        # dynamic depth
        done=sum(1 for f in self.board.frogs[self.color] if f.r==(BOARD_SIZE-1))
        depth=BASE_DEPTH+done
        # order and search
        moves=self.board.legal_moves(self.color)
        scored=[(self._eval_move(m),m) for m in moves]
        scored.sort(reverse=True,key=lambda x:x[0])
        best,mv=-math.inf,moves[0]
        alpha,beta=-math.inf,math.inf
        for _,m in scored[:MOVE_ORDERING_WIDTH]:
            b2=self.board.clone(); b2.apply_action(self.color,m)
            v=self._minimax(b2,depth-1,alpha,beta,False,start)
            if v>best: best,mv=v,m
            alpha=max(alpha,best)
            if time.time()-start>SEARCH_TIME_LIMIT: break
        return mv

    def update(self,color,action,**referee): self.board.apply_action(color,action)
    def _eval_move(self,m): b2=self.board.clone(); b2.apply_action(self.color,m);return b2.evaluate(self.color)

    def _minimax(self,board,depth,alpha,beta,maximizing,start):
        if depth==0 or time.time()-start>SEARCH_TIME_LIMIT: return board.evaluate(self.color)
        player=self.color if maximizing else self.enemy
        moves=board.legal_moves(player)
        scored=[(self._eval_static(board,player,m),m) for m in moves]
        scored.sort(reverse=maximizing,key=lambda x:x[0])
        if maximizing:
            v=-math.inf
            for _,m in scored[:MOVE_ORDERING_WIDTH]:
                b2=board.clone(); b2.apply_action(player,m)
                v2=self._minimax(b2,depth-1,alpha,beta,False,start)
                v=max(v,v2);alpha=max(alpha,v)
                if beta<=alpha: break
            return v
        else:
            v=math.inf
            for _,m in scored[:MOVE_ORDERING_WIDTH]:
                b2=board.clone(); b2.apply_action(player,m)
                v2=self._minimax(b2,depth-1,alpha,beta,True,start)
                v=min(v,v2);beta=min(beta,v)
                if beta<=alpha: break
            return v

    def _eval_static(self,board,player,m):
        b2=board.clone(); b2.apply_action(player,m)
        return b2.evaluate(self.color)
