import random
import numpy as np
from enum import Enum
from collections import deque
from copy import deepcopy

BOARD_SIZE = 4

class ECellType(Enum):
    Empty = '#'
    First = 'X'
    Second = 'O'
    StartPoint = '.'

def RandomCellType():
    return random.choice([ECellType.Empty, ECellType.First, ECellType.Second])

def RandomNonEmptyCellType():
    return random.choice([ECellType.First, ECellType.Second])

class ERowType(Enum):
    Row = 0
    Column = 1

def getRandomRowType():
    return random.choice(list(ERowType))

class EMoveDirection(Enum):
    Down = -1
    Up = 1

def getRandomMoveDirection():
    return random.choice(list(EMoveDirection))

class Move:
    def __init__(self, cellType, rowType, row, column, direction):
        self.cellType = cellType
        self.rowType = rowType
        self.row = row
        self.column = column
        self.direction = direction
    
    @staticmethod
    def ParseMoveFromString(board_str, board_size=BOARD_SIZE):
        board_str = board_str.replace(' ', '')
        
        rows = [line.strip() for line in board_str.split('\n') if line.strip()]
        if len(rows) != board_size:
            raise ValueError(f"Expected {board_size} rows, got {len(rows)}")
            
        start_coords = (-1, -1)
        end_coords = (-1, -1)
        move_cellType = ECellType.Empty
        for row_number, row in enumerate(rows):
            if len(row) != board_size:
                raise ValueError(f"Expected {board_size} columns, got {len(row)}")
            for column_number, cell in enumerate(row):
                if cell == ECellType.First.value:
                    if end_coords != (-1, -1):
                        raise ValueError(f"Invalid cell character: '{cell}'")
                    move_cellType = cell
                    end_coords = (row_number, column_number)
                elif cell == ECellType.Second.value:
                    if end_coords != (-1, -1):
                        raise ValueError(f"Invalid cell character: '{cell}'")
                    move_cellType = cell
                    end_coords = (row_number, column_number)
                elif cell == ECellType.StartPoint.value:
                    if start_coords != (-1, -1):
                        raise ValueError(f"Invalid cell character: '{cell}'")
                    start_coords = (row_number, column_number)
                elif cell != ECellType.Empty.value:
                    raise ValueError(f"Invalid cell character: '{cell}'")
                    
        if start_coords == (-1, -1) or end_coords == (-1, -1):
            raise ValueError(f"Move is incomplete") 
            
        
        move_row = start_coords[0]
        move_column = start_coords[1]
        if start_coords[0] == end_coords[0]:
            move_rowType = ERowType.Row
            if start_coords[1] < end_coords[1]:
                move_direction = EMoveDirection.Up
            else:
                move_direction = EMoveDirection.Down
        elif start_coords[1] == end_coords[1]:
            move_rowType = ERowType.Column
            if start_coords[0] < end_coords[0]:
                move_direction = EMoveDirection.Up
            else:
                move_direction = EMoveDirection.Down
        else:
            raise ValueError(f"Move not contains single row or column") 
            
        return Move(
            move_cellType,
            move_rowType,
            move_row,
            move_column,
            move_direction
        )

def GetNextMove(cellType):
    return ECellType.Second if cellType == ECellType.First else ECellType.First

class GameState:
    BoardSize = BOARD_SIZE

    def __init__(self):
        self.Board_ = np.full((self.BoardSize, self.BoardSize), ECellType.Empty)

    def GetCell(self, i, j):
        return self.Board_[i, j]

    def SetCell(self, i, j, value):
        self.Board_[i,j] = value
        
    def Fill(self, f):
        for i in range(self.BoardSize):
            for j in range(self.BoardSize):
                self.Board_[i, j] = f()

    def SerializeAsString(self):
        gameState = ""
        for i in range(self.BoardSize):
            for j in range(self.BoardSize):
                gameState += str(self.Board_[i, j].value)
                if j != self.BoardSize - 1:
                    gameState += " "
            gameState += "\n"
        return gameState

    def ApplyMove(self, move):
        row_number = move.row if move.rowType == ERowType.Row else move.column
        start_point = move.column if move.rowType == ERowType.Row else move.row
        end_point = self.BoardSize - 1 if move.direction == EMoveDirection.Up else 0
        # print(row_number, start_point,end_point )
        # print(move.row, move.column)
        self.SetCell(move.row, move.column, ECellType(move.cellType))
        
        self.PermuteRow(row_number, start_point, end_point, - move.direction.value, move.rowType)
        return self

    def checkAllInARow(self, playerSymbol):
        """Check if all cells in any row, column, or diagonal match playerSymbol."""
        # Check rows
        for row in range(self.BoardSize):
            all_match = True
            for col in range(self.BoardSize):
                if self.Board_[row, col] != playerSymbol:
                    all_match = False
                    break
            if all_match:
                return True

        # Check columns
        for col in range(self.BoardSize):
            all_match = True
            for row in range(self.BoardSize):
                if self.Board_[row, col] != playerSymbol:
                    all_match = False
                    break
            if all_match:
                return True

        # Check main diagonal (top-left to bottom-right)
        all_match = True
        for i in range(self.BoardSize):
            if self.Board_[i, i] != playerSymbol:
                all_match = False
                break
        if all_match:
            return True

        # Check anti-diagonal (top-right to bottom-left)
        all_match = True
        for i in range(self.BoardSize):
            if self.Board_[i, self.BoardSize - 1 - i] != playerSymbol:
                all_match = False
                break
        if all_match:
            return True

        return False

    def IsFinish(self):
        return self.checkAllInARow(ECellType.First) or self.checkAllInARow(ECellType.Second)

        
    def PermuteRow(self, row_number, start, end, dir, rowType):
            if start > end:
                start, end = end, start  
            if rowType == ERowType.Row:
                self.Board_[row_number, start:end+1] = np.roll(self.Board_[row_number, start:end+1], dir)
            else: 
                self.Board_[start:end+1, row_number] = np.roll(self.Board_[start:end+1, row_number], dir)
                
    def IsOnCorner(self, col, row):
        return (col == 0 and row == 0) or \
               (col == 0 and row == self.BoardSize - 1) or \
               (col == self.BoardSize - 1 and row == 0) or \
               (col == self.BoardSize - 1 and row == self.BoardSize - 1)

    def IsOnEdge(self, col, row):
        return col == 0 or row == 0 or \
               col == self.BoardSize - 1 or row == self.BoardSize - 1

    def GetPossibleMoves(self, cellType):
        moves = []
        for i in range(self.BoardSize):
            for j in range(self.BoardSize):
                if self.GetCell(i, j) == cellType or self.GetCell(i, j) == ECellType.Empty:
                    # print(i, j, cellType, self.GetCell(i, j), cellType == self.GetCell(i, j))
                    if self.IsOnCorner(i, j):
                        # Column moves for corners
                        move_col = Move(cellType, ERowType.Column, i, j, 
                                       EMoveDirection.Up if i == 0 else EMoveDirection.Down)
                        moves.append(move_col)

                        # Row moves for corners
                        move_row = Move(cellType, ERowType.Row, i, j, 
                                       EMoveDirection.Up if j == 0 else EMoveDirection.Down)
                        moves.append(move_row)

                    elif self.IsOnEdge(i, j):
                        if i == 0 or i == self.BoardSize - 1:
                            # Column move for top/bottom edges
                            move_col = Move(cellType, ERowType.Column, i, j, 
                                          EMoveDirection.Up if i == 0 else EMoveDirection.Down)
                            moves.append(move_col)

                            # Row moves with both directions
                            for direction in [EMoveDirection.Down, EMoveDirection.Up]:
                                move_row = Move(cellType, ERowType.Row, i, j, direction)
                                moves.append(move_row)

                        if j == 0 or j == self.BoardSize - 1:
                            # Row move for left/right edges
                            move_row = Move(cellType, ERowType.Row, i, j, 
                                          EMoveDirection.Up if j == 0 else EMoveDirection.Down)
                            moves.append(move_row)

                            # Column moves with both directions
                            for direction in [EMoveDirection.Down, EMoveDirection.Up]:
                                move_col = Move(cellType, ERowType.Column, i, j, direction)
                                moves.append(move_col)
        return moves

    @staticmethod
    def ParseStateFromString(serializedState):
        state = GameState()
        lines = serializedState.replace(' ', '').strip().split('\n')

        for i in range(state.BoardSize):
            for j in range(state.BoardSize):
                char = lines[i][j]
                if char == '#':
                    cell_type = ECellType.Empty
                elif char == 'X':
                    cell_type = ECellType.First
                elif char == 'O':
                    cell_type = ECellType.Second
                else:
                    cell_type = ECellType.Empty  # Default for unexpected characters
                state.SetCell(i, j, cell_type)

        return state

def SerializeMoveAsStringV1(move):
    temp_state = GameState()

    row_number = move.row if move.rowType == ERowType.Row else move.column
    start_point = move.column if move.rowType == ERowType.Row else move.row
    end_point = BOARD_SIZE - 1 if move.direction == EMoveDirection.Up else 0

    # Set the starting point to empty (represented as '.')
    if move.rowType == ERowType.Row:
        temp_state.SetCell(row_number, start_point, ECellType.StartPoint)
    else:
        temp_state.SetCell(start_point, row_number, ECellType.StartPoint)

    # Set the end point to the move's cell type
    if move.rowType == ERowType.Row:
        temp_state.SetCell(row_number, end_point, ECellType(move.cellType))
    else:
        temp_state.SetCell(end_point, row_number, ECellType(move.cellType))

    return temp_state.SerializeAsString()

class QuixoEnv:
    def __init__(self, statuses):
        self.statuses = statuses
        self.state = GameState()
        self.currentPlayer = ECellType.First

    def _invert_player(self, player):
        if player == ECellType.First:
            return ECellType.Second
        if player == ECellType.Second:
            return ECellType.First
        
    def _switch_player(self):
        self.currentPlayer = self._invert_player(self.currentPlayer)

    def get_possible_actions(self):
        return list(map(
                SerializeMoveAsStringV1, 
                self.state.GetPossibleMoves(self.currentPlayer)
            ))

    def get_possible_next_states(self):
        moves = self.state.GetPossibleMoves(self.currentPlayer)
        next_states = [deepcopy(self.state).ApplyMove(mv) for mv in self.state.GetPossibleMoves(self.currentPlayer)]
        return [(f"{state.SerializeAsString()}_{self._invert_player(self.currentPlayer).value}", SerializeMoveAsStringV1(mv)) for state, mv in zip(next_states, moves)]

    def get_current_state(self):
        return (self.state.SerializeAsString(), self.currentPlayer)

    def reset(self,):
        self.state = GameState()
        self.currentPlayer = ECellType.First
        return self

    def action(self, move_serialized):
        try:
            move = Move.ParseMoveFromString(move_serialized)
        except:
            return (self.get_current_state(), -2)
        
        self.state.ApplyMove(move)
        self._switch_player()
        
        if self.statuses:
            reward = -self.statuses[(self.state.SerializeAsString(), self.currentPlayer == ECellType.Second)]["reward"]
        else:
            reward = 0
        
        return (self.get_current_state(), reward )
    
    def get_winner(self):
        if not self.state.IsFinish():
            return None
        else:
            if self.state.checkAllInARow(ECellType.First):
                return ECellType.First
            if self.state.checkAllInARow(ECellType.Second):
                return ECellType.Second
    
    @staticmethod
    def load_statuses(data_file="./statuses/status_4x4.log"):
        statuses = dict()
        with open(data_file) as states_file:
            raw_content = states_file.read()
            for state in raw_content.split("=\n"):
                if len(state) < 34:
                    break
                source_state = state[:32]
                source_is_second_move = (state[32] == "1")
                source_status = (state[34])
                source_state_depth = int(state[36:])

                reward_mapping = {
                    "w": 1,
                    "d": 0,
                    "l": -1,
                }

                statuses[(source_state, source_is_second_move)] = {
                    "source_state": source_state,
                    "source_is_second_move": source_is_second_move,
                    "source_status": source_status,
                    "source_state_depth": source_state_depth,
                    "reward": reward_mapping[source_status]
                }
        return statuses

class RandomPlayer:
    def get_action(self, env):
        moves = env.get_possible_actions()
        move = np.random.choice(moves)
        return move

class OptimalPlayer:
    def __init__(self,):
        pass

from itertools import cycle

def play_game(env, player1, player2, max_steps=100):
    env.reset()
    players = [player1, player2]
    step = 0
    states = []
    actions = []
    for current_player in cycle(players):
        states.append(env.get_current_state())
        step += 1
        if step > max_steps:
            break

        if env.get_winner() is not None:
            return env.get_winner()
        
        try:
            action = current_player.get_action(env)
            actions.append(action)
            env.action(action)
        except:
            print(states)
            print(actions)
            raise


