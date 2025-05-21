import numpy as np

class Connect4Env:
    def __init__(self):
        """
        Initialize the Connect 4 environment.
        - The board is a grid with 6 rows and 7 columns (total 42 cells).
        - Cell values: 0 (empty), 1 (Player 1), -1 (Player 2).
        - Gravity: a move selects a column (0--6) and the disc occupies
        the lowest available cell.
        - Winning condition: a contiguous line of exactly 4 discs.
        - The game starts with an empty board; Player 1 moves first.
        - Maximum moves: 42.
        """
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1
        self.moves_made = 0
        
    def reset(self):
        """
        Reset the environment to the initial state.
        Returns:
        board: A flattened NumPy array of shape (42,) representing
        an empty board.
        current_player: An integer (1 for Player 1) indicating
        the starting player.
        """
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_player = 1
        self.moves_made = 0
        return self.board.flatten(), self.current_player
    
    def step(self, action):
        """
        Take an action in the environment.
        Parameters:
        action: An integer (0 to 6) representing the column in which to
        drop a disc.
        Returns:
        board: A flattened NumPy array of shape (42,) representing the
        updated board.
        reward: A float, where:
        +1 if the current player's move results in a win,
        0 otherwise.
        done: A Boolean indicating whether the game has ended (win or draw).
        """
        # Check if action is valid (column is not full)
        if not self._is_valid_action(action):
            # Return current state without changes if invalid move
            return self.board.flatten(), 0, False
        
        # Place the disc in the chosen column
        row = self._get_next_free_row(action)
        self.board[row, action] = self.current_player
        self.moves_made += 1
        
        # Check for win
        if self._check_win(row, action):
            return self.board.flatten(), 1, True
        
        # Check for draw
        if self.moves_made == self.rows * self.cols:
            return self.board.flatten(), 0, True
        
        # Switch player
        self.current_player = -self.current_player
        
        return self.board.flatten(), 0, False
    
    def _is_valid_action(self, action):
        """Check if a column has space for another disc."""
        return 0 <= action < self.cols and self.board[0, action] == 0
    
    def _get_next_free_row(self, col):
        """Get the lowest empty row in the given column."""
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, col] == 0:
                return row
        return -1  # This should never happen if _is_valid_action is called first
    
    def _check_win(self, row, col):
        """Check if the last move resulted in a win."""
        player = self.board[row, col]
        
        # Check horizontal
        for c in range(max(0, col - 3), min(col + 1, self.cols - 3)):
            if all(self.board[row, c + i] == player for i in range(4)):
                return True
        
        # Check vertical
        for r in range(max(0, row - 3), min(row + 1, self.rows - 3)):
            if all(self.board[r + i, col] == player for i in range(4)):
                return True
        
        # Check diagonal /
        for i in range(-3, 1):
            r, c = row - i, col + i
            if (0 <= r < self.rows - 3 and 0 <= c < self.cols - 3 and
                all(self.board[r + j, c + j] == player for j in range(4))):
                return True
        
        # Check diagonal \
        for i in range(-3, 1):
            r, c = row - i, col - i
            if (0 <= r < self.rows - 3 and 3 <= c < self.cols and
                all(self.board[r + j, c - j] == player for j in range(4))):
                return True
        
        return False
    
    def get_valid_actions(self):
        """Return a list of valid actions (columns that are not full)."""
        return [col for col in range(self.cols) if self._is_valid_action(col)]
    
    def get_current_player(self):
        """Return the current player."""
        return self.current_player