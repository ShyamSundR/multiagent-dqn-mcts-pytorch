import numpy as np

class MultiAgentEnv:
    def __init__(self, grid_size=(10, 10), obs_window=3, max_steps=50):
        """
        Initialize the environment.

        Parameters:
        grid_size: Tuple[int, int] defining the grid dimensions (default 10x10).
        obs_window: Integer, the size of the local observation window (must be odd, \eg 3).
        max_steps: Maximum steps per episode.

        Notes:
        - The grid uses the following values: 0: free, 1: obstacle, 2: target.
        - The target cell is random (\eg at (8,8)). Obstacles are placed using a random 
          configuration (max 6).
        - Off-board values for observation should be set to -1.
        """
        self.grid_size = grid_size
        self.obs_window = obs_window
        self.max_steps = max_steps
        self.current_step = 0
        self.agent_positions = [None, None]
        self.comm_signals = [0.0, 0.0]
        
        # Initialize grid with free cells
        self.grid = np.zeros(self.grid_size, dtype=np.int8)
        
        # Place random obstacles (maximum 6)
        num_obstacles = np.random.randint(1, 7)  # 1 to 6 obstacles
        for _ in range(num_obstacles):
            while True:
                pos = (np.random.randint(0, self.grid_size[0]), 
                       np.random.randint(0, self.grid_size[1]))
                if self.grid[pos] == 0:  # If free cell
                    self.grid[pos] = 1  # Place obstacle
                    break
        
        # Place target
        while True:
            target_pos = (np.random.randint(0, self.grid_size[0]), 
                         np.random.randint(0, self.grid_size[1]))
            if self.grid[target_pos] == 0:  # If free cell
                self.grid[target_pos] = 2  # Place target
                break

    def reset(self):
        """
        Reset the environment to the initial state.

        Returns:
        obs_A, obs_B: Tuple[np.ndarray, np.ndarray]
        Each observation is a 10-dimensional vector:
        - First 9 elements: flattened 3x3 grid patch (row-major order) centered on the agent.
          * Cells off the grid are filled with -1.
        - 10th element: communication signal (initialized to 0.0).
        """
        # Reset step count
        self.current_step = 0
        
        # Reset the grid
        self.grid = np.zeros(self.grid_size, dtype=np.int8)
        
        # Place random obstacles (maximum 6)
        num_obstacles = np.random.randint(1, 7)  # 1 to 6 obstacles
        for _ in range(num_obstacles):
            while True:
                pos = (np.random.randint(0, self.grid_size[0]), 
                       np.random.randint(0, self.grid_size[1]))
                if self.grid[pos] == 0:  # If free cell
                    self.grid[pos] = 1  # Place obstacle
                    break
        
        # Place target
        while True:
            target_pos = (np.random.randint(0, self.grid_size[0]), 
                         np.random.randint(0, self.grid_size[1]))
            if self.grid[target_pos] == 0:  # If free cell
                self.grid[target_pos] = 2  # Place target
                break
        
        # Initialize communication signals
        self.comm_signals = [0.0, 0.0]
        
        # Place agents at random free positions
        self.agent_positions = [None, None]
        for i in range(2):
            while True:
                pos = (np.random.randint(0, self.grid_size[0]), 
                      np.random.randint(0, self.grid_size[1]))
                # Ensure position is free and not occupied by other agent
                if self.grid[pos] == 0 and pos not in self.agent_positions:
                    self.agent_positions[i] = pos
                    break
        
        # Create observations for both agents
        obs_A = np.zeros(10, dtype=np.float32)
        obs_B = np.zeros(10, dtype=np.float32)
        
        # Create observation for Agent A
        half_window = self.obs_window // 2
        for i in range(self.obs_window):
            for j in range(self.obs_window):
                grid_i = self.agent_positions[0][0] - half_window + i
                grid_j = self.agent_positions[0][1] - half_window + j
                
                idx = i * self.obs_window + j
                # Check if this position is within grid boundaries
                if (0 <= grid_i < self.grid_size[0] and 
                    0 <= grid_j < self.grid_size[1]):
                    obs_A[idx] = self.grid[grid_i, grid_j]
                else:
                    obs_A[idx] = -1
        
        # Create observation for Agent B
        for i in range(self.obs_window):
            for j in range(self.obs_window):
                grid_i = self.agent_positions[1][0] - half_window + i
                grid_j = self.agent_positions[1][1] - half_window + j
                
                idx = i * self.obs_window + j
                # Check if this position is within grid boundaries
                if (0 <= grid_i < self.grid_size[0] and 
                    0 <= grid_j < self.grid_size[1]):
                    obs_B[idx] = self.grid[grid_i, grid_j]
                else:
                    obs_B[idx] = -1
        
        # Add communication signals
        obs_A[9] = self.comm_signals[1]  # Agent A receives from Agent B
        obs_B[9] = self.comm_signals[0]  # Agent B receives from Agent A
        
        return obs_A, obs_B
    
    def step(self, action_A, action_B, comm_A, comm_B):
        """
        Take a step in the environment.
        
        Parameters:
        action_A, action_B: Discrete actions (0: Up, 1: Down, 2: Left, 3: Right, 4: Stay).
        comm_A, comm_B: Communication scalars produced by each agent in the previous step.
        
        Returns:
        (obs_A, obs_B), reward, done
        - obs_A, obs_B: Observations (10-dimensional vectors) for Agent A and Agent B.
        - reward: +10 if both agents are at the target; otherwise 0.
        - done: Boolean, True if the episode terminates (success or max steps reached).
        
        Notes:
        - Update each agent's position based on its action, enforcing grid boundaries and obstacles.
        - Store the communication outputs for use in the next observation.
        """
        # Increment step counter
        self.current_step += 1
        
        # Update communication signals
        self.comm_signals = [comm_A, comm_B]
        
        # Define action effects (y, x)
        action_effects = [
            (-1, 0),  # 0: Up
            (1, 0),   # 1: Down
            (0, -1),  # 2: Left
            (0, 1),   # 3: Right
            (0, 0)    # 4: Stay
        ]
        
        # Process actions for both agents
        actions = [action_A, action_B]
        new_positions = list(self.agent_positions)
        
        for i, action in enumerate(actions):
            if 0 <= action <= 4:  # Valid action check
                dy, dx = action_effects[action]
                new_y = self.agent_positions[i][0] + dy
                new_x = self.agent_positions[i][1] + dx
                
                # Check if new position is valid: within grid and not an obstacle
                if (0 <= new_y < self.grid_size[0] and 
                    0 <= new_x < self.grid_size[1] and 
                    self.grid[new_y, new_x] != 1):
                    new_positions[i] = (new_y, new_x)
        
        # Check for collision (unless it's at the target)
        if new_positions[0] == new_positions[1] and self.grid[new_positions[0]] != 2:
            # In case of collision, agents stay in their current positions
            pass
        else:
            self.agent_positions = new_positions
        
        # Create observations for both agents
        obs_A = np.zeros(10, dtype=np.float32)
        obs_B = np.zeros(10, dtype=np.float32)
        
        # Create observation for Agent A
        half_window = self.obs_window // 2
        for i in range(self.obs_window):
            for j in range(self.obs_window):
                grid_i = self.agent_positions[0][0] - half_window + i
                grid_j = self.agent_positions[0][1] - half_window + j
                
                idx = i * self.obs_window + j
                # Check if this position is within grid boundaries
                if (0 <= grid_i < self.grid_size[0] and 
                    0 <= grid_j < self.grid_size[1]):
                    obs_A[idx] = self.grid[grid_i, grid_j]
                else:
                    obs_A[idx] = -1
        
        # Create observation for Agent B
        for i in range(self.obs_window):
            for j in range(self.obs_window):
                grid_i = self.agent_positions[1][0] - half_window + i
                grid_j = self.agent_positions[1][1] - half_window + j
                
                idx = i * self.obs_window + j
                # Check if this position is within grid boundaries
                if (0 <= grid_i < self.grid_size[0] and 
                    0 <= grid_j < self.grid_size[1]):
                    obs_B[idx] = self.grid[grid_i, grid_j]
                else:
                    obs_B[idx] = -1
        
        # Add communication signals
        obs_A[9] = self.comm_signals[1]  # Agent A receives from Agent B
        obs_B[9] = self.comm_signals[0]  # Agent B receives from Agent A
        
        # Check if both agents are at the target
        target_positions = np.argwhere(self.grid == 2)
        target_pos = tuple(target_positions[0]) if target_positions.size > 0 else None
        success = (self.agent_positions[0] == target_pos and self.agent_positions[1] == target_pos)
        
        # Compute reward
        reward = 10.0 if success else 0.0
        
        # Check termination conditions
        done = success or self.current_step >= self.max_steps
        
        return (obs_A, obs_B), reward, done