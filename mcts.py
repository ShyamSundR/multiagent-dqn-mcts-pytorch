import numpy as np
import torch
import math

class MCTSNode:
    def __init__(self, prior=0):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None
        self.reward = 0
        self.is_terminal = False
    
    def expanded(self):
        return len(self.children) > 0
    
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MCTS:
    def __init__(self, network, c_puct=1.0):
        """
        Initialize MCTS with the policy-value network.
        
        Parameters:
        network: The neural network for policy and value predictions.
        c_puct: Exploration constant for UCT.
        """
        self.network = network
        self.c_puct = c_puct
        self.root = MCTSNode()
    
    def _ucb_score(self, parent, child):
        """Calculate UCB score for a child node."""
        # Exploration term from UCT
        exploration = self.c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        
        # Exploitation term (Q-value)
        exploitation = child.value()
        
        return exploitation + exploration
    
    def _select_child(self, node):
        """Select the child with the highest UCB score."""
        best_score = -float('inf')
        best_action = -1
        best_child = None
        
        for action, child in node.children.items():
            score = self._ucb_score(node, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def _expand(self, node, state, valid_actions, policy_probs):
        """Expand the node with children according to valid actions and policy."""
        node.state = state.copy()
        
        for action in valid_actions:
            node.children[action] = MCTSNode(prior=policy_probs[action])
    
    def _backpropagate(self, search_path, value):
        """Update statistics of nodes along the search path."""
        current_player = 1
        
        for node in reversed(search_path):
            node.value_sum += value * current_player
            node.visit_count += 1
            value = -value  # Flip sign for alternating players
            current_player = -current_player
    
    def _run_simulation(self, env):
        """Run a single MCTS simulation."""
        # Create a copy of the environment to avoid modifying the original
        env_copy = env.__class__()
        env_copy.board = env.board.copy()
        env_copy.current_player = env.current_player
        env_copy.moves_made = env.moves_made
        
        search_path = [self.root]
        node = self.root
        done = False
        
        # Selection phase: find leaf node
        while node.expanded() and not done:
            action, node = self._select_child(node)
            state, reward, done = env_copy.step(action)
            search_path.append(node)
            
            if done:
                node.is_terminal = True
                node.reward = reward
        
        # If node is not terminal and not expanded, evaluate and expand
        if not node.expanded() and not done:
            # Get valid actions
            valid_actions = env_copy.get_valid_actions()
            
            # Prepare state for network
            state_tensor = torch.FloatTensor(env_copy.board.flatten()).unsqueeze(0)
            
            # Get policy and value from network
            with torch.no_grad():
                policy_logits, value = self.network(state_tensor)
            
            # Convert policy logits to probabilities
            policy_probs = torch.softmax(policy_logits, dim=1).squeeze(0).numpy()
            
            # Mask invalid actions
            masked_policy_probs = np.zeros_like(policy_probs)
            for action in valid_actions:
                masked_policy_probs[action] = policy_probs[action]
            
            # Normalize probabilities
            if sum(masked_policy_probs) > 0:
                masked_policy_probs /= sum(masked_policy_probs)
            else:
                # If all actions are masked, use uniform distribution
                for action in valid_actions:
                    masked_policy_probs[action] = 1.0 / len(valid_actions)
            
            # Expand the node
            self._expand(node, env_copy.board, valid_actions, masked_policy_probs)
            
            # Use network's value prediction
            value_prediction = value.item()
            
            # Backpropagate
            self._backpropagate(search_path, value_prediction)
        elif node.is_terminal:
            # For terminal nodes, use the actual reward
            self._backpropagate(search_path, node.reward)
    
    def search(self, env, num_simulations):
        """
        Run MCTS simulations and return the most visited action.
        
        Parameters:
        env: The game environment.
        num_simulations: Number of MCTS simulations to run.
        
        Returns:
        action: The chosen action (column 0-6).
        pi: Normalized visit counts for all actions.
        """
        # Initialize root node if not already
        if not self.root.expanded():
            # Get valid actions
            valid_actions = env.get_valid_actions()
            
            # Prepare state for network
            state_tensor = torch.FloatTensor(env.board.flatten()).unsqueeze(0)
            
            # Get policy and value from network
            with torch.no_grad():
                policy_logits, _ = self.network(state_tensor)
            
            # Convert policy logits to probabilities
            policy_probs = torch.softmax(policy_logits, dim=1).squeeze(0).numpy()
            
            # Mask invalid actions
            masked_policy_probs = np.zeros_like(policy_probs)
            for action in valid_actions:
                masked_policy_probs[action] = policy_probs[action]
            
            # Normalize probabilities
            if sum(masked_policy_probs) > 0:
                masked_policy_probs /= sum(masked_policy_probs)
            else:
                # If all actions are masked, use uniform distribution
                for action in valid_actions:
                    masked_policy_probs[action] = 1.0 / len(valid_actions)
            
            # Expand the root
            self._expand(self.root, env.board, valid_actions, masked_policy_probs)
        
        # Run simulations
        for _ in range(num_simulations):
            self._run_simulation(env)
        
        # Calculate pi (normalized visit counts)
        pi = np.zeros(7)  # 7 possible actions (columns)
        for action, child in self.root.children.items():
            pi[action] = child.visit_count
        
        # Normalize
        if sum(pi) > 0:
            pi /= sum(pi)
        
        # Choose the most visited action
        action = np.argmax(pi)
        
        return action, pi

def mcts_search(env, network, num_simulations):
    """
    Run MCTS and return the chosen action.
    
    Parameters:
    env: The game environment.
    network: The policy-value network.
    num_simulations: Number of MCTS simulations to run.
    
    Returns:
    action: The chosen action (column 0-6).
    """
    mcts = MCTS(network)
    action, _ = mcts.search(env, num_simulations)
    return action