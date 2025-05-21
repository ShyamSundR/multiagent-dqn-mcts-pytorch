import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyValueNet(nn.Module):
    def __init__(self, input_dim=42, hidden_dim=128, num_actions=7):
        """
        Initialize the policy--value network.
        Parameters:
        input_dim: Dimension of the flattened board (default: 42).
        hidden_dim: Number of hidden units.
        num_actions: Number of possible moves (columns 0--6, default: 7).
        Outputs:
        - policy_logits: Tensor of shape [batch_size, num_actions].
        - value: Tensor of shape [batch_size, 1] with values in [-1, 1].
        """
        super(PolicyValueNet, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Policy head
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        
        # Value head
        self.value_head = nn.Linear(hidden_dim, hidden_dim // 2)
        self.value_output = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        """
        Forward pass.
        Parameters:
        x: Tensor of shape [batch_size, input_dim].
        Returns:
        policy_logits: Tensor of shape [batch_size, num_actions].
        value: Tensor of shape [batch_size, 1].
        """
        # Shared representation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Policy head
        policy_logits = self.policy_head(x)
        
        # Value head - output between -1 and 1
        value = F.relu(self.value_head(x))
        value = torch.tanh(self.value_output(value))
        
        return policy_logits, value