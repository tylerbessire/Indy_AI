"""
Plastic Brain - Neural Network with Real-time Weight Plasticity
Implements Hebbian learning and adaptive weight updates for continuous learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime


class PlasticNeuralNetwork(nn.Module):
    """
    A neural network with plastic (adaptable) weights that update in real-time.
    Combines traditional backpropagation with Hebbian learning for neuroplasticity.
    """

    def __init__(
        self,
        input_size: int = 128,
        hidden_sizes: List[int] = [256, 512, 256],
        output_size: int = 64,
        plasticity_rate: float = 0.01,
        hebbian_rate: float = 0.001
    ):
        super(PlasticNeuralNetwork, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.plasticity_rate = plasticity_rate
        self.hebbian_rate = hebbian_rate

        # Build network layers
        self.layers = nn.ModuleList()
        prev_size = input_size

        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size

        # Output layer
        self.output_layer = nn.Linear(prev_size, output_size)

        # Track activations for Hebbian learning
        self.activations = []
        self.weight_history = []

        # Attention mechanism for proactive behavior
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_sizes[-1],
            num_heads=8,
            dropout=0.1
        )

        # Episodic memory buffer
        self.episode_buffer = []
        self.max_episodes = 1000

    def forward(self, x: torch.Tensor, return_activations: bool = False) -> torch.Tensor:
        """Forward pass with activation tracking"""
        self.activations = []

        # Input layer
        h = x
        self.activations.append(h.detach().cpu().numpy())

        # Hidden layers
        for i, layer in enumerate(self.layers):
            h = layer(h)
            h = F.gelu(h)  # Using GELU for smoother gradients
            h = F.dropout(h, p=0.1, training=self.training)
            self.activations.append(h.detach().cpu().numpy())

        # Output layer
        output = self.output_layer(h)
        self.activations.append(output.detach().cpu().numpy())

        if return_activations:
            return output, self.activations
        return output

    def apply_hebbian_learning(self, learning_signal: float = 1.0):
        """
        Apply Hebbian learning rule: "Neurons that fire together, wire together"
        Updates weights based on correlation between pre and post-synaptic activity
        """
        if len(self.activations) < 2:
            return

        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                if i + 1 < len(self.activations):
                    pre_activation = torch.tensor(self.activations[i])
                    post_activation = torch.tensor(self.activations[i + 1])

                    # Hebbian update: ΔW = η * pre * post
                    # Add batch dimension handling
                    if len(pre_activation.shape) > 1:
                        pre_activation = pre_activation.mean(dim=0)
                    if len(post_activation.shape) > 1:
                        post_activation = post_activation.mean(dim=0)

                    # Compute weight update
                    weight_update = torch.outer(
                        post_activation[:layer.weight.shape[0]],
                        pre_activation[:layer.weight.shape[1]]
                    )

                    # Apply plasticity with learning signal
                    layer.weight += self.hebbian_rate * learning_signal * weight_update

                    # Normalize to prevent explosion
                    layer.weight.data = F.normalize(layer.weight.data, dim=1)

    def apply_gradient_plasticity(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        """Apply gradient-based plasticity (traditional backprop)"""
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        optimizer.step()

        # Store weight snapshots for visualization
        self._capture_weight_snapshot()

    def _capture_weight_snapshot(self):
        """Capture current weights for visualization"""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'weights': []
        }

        for i, layer in enumerate(self.layers):
            weight_stats = {
                'layer': i,
                'mean': float(layer.weight.mean().item()),
                'std': float(layer.weight.std().item()),
                'min': float(layer.weight.min().item()),
                'max': float(layer.weight.max().item()),
            }
            snapshot['weights'].append(weight_stats)

        self.weight_history.append(snapshot)

        # Keep only recent history
        if len(self.weight_history) > 100:
            self.weight_history.pop(0)

    def get_weight_visualization_data(self) -> Dict:
        """Get weight data for frontend visualization"""
        return {
            'history': self.weight_history[-50:],  # Last 50 snapshots
            'current_activations': [
                arr.tolist() if isinstance(arr, np.ndarray) else arr
                for arr in self.activations
            ]
        }

    def store_episode(self, state: torch.Tensor, action: torch.Tensor, reward: float):
        """Store experience in episodic memory"""
        episode = {
            'state': state.detach().cpu().numpy().tolist(),
            'action': action.detach().cpu().numpy().tolist(),
            'reward': reward,
            'timestamp': datetime.now().isoformat()
        }

        self.episode_buffer.append(episode)

        # Maintain buffer size
        if len(self.episode_buffer) > self.max_episodes:
            self.episode_buffer.pop(0)

    def get_state_dict_serializable(self) -> Dict:
        """Get serializable state for persistence"""
        return {
            'model_state': {k: v.cpu().numpy().tolist() for k, v in self.state_dict().items()},
            'episode_buffer': self.episode_buffer[-100:],  # Save last 100 episodes
            'weight_history': self.weight_history[-50:],
            'config': {
                'input_size': self.input_size,
                'output_size': self.output_size,
                'plasticity_rate': self.plasticity_rate,
                'hebbian_rate': self.hebbian_rate
            }
        }

    def load_from_serializable(self, data: Dict):
        """Load state from serialized data"""
        # Load model weights
        state_dict = {
            k: torch.tensor(v) for k, v in data['model_state'].items()
        }
        self.load_state_dict(state_dict)

        # Restore episode buffer and history
        self.episode_buffer = data.get('episode_buffer', [])
        self.weight_history = data.get('weight_history', [])


class ProactiveBehaviorEngine:
    """
    Engine that makes the AI proactive - it can initiate actions
    based on its internal state and learned patterns
    """

    def __init__(self, brain: PlasticNeuralNetwork):
        self.brain = brain
        self.curiosity_threshold = 0.5
        self.action_history = []
        self.internal_state = torch.randn(brain.input_size)

    def update_internal_state(self, external_input: torch.Tensor, reward: float):
        """Update internal state based on external input and reward"""
        # Blend external input with current internal state
        alpha = 0.3  # Blending factor
        self.internal_state = alpha * external_input + (1 - alpha) * self.internal_state

        # Add reward-modulated noise for exploration
        if reward < 0:
            # Increase exploration when performing poorly
            self.internal_state += torch.randn_like(self.internal_state) * 0.1

    def should_act(self) -> bool:
        """Decide if the AI should proactively take an action"""
        # Calculate "curiosity" based on internal state entropy
        state_entropy = -torch.sum(
            F.softmax(self.internal_state, dim=0) *
            F.log_softmax(self.internal_state, dim=0)
        )

        # Act if curiosity exceeds threshold
        return state_entropy > self.curiosity_threshold

    def generate_proactive_action(self) -> torch.Tensor:
        """Generate an action based on current internal state"""
        with torch.no_grad():
            action = self.brain(self.internal_state.unsqueeze(0))

        self.action_history.append({
            'action': action.cpu().numpy().tolist(),
            'timestamp': datetime.now().isoformat(),
            'internal_state_snapshot': self.internal_state.cpu().numpy().tolist()
        })

        return action

    def get_proactive_state(self) -> Dict:
        """Get current proactive state for visualization"""
        return {
            'should_act': self.should_act(),
            'internal_state': self.internal_state.cpu().numpy().tolist(),
            'recent_actions': self.action_history[-10:]
        }
