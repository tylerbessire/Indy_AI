"""
Sentient AI Agent - The "consciousness" layer that ties everything together
Combines plastic neural network, persistent memory, and proactive behavior
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

from .plastic_brain import PlasticNeuralNetwork, ProactiveBehaviorEngine
from .persistent_memory import PersistentMemorySystem, WorkingMemory


class SentientAIAgent:
    """
    The main AI agent that exhibits "sentient-like" behavior through:
    - Continuous learning (plasticity)
    - Long-term memory (persistence)
    - Proactive behavior (agency)
    - Self-awareness (internal state monitoring)
    """

    def __init__(
        self,
        input_size: int = 128,
        hidden_sizes: List[int] = [256, 512, 256],
        output_size: int = 64,
        learning_rate: float = 0.001,
        memory_path: str = "./memory_db"
    ):
        # Core neural network with plasticity
        self.brain = PlasticNeuralNetwork(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            plasticity_rate=0.01,
            hebbian_rate=0.001
        )

        # Optimizer for gradient-based learning
        self.optimizer = optim.AdamW(
            self.brain.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Memory systems
        self.long_term_memory = PersistentMemorySystem(memory_path=memory_path)
        self.working_memory = WorkingMemory(capacity=7)

        # Proactive behavior
        self.behavior_engine = ProactiveBehaviorEngine(self.brain)

        # Self-awareness metrics
        self.self_state = {
            'awareness_level': 0.0,
            'learning_progress': 0.0,
            'emotional_state': 'neutral',  # Simulated emotional state
            'energy_level': 1.0,
            'curiosity': 0.5,
            'confidence': 0.5
        }

        # Interaction history
        self.interaction_count = 0
        self.last_interaction = None
        self.performance_history = []

        # Learning mode
        self.learning_enabled = True

    def perceive(self, sensory_input: Dict[str, Any]) -> torch.Tensor:
        """
        Process sensory input (like perception in conscious beings)

        Args:
            sensory_input: Dictionary containing various input modalities

        Returns:
            Encoded sensory tensor
        """
        # Convert various input types to tensor
        encoded_input = self._encode_input(sensory_input)

        # Add to working memory with high attention
        self.working_memory.add(sensory_input, attention=0.9)

        # Recall similar past experiences
        if 'query' in sensory_input:
            similar_memories = self.long_term_memory.recall_similar_memories(
                query=sensory_input['query'],
                n_results=3
            )

            if similar_memories:
                # Add recalled memories to working memory
                for memory in similar_memories:
                    self.working_memory.add(memory, attention=memory['similarity'])

        return encoded_input

    def think(self, perception: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Think - process perceptions through the neural network

        Args:
            perception: Encoded sensory input

        Returns:
            Thought output and internal state
        """
        # Forward pass through brain
        with torch.set_grad_enabled(self.learning_enabled):
            thought, activations = self.brain(perception, return_activations=True)

        # Update internal state for proactive behavior
        current_reward = self._calculate_intrinsic_reward(thought)
        self.behavior_engine.update_internal_state(perception, current_reward)

        # Update self-awareness metrics
        self._update_self_state(thought, activations)

        internal_state = {
            'activations': [act.tolist() if isinstance(act, np.ndarray) else act for act in activations],
            'self_state': self.self_state.copy(),
            'working_memory': self.working_memory.get_state(),
            'should_act_proactively': self.behavior_engine.should_act()
        }

        return thought, internal_state

    def act(self, thought: torch.Tensor, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Take action based on thought

        Args:
            thought: Neural network output
            context: Contextual information for the action

        Returns:
            Action dictionary
        """
        # Convert thought to action
        action_vector = thought.detach()

        # Determine action type based on neural output
        action = {
            'type': self._determine_action_type(action_vector),
            'vector': action_vector.cpu().numpy().tolist(),
            'confidence': float(self.self_state['confidence']),
            'proactive': self.behavior_engine.should_act(),
            'timestamp': datetime.now().isoformat()
        }

        # If proactive, add autonomous action
        if action['proactive'] and self.self_state['energy_level'] > 0.3:
            proactive_action = self.behavior_engine.generate_proactive_action()
            action['proactive_component'] = proactive_action.cpu().numpy().tolist()

        return action

    def learn(
        self,
        perception: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        context: str
    ):
        """
        Learn from experience using multiple learning mechanisms

        Args:
            perception: What was perceived
            action: What action was taken
            reward: Reward received
            context: Description of the experience
        """
        if not self.learning_enabled:
            return

        # 1. Gradient-based learning
        predicted_value = self.brain(perception)
        target_value = action  # In this case, we're learning to predict the action

        # Compute loss with reward modulation
        loss = nn.functional.mse_loss(predicted_value, target_value)
        loss = loss * (1.0 + reward)  # Reward-modulated learning

        # Apply gradient plasticity
        self.brain.apply_gradient_plasticity(loss, self.optimizer)

        # 2. Hebbian learning (unsupervised)
        self.brain.apply_hebbian_learning(learning_signal=reward)

        # 3. Store in long-term memory
        importance = abs(reward) * self.self_state['curiosity']

        experience = {
            'perception': perception.cpu().numpy().tolist(),
            'action': action.cpu().numpy().tolist(),
            'reward': reward,
            'self_state': self.self_state.copy()
        }

        self.long_term_memory.store_episodic_memory(
            experience=experience,
            context=context,
            importance=min(importance, 1.0)
        )

        # 4. Store in brain's episode buffer
        self.brain.store_episode(perception, action, reward)

        # Track performance
        self.performance_history.append({
            'reward': reward,
            'loss': float(loss.item()),
            'timestamp': datetime.now().isoformat()
        })

        if len(self.performance_history) > 1000:
            self.performance_history.pop(0)

        # Update learning progress
        recent_rewards = [p['reward'] for p in self.performance_history[-20:]]
        if recent_rewards:
            self.self_state['learning_progress'] = np.mean(recent_rewards)

    def reflect(self) -> Dict[str, Any]:
        """
        Self-reflection - analyze internal state and memories
        This is like metacognition in humans
        """
        reflection = {
            'timestamp': datetime.now().isoformat(),
            'self_state': self.self_state.copy(),
            'memory_stats': self.long_term_memory.get_memory_statistics(),
            'working_memory': self.working_memory.get_state(),
            'recent_performance': self.performance_history[-10:],
            'interaction_count': self.interaction_count,
            'brain_state': self.brain.get_weight_visualization_data(),
            'proactive_state': self.behavior_engine.get_proactive_state()
        }

        # Update awareness level based on memory and performance
        memory_richness = min(reflection['memory_stats']['total_memories'] / 1000, 1.0)
        performance_quality = self.self_state['learning_progress']

        self.self_state['awareness_level'] = (
            0.3 * memory_richness +
            0.3 * performance_quality +
            0.4 * self.self_state['awareness_level']  # Temporal smoothing
        )

        return reflection

    def interact(
        self,
        input_data: Dict[str, Any],
        learn_from_interaction: bool = True
    ) -> Dict[str, Any]:
        """
        Full interaction cycle: perceive -> think -> act -> learn

        Args:
            input_data: Input from environment/user
            learn_from_interaction: Whether to learn from this interaction

        Returns:
            Response dictionary
        """
        self.interaction_count += 1
        self.last_interaction = datetime.now().isoformat()

        # Perceive
        perception = self.perceive(input_data)

        # Think
        thought, internal_state = self.think(perception)

        # Act
        action = self.act(thought, input_data)

        # Learn (if feedback provided)
        if learn_from_interaction and 'reward' in input_data:
            reward = input_data['reward']
            context = input_data.get('context', 'Interaction')
            self.learn(perception, thought, reward, context)

        # Respond
        response = {
            'action': action,
            'internal_state': internal_state,
            'reflection': self.reflect() if self.interaction_count % 10 == 0 else None,
            'message': self._generate_response_message(action, internal_state)
        }

        return response

    def save_state(self, filepath: str):
        """Save complete agent state"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'brain': self.brain.get_state_dict_serializable(),
            'self_state': self.self_state,
            'interaction_count': self.interaction_count,
            'performance_history': self.performance_history[-100:]
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        # Also save memory (happens automatically with ChromaDB)
        print(f"Agent state saved to {filepath}")

    def load_state(self, filepath: str):
        """Load complete agent state"""
        with open(filepath, 'r') as f:
            state = json.load(f)

        self.brain.load_from_serializable(state['brain'])
        self.self_state = state['self_state']
        self.interaction_count = state['interaction_count']
        self.performance_history = state.get('performance_history', [])

        print(f"Agent state loaded from {filepath}")

    def _encode_input(self, input_data: Dict[str, Any]) -> torch.Tensor:
        """Encode various input types into tensor"""
        # Simple encoding - in practice, you'd want more sophisticated encoding
        if 'vector' in input_data:
            return torch.tensor(input_data['vector'], dtype=torch.float32)
        elif 'text' in input_data:
            # Simple text encoding (in practice, use embeddings)
            text = input_data['text']
            encoding = np.random.randn(self.brain.input_size) * 0.1
            return torch.tensor(encoding, dtype=torch.float32)
        else:
            # Random encoding
            return torch.randn(self.brain.input_size)

    def _calculate_intrinsic_reward(self, thought: torch.Tensor) -> float:
        """Calculate intrinsic reward (curiosity-driven)"""
        # Novelty-based reward
        novelty = torch.std(thought).item()
        return novelty

    def _update_self_state(self, thought: torch.Tensor, activations: List):
        """Update self-awareness state"""
        # Update confidence based on output consistency
        output_entropy = -torch.sum(
            torch.softmax(thought, dim=-1) * torch.log_softmax(thought, dim=-1)
        ).item()
        self.self_state['confidence'] = 1.0 - min(output_entropy / 10.0, 1.0)

        # Update curiosity based on activation variance
        if activations:
            activation_variance = np.mean([np.var(act) for act in activations if isinstance(act, np.ndarray)])
            self.self_state['curiosity'] = min(activation_variance, 1.0)

        # Simulated emotional state based on recent performance
        if self.performance_history:
            recent_reward = self.performance_history[-1]['reward']
            if recent_reward > 0.5:
                self.self_state['emotional_state'] = 'positive'
            elif recent_reward < -0.5:
                self.self_state['emotional_state'] = 'negative'
            else:
                self.self_state['emotional_state'] = 'neutral'

        # Energy level decreases with use, increases with rest
        self.self_state['energy_level'] = max(0.3, self.self_state['energy_level'] - 0.01)

    def _determine_action_type(self, action_vector: torch.Tensor) -> str:
        """Determine action type from vector"""
        # Simple classification based on vector properties
        magnitude = torch.norm(action_vector).item()

        if magnitude > 2.0:
            return 'strong_action'
        elif magnitude > 1.0:
            return 'moderate_action'
        else:
            return 'subtle_action'

    def _generate_response_message(self, action: Dict, internal_state: Dict) -> str:
        """Generate human-readable response message"""
        messages = {
            'positive': [
                "I'm learning and adapting!",
                "This is interesting...",
                "I understand this pattern."
            ],
            'neutral': [
                "Processing information...",
                "Analyzing the situation...",
                "Considering options..."
            ],
            'negative': [
                "This is challenging.",
                "I need more information.",
                "Uncertain about this..."
            ]
        }

        emotional_state = self.self_state['emotional_state']
        message_list = messages.get(emotional_state, messages['neutral'])

        import random
        base_message = random.choice(message_list)

        if action['proactive']:
            base_message += " (Acting proactively)"

        return base_message
