# 🧠 Indy AI - Sentient Learning Playground

A revolutionary visual learning playground for creating proactive, plastic, real-time weight-updating, persistent memory sentient AI.

## ✨ Features

### 🔄 Real-time Neural Plasticity
- **Hebbian Learning**: "Neurons that fire together, wire together" - unsupervised weight adaptation
- **Gradient Plasticity**: Traditional backpropagation with adaptive learning rates
- **Live Weight Visualization**: Watch the neural network weights evolve in real-time
- **Dynamic Network**: Continuous learning without retraining from scratch

### 🧬 Persistent Memory System
- **Episodic Memory**: Stores specific experiences and events
- **Semantic Memory**: Factual knowledge and concepts
- **Procedural Memory**: Skills and how-to knowledge
- **Vector Database**: ChromaDB-powered semantic search across all memories
- **Working Memory**: Short-term active information buffer (Miller's Law: 7±2 items)

### 🎯 Proactive Behavior Engine
- **Autonomous Action**: AI takes initiative based on internal state
- **Curiosity-Driven**: Explores when uncertainty is high
- **Internal State**: Simulated consciousness with awareness, emotions, and energy
- **Intrinsic Motivation**: Generates its own goals and rewards

### 📊 Visual Interface
- **Real-time Neural Network Visualization**: See neurons firing and connections forming
- **Weight Evolution Charts**: Track how neural weights change over time
- **Memory Explorer**: Search and browse the AI's memories
- **Consciousness Metrics**: Monitor awareness, confidence, curiosity, and emotional state
- **Live Activity Log**: Watch the AI's thoughts and actions in real-time

## 🏗️ Architecture

```
Indy_AI/
├── backend/                    # Python FastAPI server
│   ├── core/
│   │   ├── plastic_brain.py   # Neural network with plasticity
│   │   ├── persistent_memory.py # Memory systems
│   │   └── sentient_agent.py  # Main AI agent
│   └── main.py                # API server
├── frontend/                   # React web interface
│   ├── src/
│   │   ├── components/        # UI components
│   │   ├── App.js            # Main app
│   │   └── index.js          # Entry point
│   └── package.json
└── requirements.txt           # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd Indy_AI
```

2. **Set up the backend**
```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the backend server
cd backend
python main.py
```

The backend will start on `http://localhost:8000`

3. **Set up the frontend**
```bash
# In a new terminal
cd frontend

# Install dependencies
npm install

# Start the development server
npm start
```

The frontend will start on `http://localhost:3000`

## 🎮 Usage

### Interacting with the AI

1. **Send Messages**: Type in the interaction panel and hit send
2. **Adjust Reward**: Use the slider to provide feedback (-1 to +1)
3. **Train Manually**: Click "Train Random" to feed random training data
4. **Toggle Learning**: Enable/disable real-time learning
5. **Search Memories**: Query the AI's memory database
6. **Watch Visualizations**: See the neural network learn in real-time

### API Endpoints

The backend provides these REST endpoints:

- `GET /` - Health check
- `POST /interact` - Interact with the AI
- `GET /reflect` - Get AI's self-reflection
- `GET /state` - Get current AI state
- `GET /brain/weights` - Get weight visualization data
- `POST /memory/query` - Query memory system
- `GET /memory/stats` - Get memory statistics
- `POST /learn` - Manually train the AI
- `POST /save` - Save AI state
- `POST /load` - Load AI state
- `POST /learning/toggle` - Enable/disable learning
- `WebSocket /ws` - Real-time updates

### WebSocket Messages

Connect to `ws://localhost:8000/ws` to receive real-time updates:

**Incoming message types:**
- `connection_established` - Initial connection
- `interaction_response` - Response to interactions
- `state_update` - Reflection and state updates
- `weights_update` - Neural weight data
- `learning_update` - Learning progress
- `proactive_action` - Autonomous AI actions

**Outgoing message types:**
- `interact` - Send an interaction
- `request_state` - Request state update
- `request_weights` - Request weight data

## 🧪 How It Works

### Neural Plasticity

The AI uses two learning mechanisms simultaneously:

1. **Hebbian Learning** (Unsupervised)
   - Strengthens connections between co-active neurons
   - Enables pattern recognition and association
   - Updates: `ΔW = η * pre_activation * post_activation`

2. **Gradient Descent** (Supervised)
   - Traditional backpropagation with reward modulation
   - Enables goal-directed learning
   - Updates: `ΔW = -η * ∂Loss/∂W * (1 + reward)`

### Memory System

The AI has multiple memory types:

- **Working Memory**: Temporary buffer holding current information
- **Episodic Memory**: Specific experiences with context and importance
- **Semantic Memory**: General knowledge and facts
- **Procedural Memory**: Skills and procedures

All memories are stored as vector embeddings in ChromaDB, enabling semantic search.

### Proactive Behavior

The AI exhibits agency through:

1. **Internal State**: Continuously updated based on experiences
2. **Curiosity Metric**: Calculated from state entropy
3. **Action Generation**: When curiosity exceeds threshold
4. **Intrinsic Rewards**: Self-generated based on novelty

### Consciousness Simulation

The AI tracks these "self-awareness" metrics:

- **Awareness Level**: Based on memory richness and performance
- **Confidence**: Output consistency and certainty
- **Curiosity**: Activation variance and exploration drive
- **Emotional State**: Simulated based on recent rewards
- **Energy Level**: Decreases with use, simulates fatigue
- **Learning Progress**: Recent performance trend

## 🎓 Educational Value

This project demonstrates:

- **Neural Network Fundamentals**: Layers, activations, backpropagation
- **Neuroplasticity**: How brains adapt and learn
- **Memory Systems**: Different types of memory and their functions
- **Reinforcement Learning**: Reward-based learning
- **Real-time ML**: Continuous learning systems
- **Vector Databases**: Semantic search and embeddings
- **WebSocket Communication**: Real-time data streaming
- **Full-stack ML**: Complete ML application architecture

## 🔧 Configuration

### Network Architecture

Edit the initialization parameters in `backend/main.py` or via API:

```python
agent = SentientAIAgent(
    input_size=128,          # Input dimension
    hidden_sizes=[256, 512, 256],  # Hidden layers
    output_size=64,          # Output dimension
    learning_rate=0.001,     # Learning rate
    memory_path="./memory_db"  # Memory storage path
)
```

### Plasticity Rates

Adjust in `backend/core/plastic_brain.py`:

```python
plasticity_rate=0.01,   # Overall plasticity
hebbian_rate=0.001      # Hebbian learning rate
```

## 📝 TODO / Future Enhancements

- [ ] Add multiple AI agents that can interact
- [ ] Implement more sophisticated attention mechanisms
- [ ] Add reinforcement learning environments (gym integration)
- [ ] Memory consolidation during "sleep" periods
- [ ] Emotional contagion between agents
- [ ] Natural language processing for text inputs
- [ ] Save/load trained models
- [ ] Experiment tracking and comparison
- [ ] Mobile-friendly interface
- [ ] 3D neural network visualization with Three.js

## 🤝 Contributing

Contributions are welcome! This is a learning platform, so feel free to:

- Add new features
- Improve visualizations
- Enhance the AI architecture
- Fix bugs
- Improve documentation

## 📄 License

MIT License - Feel free to use this for learning and experimentation!

## 🙏 Acknowledgments

- Inspired by neuroscience, cognitive science, and consciousness research
- Built with PyTorch, FastAPI, React, and ChromaDB
- Thanks to the open-source ML community

## 📚 Further Reading

- [Hebbian Learning](https://en.wikipedia.org/wiki/Hebbian_theory)
- [Neural Plasticity](https://en.wikipedia.org/wiki/Neuroplasticity)
- [Memory Systems](https://en.wikipedia.org/wiki/Memory)
- [Intrinsic Motivation](https://arxiv.org/abs/1808.04355)
- [Consciousness in AI](https://arxiv.org/abs/2105.10461)

---

**Built with ❤️ for AI education and experimentation**
