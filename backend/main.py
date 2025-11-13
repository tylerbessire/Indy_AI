"""
FastAPI Backend Server for Indy AI
Provides REST API and WebSocket connections for real-time AI interaction
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import asyncio
import json
from datetime import datetime
import torch

from core.sentient_agent import SentientAIAgent

# Initialize FastAPI
app = FastAPI(
    title="Indy AI - Sentient Learning System",
    description="Real-time plastic neural network with persistent memory",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global AI agent instance
agent: Optional[SentientAIAgent] = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()


# Pydantic models for API
class InitializeRequest(BaseModel):
    input_size: int = 128
    hidden_sizes: List[int] = [256, 512, 256]
    output_size: int = 64
    learning_rate: float = 0.001

class InteractionRequest(BaseModel):
    input_data: Dict[str, Any]
    learn: bool = True

class MemoryQuery(BaseModel):
    query: str
    memory_type: str = "all"
    n_results: int = 5

class LearnRequest(BaseModel):
    perception: List[float]
    action: List[float]
    reward: float
    context: str


# Lifecycle events
@app.on_event("startup")
async def startup_event():
    """Initialize AI agent on startup"""
    global agent
    agent = SentientAIAgent(
        input_size=128,
        hidden_sizes=[256, 512, 256],
        output_size=64,
        learning_rate=0.001,
        memory_path="./memory_db"
    )
    print("🧠 Indy AI Agent initialized!")

    # Start background task for proactive behavior
    asyncio.create_task(proactive_behavior_loop())


async def proactive_behavior_loop():
    """Background task that makes the AI act proactively"""
    while True:
        await asyncio.sleep(5)  # Check every 5 seconds

        if agent and agent.behavior_engine.should_act():
            # Generate proactive action
            proactive_action = agent.behavior_engine.generate_proactive_action()

            # Broadcast to all connected clients
            await manager.broadcast({
                'type': 'proactive_action',
                'timestamp': datetime.now().isoformat(),
                'action': proactive_action.cpu().numpy().tolist(),
                'internal_state': agent.behavior_engine.get_proactive_state()
            })


# REST API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Indy AI - Sentient Learning System",
        "timestamp": datetime.now().isoformat(),
        "agent_initialized": agent is not None
    }

@app.post("/initialize")
async def initialize_agent(request: InitializeRequest):
    """Initialize or reinitialize the AI agent"""
    global agent

    agent = SentientAIAgent(
        input_size=request.input_size,
        hidden_sizes=request.hidden_sizes,
        output_size=request.output_size,
        learning_rate=request.learning_rate,
        memory_path="./memory_db"
    )

    return {
        "status": "success",
        "message": "Agent initialized",
        "config": {
            "input_size": request.input_size,
            "hidden_sizes": request.hidden_sizes,
            "output_size": request.output_size,
            "learning_rate": request.learning_rate
        }
    }

@app.post("/interact")
async def interact(request: InteractionRequest):
    """Interact with the AI agent"""
    if not agent:
        raise HTTPException(status_code=400, detail="Agent not initialized")

    try:
        response = agent.interact(
            input_data=request.input_data,
            learn_from_interaction=request.learn
        )

        # Broadcast update to WebSocket clients
        await manager.broadcast({
            'type': 'interaction_update',
            'timestamp': datetime.now().isoformat(),
            'response': response
        })

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reflect")
async def reflect():
    """Get AI's self-reflection"""
    if not agent:
        raise HTTPException(status_code=400, detail="Agent not initialized")

    reflection = agent.reflect()
    return reflection

@app.get("/state")
async def get_state():
    """Get current AI state"""
    if not agent:
        raise HTTPException(status_code=400, detail="Agent not initialized")

    return {
        'self_state': agent.self_state,
        'interaction_count': agent.interaction_count,
        'last_interaction': agent.last_interaction,
        'learning_enabled': agent.learning_enabled
    }

@app.get("/brain/weights")
async def get_brain_weights():
    """Get neural network weight visualization data"""
    if not agent:
        raise HTTPException(status_code=400, detail="Agent not initialized")

    return agent.brain.get_weight_visualization_data()

@app.get("/brain/activations")
async def get_activations():
    """Get current neural activations"""
    if not agent:
        raise HTTPException(status_code=400, detail="Agent not initialized")

    return {
        'activations': [
            act.tolist() if isinstance(act, torch.Tensor) else act
            for act in agent.brain.activations
        ]
    }

@app.post("/memory/query")
async def query_memory(query: MemoryQuery):
    """Query the persistent memory system"""
    if not agent:
        raise HTTPException(status_code=400, detail="Agent not initialized")

    results = agent.long_term_memory.recall_similar_memories(
        query=query.query,
        memory_type=query.memory_type,
        n_results=query.n_results
    )

    return {'results': results}

@app.get("/memory/stats")
async def memory_stats():
    """Get memory system statistics"""
    if not agent:
        raise HTTPException(status_code=400, detail="Agent not initialized")

    stats = agent.long_term_memory.get_memory_statistics()
    return stats

@app.post("/learn")
async def learn(request: LearnRequest):
    """Explicitly train the AI with given data"""
    if not agent:
        raise HTTPException(status_code=400, detail="Agent not initialized")

    try:
        perception = torch.tensor(request.perception, dtype=torch.float32)
        action = torch.tensor(request.action, dtype=torch.float32)

        agent.learn(
            perception=perception,
            action=action,
            reward=request.reward,
            context=request.context
        )

        # Broadcast learning update
        await manager.broadcast({
            'type': 'learning_update',
            'timestamp': datetime.now().isoformat(),
            'reward': request.reward,
            'context': request.context
        })

        return {
            "status": "success",
            "message": "Learning completed",
            "learning_progress": agent.self_state['learning_progress']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save")
async def save_agent(filepath: str = "./agent_state.json"):
    """Save agent state to file"""
    if not agent:
        raise HTTPException(status_code=400, detail="Agent not initialized")

    try:
        agent.save_state(filepath)
        return {
            "status": "success",
            "message": f"Agent saved to {filepath}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load")
async def load_agent(filepath: str = "./agent_state.json"):
    """Load agent state from file"""
    if not agent:
        raise HTTPException(status_code=400, detail="Agent not initialized")

    try:
        agent.load_state(filepath)
        return {
            "status": "success",
            "message": f"Agent loaded from {filepath}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/learning/toggle")
async def toggle_learning(enabled: bool):
    """Enable or disable learning"""
    if not agent:
        raise HTTPException(status_code=400, detail="Agent not initialized")

    agent.learning_enabled = enabled
    return {
        "status": "success",
        "learning_enabled": agent.learning_enabled
    }


# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection for real-time AI updates"""
    await manager.connect(websocket)

    try:
        # Send initial state
        if agent:
            await websocket.send_json({
                'type': 'connection_established',
                'timestamp': datetime.now().isoformat(),
                'agent_state': {
                    'initialized': True,
                    'self_state': agent.self_state,
                    'interaction_count': agent.interaction_count
                }
            })

        # Listen for messages from client
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message['type'] == 'interact':
                # Process interaction
                response = agent.interact(
                    input_data=message['data'],
                    learn_from_interaction=message.get('learn', True)
                )

                # Send response
                await websocket.send_json({
                    'type': 'interaction_response',
                    'timestamp': datetime.now().isoformat(),
                    'response': response
                })

            elif message['type'] == 'request_state':
                # Send current state
                reflection = agent.reflect()
                await websocket.send_json({
                    'type': 'state_update',
                    'timestamp': datetime.now().isoformat(),
                    'reflection': reflection
                })

            elif message['type'] == 'request_weights':
                # Send weight visualization
                weights_data = agent.brain.get_weight_visualization_data()
                await websocket.send_json({
                    'type': 'weights_update',
                    'timestamp': datetime.now().isoformat(),
                    'data': weights_data
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
