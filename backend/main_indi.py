"""
FastAPI Backend for INDI - Pure Developmental System
The Training Ground API

This exposes Indi's learning environment through REST and WebSocket.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import asyncio
import json
from datetime import datetime
import base64
from PIL import Image
import io
import numpy as np

from core.indi import Indi

# Initialize FastAPI
app = FastAPI(
    title="INDI - Independent Neural Developmental Intelligence",
    description="Pure developmental learning system - Training Ground API",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Indi instance
indi: Optional[Indi] = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections[:]:  # Copy list to avoid modification during iteration
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error broadcasting: {e}")
                self.active_connections.remove(connection)

manager = ConnectionManager()


# Pydantic models
class BirthRequest(BaseModel):
    z_dim: int = 64
    learning_rate: float = 0.0001
    n_concepts: int = 100

class TeachPictureRequest(BaseModel):
    image_base64: str
    word: str
    repetition: int = 1

class SpeakRequest(BaseModel):
    text: str

class CorrectionRequest(BaseModel):
    correction: str
    episode_id: Optional[int] = None

class ContrastRequest(BaseModel):
    item1_type: str  # "text" or "image"
    item1_data: str
    item2_type: str
    item2_data: str
    are_same: bool
    label1: Optional[str] = None
    label2: Optional[str] = None


# Lifecycle
@app.on_event("startup")
async def startup_event():
    """Birth Indi on startup"""
    global indi
    indi = Indi(
        z_dim=64,
        memory_path="./indi_memory",
        learning_rate=0.0001,
        n_concepts=100
    )
    print("✨ Indi is alive!")

    # Start circadian rhythm monitor
    asyncio.create_task(circadian_rhythm_monitor())


async def circadian_rhythm_monitor():
    """Background task monitoring day/night transitions"""
    while True:
        await asyncio.sleep(60)  # Check every minute

        if indi:
            result = indi.check_circadian_rhythm()

            if result.get('should_transition'):
                # Broadcast phase transition
                await manager.broadcast({
                    'type': 'phase_transition',
                    'timestamp': datetime.now().isoformat(),
                    'data': result
                })


# REST API Endpoints

@app.get("/")
async def root():
    return {
        "status": "alive",
        "name": "Indi",
        "message": "Independent Neural Developmental Intelligence",
        "timestamp": datetime.now().isoformat(),
        "alive": indi is not None
    }


@app.post("/birth")
async def birth(request: BirthRequest):
    """Birth a new Indi (or rebirth with fresh brain)"""
    global indi

    indi = Indi(
        z_dim=request.z_dim,
        memory_path="./indi_memory",
        learning_rate=request.learning_rate,
        n_concepts=request.n_concepts
    )

    await manager.broadcast({
        'type': 'birth',
        'timestamp': datetime.now().isoformat(),
        'message': 'Indi has been born!'
    })

    return {
        "status": "success",
        "message": "Indi is alive",
        "birth_time": indi.birth_time.isoformat()
    }


@app.get("/state")
async def get_state():
    """Get Indi's complete state"""
    if not indi:
        raise HTTPException(status_code=400, detail="Indi not initialized")

    return indi.get_state()


@app.get("/dashboard")
async def get_dashboard():
    """Get dashboard data"""
    if not indi:
        raise HTTPException(status_code=400, detail="Indi not initialized")

    return indi.get_dashboard_data()


@app.post("/teach/picture")
async def teach_picture(request: TeachPictureRequest):
    """
    Teach Indi by showing a picture and saying a word.
    Core grounding mechanism.
    """
    if not indi:
        raise HTTPException(status_code=400, detail="Indi not initialized")

    try:
        # Decode image
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes))

        # Teach
        result = indi.teach_picture_word(image, request.word, request.repetition)

        # Broadcast to connected clients
        await manager.broadcast({
            'type': 'picture_taught',
            'timestamp': datetime.now().isoformat(),
            'word': request.word,
            'result': result
        })

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/teach/picture-upload")
async def teach_picture_upload(file: UploadFile = File(...), word: str = "", repetition: int = 1):
    """
    Teach Indi with file upload.
    Alternative to base64 encoding.
    """
    if not indi:
        raise HTTPException(status_code=400, detail="Indi not initialized")

    try:
        # Read image from upload
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Teach
        result = indi.teach_picture_word(image, word, repetition)

        # Broadcast
        await manager.broadcast({
            'type': 'picture_taught',
            'timestamp': datetime.now().isoformat(),
            'word': word,
            'result': result
        })

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speak")
async def speak(request: SpeakRequest):
    """
    Speak to Indi.
    She'll listen and try to respond.
    """
    if not indi:
        raise HTTPException(status_code=400, detail="Indi not initialized")

    try:
        result = indi.speak(request.text)

        # Broadcast
        await manager.broadcast({
            'type': 'speech',
            'timestamp': datetime.now().isoformat(),
            'teacher_said': request.text,
            'indi_responded': result.get('response_text'),
            'result': result
        })

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/correct")
async def correct(request: CorrectionRequest):
    """
    Correct Indi's understanding.
    High-salience learning signal.
    """
    if not indi:
        raise HTTPException(status_code=400, detail="Indi not initialized")

    try:
        result = indi.correct(request.correction, request.episode_id)

        # Broadcast
        await manager.broadcast({
            'type': 'correction',
            'timestamp': datetime.now().isoformat(),
            'correction': request.correction,
            'result': result
        })

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/teach/contrast")
async def teach_contrast(request: ContrastRequest):
    """
    Teach contrast learning: same or different?
    """
    if not indi:
        raise HTTPException(status_code=400, detail="Indi not initialized")

    try:
        # Process item1
        if request.item1_type == "text":
            item1 = request.item1_data
        else:
            # Image
            image_bytes = base64.b64decode(request.item1_data)
            item1 = Image.open(io.BytesIO(image_bytes))

        # Process item2
        if request.item2_type == "text":
            item2 = request.item2_data
        else:
            image_bytes = base64.b64decode(request.item2_data)
            item2 = Image.open(io.BytesIO(image_bytes))

        # Teach
        result = indi.teach_contrast(
            item1, item2,
            request.are_same,
            request.label1, request.label2
        )

        # Broadcast
        await manager.broadcast({
            'type': 'contrast_taught',
            'timestamp': datetime.now().isoformat(),
            'result': result
        })

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/start")
async def start_session():
    """Start a teaching session"""
    if not indi:
        raise HTTPException(status_code=400, detail="Indi not initialized")

    session_id = indi.start_teaching_session()

    return {
        "status": "success",
        "session_id": session_id
    }


@app.post("/session/end")
async def end_session():
    """End teaching session"""
    if not indi:
        raise HTTPException(status_code=400, detail="Indi not initialized")

    summary = indi.end_teaching_session()

    return {
        "status": "success",
        "summary": summary
    }


@app.post("/sleep")
async def sleep():
    """
    Put Indi to sleep for consolidation.
    NREM → REM → Schema updates.
    """
    if not indi:
        raise HTTPException(status_code=400, detail="Indi not initialized")

    try:
        result = indi.sleep()

        # Broadcast
        await manager.broadcast({
            'type': 'sleep_cycle',
            'timestamp': datetime.now().isoformat(),
            'result': result
        })

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/circadian")
async def get_circadian():
    """Get circadian rhythm status"""
    if not indi:
        raise HTTPException(status_code=400, detail="Indi not initialized")

    phase_info = indi.scheduler.get_current_phase_info()

    return phase_info


@app.post("/save")
async def save(filepath: Optional[str] = None):
    """Save Indi's state"""
    if not indi:
        raise HTTPException(status_code=400, detail="Indi not initialized")

    try:
        indi.save(filepath)

        return {
            "status": "success",
            "message": "Indi saved"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load")
async def load(filepath: Optional[str] = None):
    """Load Indi's state"""
    if not indi:
        raise HTTPException(status_code=400, detail="Indi not initialized")

    try:
        success = indi.load(filepath)

        if success:
            return {
                "status": "success",
                "message": "Indi loaded"
            }
        else:
            raise HTTPException(status_code=404, detail="Save file not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/concepts")
async def get_concepts():
    """Get discovered concepts"""
    if not indi:
        raise HTTPException(status_code=400, detail="Indi not initialized")

    concepts = []
    for concept in indi.memory.semantic.get_all_concepts():
        concepts.append({
            'id': concept.concept_id,
            'words': concept.words,
            'activation_count': concept.activation_count,
            'created_at': concept.created_at
        })

    return {
        "concepts": concepts,
        "total": len(concepts)
    }


@app.get("/memory/episodes")
async def get_episodes(n: int = 10):
    """Get recent episodes"""
    if not indi:
        raise HTTPException(status_code=400, detail="Indi not initialized")

    episodes = [ep.to_dict() for ep in indi.memory.episodic.get_recent(n)]

    return {
        "episodes": episodes,
        "total_episodes": indi.memory.episodic.count()
    }


# WebSocket for real-time interaction
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time Training Ground connection"""
    await manager.connect(websocket)

    try:
        # Send initial state
        if indi:
            state = indi.get_dashboard_data()
            await websocket.send_json({
                'type': 'connected',
                'timestamp': datetime.now().isoformat(),
                'state': state
            })

        # Listen for commands
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            message_type = message.get('type')

            if message_type == 'speak':
                result = indi.speak(message['text'])
                await websocket.send_json({
                    'type': 'speak_response',
                    'timestamp': datetime.now().isoformat(),
                    'result': result
                })

            elif message_type == 'request_state':
                state = indi.get_dashboard_data()
                await websocket.send_json({
                    'type': 'state_update',
                    'timestamp': datetime.now().isoformat(),
                    'state': state
                })

            elif message_type == 'sleep':
                result = indi.sleep()
                await websocket.send_json({
                    'type': 'sleep_result',
                    'timestamp': datetime.now().isoformat(),
                    'result': result
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected from Training Ground")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
