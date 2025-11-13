import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import NeuralVisualization from './components/NeuralVisualization';
import WeightMatrix from './components/WeightMatrix';
import MemoryExplorer from './components/MemoryExplorer';
import InteractionPanel from './components/InteractionPanel';
import MetricsPanel from './components/MetricsPanel';

function App() {
  const [connected, setConnected] = useState(false);
  const [agentState, setAgentState] = useState(null);
  const [weightData, setWeightData] = useState(null);
  const [activityLog, setActivityLog] = useState([]);
  const [reflection, setReflection] = useState(null);
  const wsRef = useRef(null);

  useEffect(() => {
    connectWebSocket();
    fetchInitialState();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const connectWebSocket = () => {
    const ws = new WebSocket('ws://localhost:8000/ws');

    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnected(true);
      addActivityLog('Connected to AI agent', 'success');
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      handleWebSocketMessage(message);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      addActivityLog('Connection error', 'error');
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setConnected(false);
      addActivityLog('Disconnected from AI agent', 'warning');
      // Attempt reconnection after 3 seconds
      setTimeout(connectWebSocket, 3000);
    };

    wsRef.current = ws;
  };

  const handleWebSocketMessage = (message) => {
    console.log('Received:', message);

    switch (message.type) {
      case 'connection_established':
        setAgentState(message.agent_state);
        break;

      case 'interaction_response':
        addActivityLog(`Interaction: ${message.response.message}`, 'info');
        setAgentState(message.response.internal_state);
        break;

      case 'state_update':
        setReflection(message.reflection);
        break;

      case 'weights_update':
        setWeightData(message.data);
        break;

      case 'learning_update':
        addActivityLog(`Learning: reward=${message.reward.toFixed(3)}`, 'success');
        break;

      case 'proactive_action':
        addActivityLog('AI took proactive action!', 'proactive');
        break;

      default:
        console.log('Unknown message type:', message.type);
    }
  };

  const fetchInitialState = async () => {
    try {
      const response = await fetch('http://localhost:8000/state');
      const data = await response.json();
      setAgentState(data);
    } catch (error) {
      console.error('Error fetching state:', error);
    }
  };

  const addActivityLog = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setActivityLog(prev => [
      { message, type, timestamp },
      ...prev.slice(0, 99) // Keep last 100 items
    ]);
  };

  const requestWeightsUpdate = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'request_weights' }));
    }
  };

  const requestStateUpdate = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'request_state' }));
    }
  };

  // Auto-refresh weights every 2 seconds when connected
  useEffect(() => {
    if (connected) {
      const interval = setInterval(requestWeightsUpdate, 2000);
      return () => clearInterval(interval);
    }
  }, [connected]);

  return (
    <div className="App">
      <header className="app-header">
        <h1>🧠 Indy AI - Sentient Learning Playground</h1>
        <div className="connection-status">
          <span className={`status-indicator ${connected ? 'status-online' : 'status-offline'}`}></span>
          {connected ? 'Connected' : 'Disconnected'}
        </div>
      </header>

      <div className="container">
        <div className="main-grid">
          {/* Left Column - Metrics and Controls */}
          <div className="left-column">
            <MetricsPanel
              agentState={agentState}
              reflection={reflection}
              onRefresh={requestStateUpdate}
            />

            <InteractionPanel
              websocket={wsRef.current}
              connected={connected}
              onActivity={addActivityLog}
            />

            <div className="card">
              <h3>Activity Log</h3>
              <div className="activity-log">
                {activityLog.map((item, index) => (
                  <div key={index} className={`activity-item activity-${item.type}`}>
                    {item.message}
                    <span className="timestamp">{item.timestamp}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Right Column - Visualizations */}
          <div className="right-column">
            <NeuralVisualization
              weightData={weightData}
              activations={agentState?.self_state}
            />

            <WeightMatrix
              weightHistory={weightData?.history}
            />

            <MemoryExplorer
              connected={connected}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
