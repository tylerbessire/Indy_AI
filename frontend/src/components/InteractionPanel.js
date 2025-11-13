import React, { useState } from 'react';

function InteractionPanel({ websocket, connected, onActivity }) {
  const [inputText, setInputText] = useState('');
  const [rewardValue, setRewardValue] = useState(0.5);
  const [isLearning, setIsLearning] = useState(true);

  const sendInteraction = async () => {
    if (!inputText.trim()) return;

    const interactionData = {
      text: inputText,
      reward: parseFloat(rewardValue),
      context: 'User interaction',
      query: inputText
    };

    if (websocket && websocket.readyState === WebSocket.OPEN) {
      // Send via WebSocket
      websocket.send(JSON.stringify({
        type: 'interact',
        data: interactionData,
        learn: isLearning
      }));
      onActivity(`Sent: "${inputText}"`, 'info');
    } else {
      // Fallback to REST API
      try {
        const response = await fetch('http://localhost:8000/interact', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            input_data: interactionData,
            learn: isLearning
          })
        });
        const data = await response.json();
        onActivity(`Response: ${data.message}`, 'success');
      } catch (error) {
        onActivity(`Error: ${error.message}`, 'error');
      }
    }

    setInputText('');
  };

  const trainWithExample = async () => {
    try {
      const response = await fetch('http://localhost:8000/learn', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          perception: Array(128).fill(0).map(() => Math.random() * 2 - 1),
          action: Array(64).fill(0).map(() => Math.random() * 2 - 1),
          reward: parseFloat(rewardValue),
          context: 'Manual training example'
        })
      });
      const data = await response.json();
      onActivity(`Training completed: ${data.message}`, 'success');
    } catch (error) {
      onActivity(`Training error: ${error.message}`, 'error');
    }
  };

  const toggleLearning = async () => {
    try {
      const newState = !isLearning;
      const response = await fetch(`http://localhost:8000/learning/toggle?enabled=${newState}`, {
        method: 'POST'
      });
      const data = await response.json();
      setIsLearning(newState);
      onActivity(`Learning ${newState ? 'enabled' : 'disabled'}`, 'info');
    } catch (error) {
      onActivity(`Error toggling learning: ${error.message}`, 'error');
    }
  };

  return (
    <div className="card">
      <h3>Interaction Panel</h3>

      <div className="input-group">
        <label>Message to AI</label>
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Type your message here..."
          rows="4"
          onKeyPress={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              sendInteraction();
            }
          }}
        />
      </div>

      <div className="input-group">
        <label>Reward Value ({rewardValue})</label>
        <input
          type="range"
          min="-1"
          max="1"
          step="0.1"
          value={rewardValue}
          onChange={(e) => setRewardValue(e.target.value)}
        />
        <div style={{ fontSize: '12px', opacity: 0.7, marginTop: '5px' }}>
          Positive rewards encourage behavior, negative discourage it
        </div>
      </div>

      <div className="control-panel">
        <button onClick={sendInteraction} disabled={!connected}>
          Send Interaction
        </button>
        <button onClick={trainWithExample} disabled={!connected}>
          Train Random
        </button>
        <button
          onClick={toggleLearning}
          style={{
            background: isLearning
              ? 'linear-gradient(135deg, #00ff88, #00d4ff)'
              : 'rgba(255, 255, 255, 0.2)'
          }}
        >
          {isLearning ? 'Learning ON' : 'Learning OFF'}
        </button>
      </div>

      <div style={{ marginTop: '20px', fontSize: '12px', opacity: 0.7 }}>
        Status: {connected ? '🟢 Connected' : '🔴 Disconnected'}
        {isLearning && ' | 🧠 Learning Active'}
      </div>
    </div>
  );
}

export default InteractionPanel;
