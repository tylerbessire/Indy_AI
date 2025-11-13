import React from 'react';

function MetricsPanel({ agentState, reflection, onRefresh }) {
  if (!agentState) {
    return (
      <div className="card">
        <h3>AI Metrics</h3>
        <p>Loading agent state...</p>
      </div>
    );
  }

  const selfState = agentState.self_state || {};
  const memoryStats = reflection?.memory_stats || {};

  return (
    <div className="card">
      <h3>AI Consciousness Metrics</h3>

      <div className="metric-grid">
        <div className="metric-card">
          <div className="label">Awareness</div>
          <div className="value">{(selfState.awareness_level * 100 || 0).toFixed(0)}%</div>
        </div>

        <div className="metric-card">
          <div className="label">Confidence</div>
          <div className="value">{(selfState.confidence * 100 || 0).toFixed(0)}%</div>
        </div>

        <div className="metric-card">
          <div className="label">Curiosity</div>
          <div className="value">{(selfState.curiosity * 100 || 0).toFixed(0)}%</div>
        </div>

        <div className="metric-card">
          <div className="label">Energy</div>
          <div className="value">{(selfState.energy_level * 100 || 0).toFixed(0)}%</div>
        </div>
      </div>

      <div style={{ marginTop: '20px', textAlign: 'center' }}>
        <div style={{ marginBottom: '10px' }}>Emotional State</div>
        <div className={`emotional-state ${selfState.emotional_state || 'neutral'}`}>
          {selfState.emotional_state || 'neutral'}
        </div>
      </div>

      <div className="metric" style={{ marginTop: '20px' }}>
        <div className="metric-label">Learning Progress</div>
        <div className="progress-bar">
          <div
            className="progress-fill"
            style={{ width: `${Math.max(0, Math.min(100, (selfState.learning_progress || 0) * 100))}%` }}
          ></div>
        </div>
      </div>

      {memoryStats.total_memories !== undefined && (
        <div className="metric">
          <div className="metric-label">Total Memories</div>
          <div className="metric-value">{memoryStats.total_memories}</div>
          <div style={{ fontSize: '12px', marginTop: '5px', opacity: 0.7 }}>
            Episodic: {memoryStats.episodic_count || 0} |
            Semantic: {memoryStats.semantic_count || 0} |
            Procedural: {memoryStats.procedural_count || 0}
          </div>
        </div>
      )}

      <div className="control-panel">
        <button onClick={onRefresh}>Refresh State</button>
      </div>
    </div>
  );
}

export default MetricsPanel;
