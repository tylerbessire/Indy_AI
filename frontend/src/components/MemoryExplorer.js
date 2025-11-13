import React, { useState } from 'react';

function MemoryExplorer({ connected }) {
  const [searchQuery, setSearchQuery] = useState('');
  const [memoryType, setMemoryType] = useState('all');
  const [searchResults, setSearchResults] = useState([]);
  const [memoryStats, setMemoryStats] = useState(null);
  const [loading, setLoading] = useState(false);

  const searchMemory = async () => {
    if (!searchQuery.trim()) return;

    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/memory/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: searchQuery,
          memory_type: memoryType,
          n_results: 5
        })
      });
      const data = await response.json();
      setSearchResults(data.results || []);
    } catch (error) {
      console.error('Error searching memory:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadMemoryStats = async () => {
    try {
      const response = await fetch('http://localhost:8000/memory/stats');
      const data = await response.json();
      setMemoryStats(data);
    } catch (error) {
      console.error('Error loading memory stats:', error);
    }
  };

  React.useEffect(() => {
    if (connected) {
      loadMemoryStats();
      const interval = setInterval(loadMemoryStats, 5000);
      return () => clearInterval(interval);
    }
  }, [connected]);

  return (
    <div className="card">
      <h3>Memory System Explorer</h3>

      {memoryStats && (
        <div className="metric-grid" style={{ marginBottom: '20px' }}>
          <div className="metric-card">
            <div className="label">Total Memories</div>
            <div className="value">{memoryStats.total_memories}</div>
          </div>
          <div className="metric-card">
            <div className="label">Episodic</div>
            <div className="value">{memoryStats.episodic_count}</div>
          </div>
          <div className="metric-card">
            <div className="label">Semantic</div>
            <div className="value">{memoryStats.semantic_count}</div>
          </div>
          <div className="metric-card">
            <div className="label">Procedural</div>
            <div className="value">{memoryStats.procedural_count}</div>
          </div>
        </div>
      )}

      <div className="input-group">
        <label>Search Memories</label>
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Enter search query..."
          onKeyPress={(e) => {
            if (e.key === 'Enter') {
              searchMemory();
            }
          }}
        />
      </div>

      <div className="input-group">
        <label>Memory Type</label>
        <select
          value={memoryType}
          onChange={(e) => setMemoryType(e.target.value)}
          style={{
            background: 'rgba(255, 255, 255, 0.2)',
            border: '1px solid rgba(255, 255, 255, 0.3)',
            borderRadius: '8px',
            padding: '10px',
            color: 'white',
            fontSize: '14px',
            width: '100%'
          }}
        >
          <option value="all">All Memories</option>
          <option value="episodic">Episodic (Experiences)</option>
          <option value="semantic">Semantic (Knowledge)</option>
          <option value="procedural">Procedural (Skills)</option>
        </select>
      </div>

      <button onClick={searchMemory} disabled={!connected || loading}>
        {loading ? 'Searching...' : 'Search Memory'}
      </button>

      {searchResults.length > 0 && (
        <div style={{ marginTop: '20px' }}>
          <h4 style={{ fontSize: '16px', marginBottom: '10px' }}>Search Results</h4>
          <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
            {searchResults.map((result, idx) => (
              <div
                key={idx}
                style={{
                  background: 'rgba(255, 255, 255, 0.1)',
                  padding: '15px',
                  marginBottom: '10px',
                  borderRadius: '8px',
                  borderLeft: `3px solid hsl(${result.similarity * 120}, 100%, 50%)`
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                  <span style={{ fontSize: '12px', opacity: 0.7, textTransform: 'uppercase' }}>
                    {result.type}
                  </span>
                  <span style={{ fontSize: '12px', opacity: 0.7' }}>
                    Similarity: {(result.similarity * 100).toFixed(1)}%
                  </span>
                </div>
                <div style={{ fontSize: '14px' }}>{result.content}</div>
                {result.metadata && (
                  <div style={{ fontSize: '11px', opacity: 0.6, marginTop: '5px' }}>
                    {result.metadata.timestamp && (
                      <span>{new Date(result.metadata.timestamp).toLocaleString()}</span>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      <div style={{ fontSize: '12px', opacity: 0.7, marginTop: '15px' }}>
        Semantic search across all AI memories using vector embeddings
      </div>
    </div>
  );
}

export default MemoryExplorer;
