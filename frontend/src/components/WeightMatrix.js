import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function WeightMatrix({ weightHistory }) {
  if (!weightHistory || weightHistory.length === 0) {
    return (
      <div className="card">
        <h3>Weight Evolution</h3>
        <div className="visualization-container">
          <p>Waiting for weight data...</p>
        </div>
      </div>
    );
  }

  // Prepare data for the chart
  const chartData = weightHistory.map((snapshot, idx) => {
    const dataPoint = { index: idx };

    snapshot.weights.forEach((layer, layerIdx) => {
      dataPoint[`L${layerIdx}_mean`] = layer.mean;
      dataPoint[`L${layerIdx}_std`] = layer.std;
    });

    return dataPoint;
  });

  // Get layer count
  const layerCount = weightHistory[0]?.weights.length || 0;

  return (
    <div className="card">
      <h3>Weight Plasticity Over Time</h3>

      <div style={{ marginBottom: '20px' }}>
        <h4 style={{ fontSize: '16px', marginBottom: '10px' }}>Weight Means</h4>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis dataKey="index" stroke="rgba(255,255,255,0.5)" />
            <YAxis stroke="rgba(255,255,255,0.5)" />
            <Tooltip
              contentStyle={{
                background: 'rgba(0, 0, 0, 0.8)',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '5px'
              }}
            />
            <Legend />
            {[...Array(layerCount)].map((_, idx) => (
              <Line
                key={idx}
                type="monotone"
                dataKey={`L${idx}_mean`}
                stroke={`hsl(${idx * 60}, 100%, 60%)`}
                strokeWidth={2}
                dot={false}
                name={`Layer ${idx}`}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div>
        <h4 style={{ fontSize: '16px', marginBottom: '10px' }}>Weight Standard Deviations</h4>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis dataKey="index" stroke="rgba(255,255,255,0.5)" />
            <YAxis stroke="rgba(255,255,255,0.5)" />
            <Tooltip
              contentStyle={{
                background: 'rgba(0, 0, 0, 0.8)',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: '5px'
              }}
            />
            <Legend />
            {[...Array(layerCount)].map((_, idx) => (
              <Line
                key={idx}
                type="monotone"
                dataKey={`L${idx}_std`}
                stroke={`hsl(${idx * 60 + 30}, 100%, 60%)`}
                strokeWidth={2}
                dot={false}
                name={`Layer ${idx}`}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div style={{ fontSize: '12px', opacity: 0.7, marginTop: '15px' }}>
        Real-time weight updates showing neural plasticity in action
      </div>
    </div>
  );
}

export default WeightMatrix;
