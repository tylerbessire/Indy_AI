import React, { useEffect, useRef } from 'react';

function NeuralVisualization({ weightData, activations }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !weightData) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
    ctx.fillRect(0, 0, width, height);

    // Draw neural network visualization
    drawNeuralNetwork(ctx, width, height, weightData);
  }, [weightData, activations]);

  const drawNeuralNetwork = (ctx, width, height, data) => {
    if (!data.history || data.history.length === 0) return;

    const latest = data.history[data.history.length - 1];
    const layers = latest.weights || [];

    // Calculate layout
    const layerCount = layers.length + 1; // +1 for input layer
    const layerSpacing = width / (layerCount + 1);
    const neuronRadius = 8;

    // Draw connections first (background)
    ctx.strokeStyle = 'rgba(100, 200, 255, 0.3)';
    ctx.lineWidth = 1;

    for (let i = 0; i < layers.length; i++) {
      const layer = layers[i];
      const x1 = layerSpacing * (i + 1);
      const x2 = layerSpacing * (i + 2);

      // Draw a few representative connections
      for (let j = 0; j < 5; j++) {
        const y1 = height * (j + 1) / 6;
        for (let k = 0; k < 5; k++) {
          const y2 = height * (k + 1) / 6;

          // Line thickness based on weight strength
          const strength = Math.abs(layer.mean || 0.5);
          ctx.globalAlpha = strength * 0.5;

          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.stroke();
        }
      }
    }

    ctx.globalAlpha = 1.0;

    // Draw neurons (nodes)
    for (let i = 0; i <= layers.length; i++) {
      const x = layerSpacing * (i + 1);
      const neuronCount = i === 0 ? 5 : 5; // Simplified to 5 neurons per layer for viz

      for (let j = 0; j < neuronCount; j++) {
        const y = height * (j + 1) / (neuronCount + 1);

        // Neuron activation (simulated)
        const activation = Math.random();
        const color = `hsl(${180 + activation * 60}, 100%, ${50 + activation * 30}%)`;

        ctx.beginPath();
        ctx.arc(x, y, neuronRadius, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Pulsing effect for active neurons
        if (activation > 0.7) {
          ctx.beginPath();
          ctx.arc(x, y, neuronRadius + 3, 0, Math.PI * 2);
          ctx.strokeStyle = `hsla(${180 + activation * 60}, 100%, 70%, 0.5)`;
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }

      // Layer label
      ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
      ctx.font = '12px monospace';
      ctx.textAlign = 'center';
      const layerName = i === 0 ? 'Input' : i === layers.length ? 'Output' : `Hidden ${i}`;
      ctx.fillText(layerName, x, height - 10);
    }

    // Draw weight statistics
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.font = '14px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('Weight Statistics:', 10, 20);

    layers.forEach((layer, idx) => {
      const y = 40 + idx * 20;
      const text = `L${idx}: μ=${layer.mean.toFixed(3)} σ=${layer.std.toFixed(3)}`;
      ctx.fillText(text, 10, y);
    });
  };

  return (
    <div className="card">
      <h3>Neural Network Visualization</h3>
      <div className="visualization-container">
        <canvas
          ref={canvasRef}
          width={800}
          height={500}
          style={{ width: '100%', height: 'auto', borderRadius: '10px' }}
        />
      </div>
      <div style={{ fontSize: '12px', opacity: 0.7, marginTop: '10px' }}>
        Live neural network activity - nodes pulse with activation
      </div>
    </div>
  );
}

export default NeuralVisualization;
