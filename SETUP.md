# Setup Guide for Indy AI

This guide will help you get Indy AI up and running on your system.

## System Requirements

- **Operating System**: Linux, macOS, or Windows (with WSL recommended)
- **Python**: 3.8 or higher
- **Node.js**: 16 or higher
- **RAM**: At least 4GB recommended
- **Storage**: At least 2GB free space (for dependencies and memory database)

## Installation Steps

### Option 1: Quick Start (Linux/macOS)

```bash
# Make the start script executable
chmod +x start.sh

# Run the start script
./start.sh
```

The script will:
1. Check prerequisites
2. Create a Python virtual environment
3. Install all dependencies
4. Start both backend and frontend servers

### Option 2: Manual Setup

#### Step 1: Set up Python Backend

```bash
# Create virtual environment (recommended)
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

#### Step 2: Set up React Frontend

```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Return to root directory
cd ..
```

#### Step 3: Start the Servers

**Terminal 1 - Backend:**
```bash
cd backend
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

## Accessing the Application

Once both servers are running:

- **Frontend UI**: Open browser to `http://localhost:3000`
- **Backend API**: Available at `http://localhost:8000`
- **API Documentation**: Visit `http://localhost:8000/docs` for interactive API docs

## Troubleshooting

### Port Already in Use

If port 8000 or 3000 is already in use:

**Backend (port 8000):**
Edit `backend/main.py`, change the last line:
```python
uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
```

**Frontend (port 3000):**
```bash
PORT=3001 npm start
```

### Python Dependencies Issues

If you get errors installing Python packages:

```bash
# Upgrade pip
pip install --upgrade pip

# Install with verbose output to see errors
pip install -v -r requirements.txt
```

### ChromaDB Issues

If ChromaDB fails to install:

**On Linux:**
```bash
# Install build essentials
sudo apt-get install build-essential python3-dev
```

**On macOS:**
```bash
# Install Xcode command line tools
xcode-select --install
```

### Node.js Memory Issues

If npm install fails with memory errors:

```bash
# Increase Node memory limit
export NODE_OPTIONS="--max-old-space-size=4096"
npm install
```

### PyTorch Installation

PyTorch is large (~2GB). If installation is slow or fails:

**CPU-only version (smaller, faster):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**GPU version (if you have CUDA):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Verifying Installation

### Test Backend

```bash
curl http://localhost:8000
```

Expected response:
```json
{
  "status": "online",
  "message": "Indy AI - Sentient Learning System",
  ...
}
```

### Test Frontend

Open `http://localhost:3000` in your browser. You should see:
- Header: "Indy AI - Sentient Learning Playground"
- Connection status indicator
- Multiple panels with metrics and controls

## Optional: Development Setup

### Install Development Tools

```bash
# Python code formatting
pip install black flake8

# React development tools
# Install React DevTools extension in your browser
```

### Environment Variables

Create `.env` files for configuration:

**backend/.env:**
```
MEMORY_DB_PATH=./memory_db
LOG_LEVEL=info
```

**frontend/.env:**
```
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
```

## Next Steps

Once installation is complete:

1. Read the [README.md](README.md) for usage instructions
2. Try the example interactions
3. Explore the API documentation at `http://localhost:8000/docs`
4. Experiment with different learning rates and rewards

## Getting Help

If you encounter issues:

1. Check this troubleshooting section
2. Review the error messages carefully
3. Check system requirements
4. Try the manual installation steps
5. Create an issue on GitHub with:
   - Your operating system
   - Python and Node.js versions
   - Complete error message
   - Steps to reproduce

## Uninstallation

To remove Indy AI:

```bash
# Remove virtual environment
rm -rf venv

# Remove Node modules
rm -rf frontend/node_modules

# Remove memory database
rm -rf memory_db

# Remove Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

---

Happy learning with Indy AI!
