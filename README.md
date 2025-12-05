# CVRP-DRL-GA Backend

A hybrid solver for the Capacitated Vehicle Routing Problem (CVRP) combining Deep Reinforcement Learning (DRL) with Genetic Algorithms (GA).

## 📋 About the Problem

The **Capacitated Vehicle Routing Problem (CVRP)** is a classic optimization problem in operations research and logistics. Given a set of customers with known demands and a fleet of vehicles with limited capacity, the goal is to:

- Design optimal routes starting and ending at a central depot
- Visit all customers exactly once
- Minimize the total travel distance
- Respect vehicle capacity constraints

CVRP is NP-hard, making it computationally challenging for large instances. This project addresses it using a hybrid metaheuristic approach.

## 🧠 Solution Approach

This project implements three different solving strategies:

### 1. **NeuroGen (DRL+GA)** - Hybrid Approach
- Uses a trained Deep Reinforcement Learning model to generate initial high-quality populations
- Applies Genetic Algorithm operators (selection, crossover, mutation) for refinement
- Combines the exploration power of GA with DRL's learned heuristics
- Typically provides the best balance between solution quality and computational time

### 2. **Pure Genetic Algorithm (GA)**
- Traditional metaheuristic approach
- Population-based evolution with selection, crossover, and mutation
- Highly configurable parameters (population size, mutation rate, crossover rate)
- Good for baseline comparison and smaller instances

### 3. **Pure DRL**
- Direct application of trained Pointer Network with attention mechanism
- Fast inference, no iterative optimization
- Three pre-trained models available: Junior (25-30 customers), Mid (50-60 customers), Expert (100-120 customers)
- Best for quick approximate solutions

## 🏗️ Project Structure

```
app/
├── api/                  # REST API endpoints
│   └── routers/
│       ├── drl/          # DRL model handlers
│       ├── instances/    # Instance handlers
│       └── solve/        # Solving handlers
├── core/                 # Core algorithms
│   ├── drl/              # Deep RL implementation (Pointer Networks)
│   ├── ga/               # Genetic Algorithm implementation
│   ├── operations/       # Business logic
│   └── utils/            # Helper functions
├── data/
│   └── presets/          # Pre-configured CVRP instances
├── instances/            # Stored user generated CVRP intances
├── schemas/              # Pydantic data models
└── scripts/              
    ├── drl/              # Training scripts for DRL models
    └── presets/          # Script for generate the presets (Pre-configured CVRP intances)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip

### Installation from Scratch

1. **Clone the repository**
   ```bash
   git clone https://github.com/natandreli/CVRP-DRL-GA-backend.git
   cd CVRP-DRL-GA-backend
   ```

2. **Create a virtual environment**
   
   Using uv (recommended):
   ```bash
   uv venv
   ```
   
   Using Python's venv:
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   
   On Windows:
   ```bash
   .venv\Scripts\activate
   ```
   
   On macOS/Linux:
   ```bash
   source .venv/bin/activate
   ```

4. **Install dependencies**
   
   Using uv (recommended):
   ```bash
   uv sync
   ```
   
   Using pip:
   ```bash
   pip install -e .
   ```

### Running the Server

**Option 1: Using uv (recommended)**
```bash
uv run fastapi dev app/main.py
```

**Option 2: Using uvicorn directly (after activating virtual environment)**
```bash
.venv\Scripts\activate  # Activate first
uvicorn app.main:app --reload
```

**Production mode:**
```bash
uv run fastapi run app/main.py
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🧪 Development

### Code Quality

**Check code quality:**
```bash
uv run ruff check
```

**Auto-fix import sorting:**
```bash
uv run ruff check --select I --fix
```

**Format code:**
```bash
uv run ruff format app/
```

**Run all checks (recommended before committing):**
```bash
uv run ruff check --select I --fix
uv run ruff format app/
uv run ruff check
```

### Training DRL Models

Pre-trained models are included, but you can train custom models:

```bash
# Train Junior model (25-30 customers)
uv run python -m app.scripts.drl.train_junior

# Train Mid model (50-60 customers)
uv run python -m app.scripts.drl.train_mid

# Train Expert model (100-120 customers)
uv run python -m app.scripts.drl.train_expert
```

Trained models are saved in `app/core/drl/checkpoints/`

## 📡 API Endpoints

### Instances
- `POST /instances/generate/random` - Generate random CVRP instance
- `POST /instances/generate/clustered` - Generate clustered CVRP instance
- `POST /instances/upload` - Upload custom instance (VRP format) *No tested yet*
- `GET /instances` - List all instances
- `GET /instances/{id}` - Get specific instance
- `DELETE /instances/{id}` - Delete instance

### Solving
- `POST /solve/comparision` - Run both algorithms in parallel

### DRL Models
- `GET /drl/models` - List available DRL models

## 🛠️ Technologies

- **FastAPI** - Modern async web framework
- **PyTorch** - Deep learning for DRL models
- **NumPy** - Numerical computations
- **Pydantic** - Data validation and settings management
- **Uvicorn** - ASGI server
- **VRPLib** - VRP file format support

## 📄 License

![License](https://img.shields.io/badge/License-MIT-yellow)

## 👥 Authors

Natalia Andrea García Ríos
natalia.garcia9@udea.edu.co
ngarciarios2001@gmail.com

## 🙏 Acknowledgments

- DRL+GA inspired by Lu, Y. et al. (2025) [https://doi.org/10.3390/math13040545](https://doi.org/10.3390/math13040545)
- Genetic Algorithm implementation based on classical GA literature
