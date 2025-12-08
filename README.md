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
├── api/                       # REST API endpoints
│   └── routers/
│       ├── drl/               # DRL model handlers
│       ├── instances/         # Instance handlers
│       └── solve/             # Solving handlers
├── core/                      # Core algorithms
│   ├── drl/                   # Deep RL implementation (Pointer Networks)
│   ├── experiments/           # Experimental framework (runners, collectors)
│   ├── ga/                    # Genetic Algorithm implementation
│   ├── operations/            # Business logic
│   └── utils/                 # Helper functions
├── data/
│   ├── experiment_instances/  # Experimental dataset (120 stratified instances)
│   ├── experiment_results/    # Experimental results and analysis
│   └── presets/               # Pre-configured CVRP instances
├── instances/                 # Stored user generated CVRP instances
├── schemas/                   # Pydantic data models
└── scripts/              
    ├── drl/                   # Training scripts for DRL models
    ├── experiments/           # Experimental execution and visualization
    └── presets/               # Script for generate the presets
```

## 🚀 Getting Started

### Prerequisites

- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip

### Installing Python and uv

#### Step 1: Install Python

If you don't have Python installed:

1. Visit [python.org/downloads](https://www.python.org/downloads/)
2. Download Python 3.13 or higher for your operating system
3. Run the installer and **check the box "Add Python to PATH"** (important!)
4. Follow the installation wizard

To verify Python is installed, open a terminal/command prompt and run:
```bash
python --version
```

#### Step 2: Install uv (Python Package Manager)

uv is a fast Python package manager that makes managing dependencies easier.

**On Windows:**
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**On macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To verify uv is installed:
```bash
uv --version
```

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

## 🧪 Experimental Validation

The project includes a complete experimental framework for systematic validation of the hybrid DRL-GA approach.

### Experimental Dataset

Pre-generated instances are located in `app/data/experiment_instances/`:

- **Junior Range**: 20-50 customers (4 sizes: 20, 30, 40, 50)
- **Mid Range**: 60-100 customers (4 sizes: 60, 75, 90, 100)  
- **Expert Range**: 110-150 customers (4 sizes: 110, 125, 140, 150)

Each size includes:
- 5 random distribution instances
- 5 clustered distribution instances
- **Total: 120 instances** (12 sizes × 10 instances)

### Running Experiments

#### 1. Generate New Experimental Instances (Optional)

If you want to regenerate the dataset with different seeds or parameters:

```bash
python -m app.scripts.experiments.generate_experiment_instances
```

This creates stratified synthetic instances with controlled characteristics for systematic evaluation.

#### 2. Run Experiments

Execute the complete experimental protocol:

```bash
# Full experiments (2,400 runs - takes several hours)
python -m app.scripts.experiments.run_experiments

# Quick test mode (80 runs - ~15 minutes)
python -m app.scripts.experiments.run_experiments --quick

# Run specific range only
python -m app.scripts.experiments.run_experiments --range junior
python -m app.scripts.experiments.run_experiments --range mid
python -m app.scripts.experiments.run_experiments --range expert

# Run specific configuration only
python -m app.scripts.experiments.run_experiments --config ga_pure
python -m app.scripts.experiments.run_experiments --config drl_junior_ga
python -m app.scripts.experiments.run_experiments --config drl_mid_ga
python -m app.scripts.experiments.run_experiments --config drl_expert_ga
```

**Configurations evaluated:**
- `ga_pure`: Pure Genetic Algorithm (random initialization)
- `drl_junior_ga`: DRL-Junior + GA hybrid
- `drl_mid_ga`: DRL-Mid + GA hybrid
- `drl_expert_ga`: DRL-Expert + GA hybrid

Results are saved in `app/data/experiment_results/experiments_YYYYMMDD_HHMMSS/`

#### 3. Visualize Results

Generate publication-ready figures and LaTeX tables:

```bash
# Visualize most recent experiment
python -m app.scripts.experiments.visualize_results

# Visualize specific experiment
python -m app.scripts.experiments.visualize_results app\data\experiment_results\experiments_20251207_212609
```

**Generated outputs:**
- `figure1_h1_validation.png` - H1: DRL improves solution quality
- `figure2_h2_specialization.png` - H2: Agent specialization analysis
- `figure3_convergence.png` - Convergence behavior by range
- `figure4_cost_benefit.png` - Computational cost-benefit analysis
- `latex_tables.tex` - Four publication-ready LaTeX tables
- `tables_summary.txt` - Text preview of tables

### Experimental Design

The experiments validate two main hypotheses:

**H1: DRL Initialization Improves Solution Quality**
- DRL-initialized populations consistently outperform random initialization
- Measured across all instance ranges and configurations

**H2: Specialized Agents Perform Best in Their Training Range**
- Junior agent excels in small instances (20-50 customers)
- Mid agent excels in medium instances (60-100 customers)
- Expert agent excels in large instances (110-150 customers)

Each instance-configuration pair is replicated 5 times with different seeds for statistical validity (2 replicates in quick mode).

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
