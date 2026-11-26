# How to Run the Project

This guide provides step-by-step instructions for setting up the environment, generating data, training models, running automated tests, and launching the Streamlit web application.

---

## 1. Prerequisites

- Python 3.10, 3.11, 3.12, 3.13, or 3.14.
- Windows PowerShell, Command Prompt, or Linux/macOS Terminal.
- Internet connection (initial pip dependency installation only).

---

## 2. Environment Setup

### Step A: Clone or Navigate to the Project Root
```bash
cd d:\College\Dynamic-Inventory-Replenishment
```

### Step B: (Optional but Recommended) Create a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step C: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 3. Step-by-Step Pipeline Execution

Each step can be executed independently from the command line:

### Step 1: Generate the Synthetic Dataset
```bash
python src/generate_data.py
```
*Expected Output*: Generates `data/inventory_data.csv` with 800 valid rows and prints descriptive statistics.

### Step 2: Run Exploratory Data Analysis (EDA)
```bash
python src/eda.py
```
*Expected Output*: Computes summary statistics and saves 6 high-resolution charts to `outputs/eda/`.

### Step 3: Train & Compare Machine Learning Models
```bash
python src/train_model.py
```
*Expected Output*:
- Evaluates Linear Regression and Random Forest Regressor on chronological test data.
- Saves `outputs/model_comparison.csv` and `outputs/model_comparison.png`.
- Serializes the winning pipeline to `models/best_model.joblib`.

### Step 4: Run the Supply Chain Simulation
```bash
python src/simulation.py
```
*Expected Output*:
- Simulates 40 test days comparing Traditional vs. ML-Based policies.
- Saves `outputs/simulation_results.csv` and `outputs/simulation_comparison.png`.

### Step 5: Run Automated Unit Tests
```bash
pytest
```
*Expected Output*: All 9 tests pass in under 3 seconds.

---

## 4. Launching the Interactive Streamlit Application

Run the Streamlit application:
```bash
streamlit run app.py
```

- Streamlit will launch a local web server (usually at `http://localhost:8501`).
- If your default browser does not open automatically, open your browser and navigate to:
  ```
  http://localhost:8501
  ```

---

## 5. Troubleshooting Common Issues

### Issue 1: `ModuleNotFoundError: No module named 'src'`
- **Cause**: Running a script from inside `src/` without project root in `PYTHONPATH`.
- **Fix**: All scripts have internal `sys.path.insert(0, ...)` handlers, but always run commands from the project root directory (`d:\College\Dynamic-Inventory-Replenishment`).

### Issue 2: `PermissionError: [WinError 10013]` on Streamlit Launch
- **Cause**: Port 8501 is already occupied by another application or blocked by sandbox permissions.
- **Fix**: Run on a different port:
  ```bash
  streamlit run app.py --server.port 8502
  ```

### Issue 3: Matplotlib Font or Cache Warnings
- **Cause**: System temporary cache directory permission restrictions on Windows.
- **Fix**: The scripts automatically configure `os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()`, suppressing cache conflicts.
