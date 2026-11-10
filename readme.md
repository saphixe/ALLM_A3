## Setup (PowerShell)

Create an environment using Python 3.11 or 3.12 and activate it:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install project requirements:
```powershell
python.exe -m pip install -r requirements.txt
```

Train the classifiers on data:
```powershell
python.exe 0_train_models.py
```

This creates the classifiers in `model/` and held-out test sets in `data/test/`.

## Basic planner

```powershell
python.exe 1_basic_mcp.py
```

This prints the following results:
- `results/confusion_diabetes.png`
- `results/confusion_breast_cancer.png`
- `results/confusion_heart.png`
- `results/output.png`

## LLM planner

### Set up local LLM
1. Download and install Ollama on your OS (https://ollama.com/download). This installs the local model server.
2. Open Ollama application and leave it running.
3. Install the Python client into the project's virtual environment and download the model weights:

```powershell
python.exe -m pip install ollama
ollama pull qwen3:4b-instruct
```

### Run simulation

```powershell
python.exe 2_llm_mcp.py
```

This prints the final results.