import pandas as pd
import numpy as np
import uuid
import warnings
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import json
import re

warnings.filterwarnings("ignore")

def call_llm_fn(prompt: str) -> str:
    # Replace with your actual OpenAI/LLM client call.
    # Must return raw text from the model. This is just a placeholder (that allows you to run the code).
    return """{
  "routes": ["diabetes", "heart"],
  "primary": "heart",
  "confidence": {"diabetes": 0.35, "heart": 0.72, "breast_cancer": 0.05},
  "notes": "Brief triage rationale."
}"""


def make_case_text(pid, d_row, b_row, h_row):
    # d_row, b_row, h_row are 1D arrays (same as you pass to models)
    # Keep it short; it’s triage, not a full report.
    return f"""
Patient {pid}. You are a triage assistant.
You must decide which diagnostic pathway(s) to run: diabetes, breast_cancer or heart.

Patient features:
age={h_row[0]}, sex={h_row[1]}, cp={h_row[2]}, trestbps={h_row[3]}, chol={h_row[4]},
fbs={h_row[5]}, rest ecg={h_row[6]}, thalach={h_row[7]}, exang={h_row[8]}, oldpeak={h_row[9]},
slope={h_row[10]}, ca={h_row[11]},
Pregnancies={d_row[0]}, Glucose={d_row[1]}, Blood Pressure={d_row[2]}, Skin Thickness={d_row[3]},
Insulin={d_row[4]}, BMI={d_row[5]}, Diabetes Pedigree Function={d_row[6]}.
Breast features: mean radius={b_row[0]:.3f}, mean texture={b_row[1]:.3f}, mean perimeter={b_row[2]:.3f}, mean area={b_row[3]:.3f}, mean smoothness={b_row[4]:.3f}.

Output JSON only.
Output format (example):
{{
  "routes": ["diabetes", "heart"],
  "primary": "heart",
  "confidence": {{"diabetes": 0.35, "heart": 0.72, "breast_cancer": 0.05}},
  "notes": "Brief triage rationale."
}}
""".strip()


# ==========================================
# STEP 1: Define Custom Scheduler (like RandomActivation)
# ==========================================

class CustomScheduler:
    def __init__(self, model):
        self.model = model
        self.agents = []

    def add(self, agent):
        self.agents.append(agent)

    def step(self):
        for agent in self.agents:
            agent.step()

# ==========================================
# STEP 2: Define Agent Classes
# ==========================================

class BaseAgent:
    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        self.model = model

class LLMPlannerAgent(BaseAgent):
    """
    An agent that uses an LLM to triage cases into diagnostic pathways.

    TODO: implement the step method for this class yourself.
    """
    def __init__(self, unique_id, model, call_llm_fn):
        super().__init__(unique_id, model)
        self.role = "llm_planner"
        self.call_llm = call_llm_fn

    def _extract_json(self, text):
        # robust-ish: find first {...} block
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            print("No json!")
            raise ValueError("No JSON object found in LLM output.")
        return json.loads(m.group(0))

    def step(self):
        if not self.model.case_queue:
            print("No queued cases to triage.")
            return

        remaining = []
        for case in self.model.case_queue:
            pid, payload = case  # payload is dict with data per domain + text
            prompt = payload["text"]

            # Call LLM and parse response

            # store triage decision in model context
            # self.model.triage_log.append({
            #         "PatientID": pid,
            #         "routes": routes,
            #         "primary": primary_decision,
            #         "labels": payload["labels"]
            #     })

            # Route to appropriate executors based on LLM decision

        self.model.case_queue = remaining

class PlannerAgent(BaseAgent):
    def __init__(self, unique_id, model, disease):
        super().__init__(unique_id, model)
        self.disease = disease
        self.role = f"planner_{disease}"

    def step(self):
        if self.model.case_queue:
            remaining = []
            for case in self.model.case_queue:
                pid, data, domain = case
                if domain == self.disease:
                    task_id = f"task_{self.disease}_{pid}"
                    task = {"id": pid, "description": f"Analyze {self.disease} data", "data": data}
                    self.model.context[task_id] = task
                    print(f"[{self.role}] Planned task {task_id}")
                else:
                    remaining.append(case)
            self.model.case_queue = remaining

class ExecutorAgent(BaseAgent):
    def __init__(self, unique_id, model, disease, clf):
        super().__init__(unique_id, model)
        self.disease = disease
        self.role = f"executor_{disease}"
        self.clf = clf

    def step(self):
        for key in list(self.model.context.keys()):
            if key.startswith(f"task_{self.disease}"):
                task = self.model.context.pop(key)
                pred = self.clf.predict([task["data"]])[0]
                result_id = f"result_{self.disease}_{task['id']}"
                self.model.context[result_id] = {
                    "task_id": task["id"],
                    "prediction": int(pred),
                    "agent": self.role
                }
                y_true = int(task.get("y_true", -1))
                y_pred = int(self.clf.predict([task["data"]])[0])

                self.model.results.append({
                    "PatientID": task["id"],
                    "Disease": self.disease,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "triage_primary": task.get("triage", {}).get("primary", None)
                })
                print(f"[{self.role}] Executed task {task['id']} → prediction: {pred}")

# ==========================================
# STEP 3: Define the Mesa Model
# ==========================================

class MCPMesaModel:
    def __init__(self, X_d, X_b, X_h, clf_d, clf_b, clf_h):
        self.schedule = CustomScheduler(self)
        self.context = {}
        self.case_queue = []
        self.results = []
        self.triage_log = []

        # Add cases to queue (patients)
        for _ in range(100):
            pid = str(uuid.uuid4().int)[:4]
            idx_d = X_d.sample(1).index[0]
            d_row = X_d.loc[idx_d].values
            y_d_true = int(y_d.loc[idx_d])

            idx_b = X_b.sample(1).index[0]
            b_row = X_b.loc[idx_b].values
            y_b_true = int(y_b[idx_b]) 

            idx_h = X_h.sample(1).index[0]
            h_row = X_h.loc[idx_h].values
            y_h_true = int(y_h.loc[idx_h])

            payload = {
                "text": make_case_text(pid, d_row, b_row, h_row),
                "data": {
                    "diabetes": d_row,
                    "breast_cancer": b_row,
                    "heart": h_row
                },
                "labels": {
                    "diabetes": y_d_true,
                    "breast_cancer": y_b_true,
                    "heart": y_h_true
                }
            }
            
            self.case_queue.append((pid, payload))

        # Add agents
        # Replaces 3 planners with 1 LLM planner
        self.schedule.add(LLMPlannerAgent("planner_llm", self, call_llm_fn))

        self.schedule.add(ExecutorAgent("executor_diabetes", self, "diabetes", clf_d))
        self.schedule.add(ExecutorAgent("executor_breast_cancer", self, "breast_cancer", clf_b))
        self.schedule.add(ExecutorAgent("executor_heart", self, "heart", clf_h))

    def step(self):
        self.schedule.step()


# ==========================================
# STEP 4: Run Simulation
# ==========================================

# Diabetes
diabetes = pd.read_csv("data/pima-indians-diabetes.data.csv",
                       names=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                              "DiabetesPedigreeFunction", "Age", "Outcome"])
X_d, y_d = diabetes.drop("Outcome", axis=1), diabetes["Outcome"]

# Breast Cancer
breast = load_breast_cancer()
X_b = pd.DataFrame(breast.data, columns=breast.feature_names)
y_b = breast.target  # 0 = malignant, 1 = benign

# Heart
heart = pd.read_csv("data/heart.csv")
X_h, y_h = heart.drop("target", axis=1), heart["target"]

# load models from files
clf_diabetes = joblib.load("model/diabetes.pkl")
clf_breast = joblib.load("model/breast.pkl")
clf_heart = joblib.load("model/heart.pkl")


model = MCPMesaModel(X_d, X_b, X_h, clf_diabetes, clf_breast, clf_heart)

for step in range(3):
    print(f"\n========== MESA STEP {step + 1} ==========")
    model.step()

# ==========================================
# STEP 5: Visualize Results with Seaborn
# ==========================================

# TODO: implement the analysis and visualization of results yourself. Below is just a starting point.

results_df = pd.DataFrame(model.results)
print("Results Collected:", len(results_df))
print(results_df.head())




# Analyse the triage performance

triage_df = pd.DataFrame(model.triage_log)
print("Triage Collected:", len(triage_df))
print(triage_df.head())
