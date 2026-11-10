from ollama import chat
import pandas as pd
import numpy as np
import uuid
import warnings
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

import json
import re

warnings.filterwarnings("ignore")

def call_llm_fn(prompt: str) -> str:
    # Replace with your actual OpenAI/LLM client call.
    # Must return raw text from the model. This is just a placeholder (that allows you to run the code).
    response = chat(
        model = "qwen3:4b-instruct",
        messages = [{"role": "user", "content": prompt}],
        format = "json"
    )

    return response.message.content


    '''
    return """{
    "routes": ["diabetes", "heart"],
    "primary": "heart",
    "confidence": {"diabetes": 0.35, "heart": 0.72, "breast_cancer": 0.05},
    "notes": "Brief triage rationale." }""" 
    ''' 


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
            try:
                response = self.call_llm(prompt)
                #print(f"\nQwen response for patient {pid}:")
                #print(repr(response))
                parsed = self._extract_json(response)

                routes = parsed["routes"]
                primary = parsed["primary"]

                #handling invalid responses
                if not isinstance(routes, list):
                    raise ValueError("'routes' must be a list")

                if not isinstance(primary, str):
                    raise ValueError("'primary' must be a string")

            except (ValueError, KeyError, TypeError) as e:
                print(f"[{self.role}] Failed to triage patient {pid}: {e}")
                remaining.append(case) #try again if routes and primary are invalid
                continue


            valid_routes = []

            for route in routes:
                if route in ["diabetes", "breast_cancer", "heart"] and route not in valid_routes: #add route avoid duplicates
                    valid_routes.append(route)

            if not valid_routes: #if no valid route, record and continue
                print(f"[{self.role}] No valid routes for patient {pid}")

                self.model.triage_log.append({
                    "PatientID": pid,
                    "routes": [],
                    "primary": None,
                    "labels": payload["labels"]
                })

                continue

            if primary in valid_routes:
                primary_decision = primary
            else:
                primary_decision = valid_routes[0] #fallback if no valid primary

            for valid_route in valid_routes:
                task_id = f"task_{valid_route}_{pid}"
                task = {
                    "id": pid,
                    "description": f"Analyze {valid_route} data",
                    "data": payload["data"][valid_route],
                    "y_true": payload["labels"][valid_route],
                    "triage": {
                        "routes": valid_routes,
                        "primary": primary_decision,
                        "labels": payload["labels"]
                    }
                }
                self.model.context[task_id] = task

            # store triage decision in model context for evaluation
            self.model.triage_log.append({
                "PatientID": pid,
                "routes": valid_routes,
                "primary": primary_decision,
                "labels": payload["labels"]
                })

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
diabetes = joblib.load("data/test/diabetes_test.pkl")
X_d, y_d = diabetes["X"], diabetes["y"]

# Breast Cancer
breast = joblib.load("data/test/breast_cancer_test.pkl")
X_b, y_b = breast["X"], breast["y"]  # 0 = malignant, 1 = benign

# Heart
heart = joblib.load("data/test/heart_test.pkl")
X_h, y_h = heart["X"], heart["y"]

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

diseases = ["diabetes", "breast_cancer", "heart"] 

# Analyse the triage performance

triage_df = pd.DataFrame(model.triage_log)

print("\n========== LLM PLANNER EVALUATION ==========")

if triage_df.empty:
    print("No triage decisions were made.")
else:
    true_pathways = []
    predicted_pathways = []

    for disease in diseases:

        #breast cancer uses 0 = malignant, 1 = benign.
        if disease == "breast_cancer":
            y_true = triage_df["labels"].apply(
                lambda labels: int(labels[disease] == 0)) #int(true) malignant = 1, benign = 0
            
        else:
            y_true = triage_df["labels"].apply(
                lambda labels: int(labels[disease] == 1)) #else 1 means disease is present
            

        y_pred = triage_df["routes"].apply(
            lambda routes: int(disease in routes))

        true_pathways.append(y_true.to_numpy())
        predicted_pathways.append(y_pred.to_numpy())

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"\n--- {disease.upper()} ---")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1-score:  {f1:.3f}")

#evaluating random forest executors

results_df = pd.DataFrame(model.results)

print("\n========== EXECUTOR EVALUATION ==========")

for disease in diseases:

    if results_df.empty:
        print("No executor results were made")
        break

    disease_results = results_df[results_df["Disease"] == disease]

    if disease_results.empty:
        print(f"\nNo executor predictions for {disease}") #if planner selected no patways for this disease
        continue

    y_true = disease_results["y_true"]
    y_pred = disease_results["y_pred"]

    print(f"\n--- {disease.upper()} EXECUTOR ---")
    print(f"Patients evaluated: {len(disease_results)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")

    print(classification_report(y_true, y_pred, labels=[0, 1], zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])

    disp.plot(cmap="Blues")
    plt.title(f"Random Forest Executor - {disease}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.show()

print("\n========== SIMULATION SUMMARY ==========")
print(f"Patients triaged: {len(model.triage_log)}")
print(f"Executor predictions: {len(model.results)}")
print(f"Patients awaiting retry: {len(model.case_queue)}")



