import tkinter as tk
from tkinter import messagebox
import numpy as np
import pickle

# Load model and scaler
with open('heart_attack_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Predict function
def predict():
    try:
        age = float(entry_age.get())
        sex = int(entry_sex.get())
        cp = int(entry_cp.get())
        trtbps = float(entry_trtbps.get())
        chol = float(entry_chol.get())
        fbs = int(entry_fbs.get())
        restecg = int(entry_restecg.get())
        thalachh = float(entry_thalachh.get())
        exng = int(entry_exng.get())
        oldpeak = float(entry_oldpeak.get())
        slp = int(entry_slp.get())
        caa = int(entry_caa.get())
        thall = int(entry_thall.get())

        X_new = np.array([[age, sex, cp, trtbps, chol, fbs, restecg,
                           thalachh, exng, oldpeak, slp, caa, thall]])

        X_scaled = scaler.transform(X_new)
        result = model.predict(X_scaled)

        if result[0] == 1:
            messagebox.showinfo("Result", "⚠️ Prone to Heart Attack")
        else:
            messagebox.showinfo("Result", "✅ Not Prone to Heart Attack")

    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

# GUI setup
root = tk.Tk()
root.title("Heart Attack Predictor")

labels = [
    "Age", "Sex (1=Male, 0=Female)", "Chest Pain Type (0-3)",
    "Resting BP", "Cholesterol", "Fasting Blood Sugar (1=True, 0=False)",
    "Rest ECG (0-2)", "Max Heart Rate", "Exercise Induced Angina (1/0)",
    "Oldpeak", "Slope (0-2)", "Number of Major Vessels (0-3)", "Thal (0-2)"
]

entries = []

for idx, label in enumerate(labels):
    tk.Label(root, text=label).grid(row=idx, column=0, padx=5, pady=5, sticky='e')
    entry = tk.Entry(root)
    entry.grid(row=idx, column=1, padx=5, pady=5)
    entries.append(entry)

(entry_age, entry_sex, entry_cp, entry_trtbps, entry_chol,
 entry_fbs, entry_restecg, entry_thalachh, entry_exng,
 entry_oldpeak, entry_slp, entry_caa, entry_thall) = entries

tk.Button(root, text="Predict", command=predict).grid(
    row=len(labels), column=0, columnspan=2, pady=10
)

root.mainloop()
