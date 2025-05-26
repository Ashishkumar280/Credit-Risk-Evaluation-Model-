import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# ------------------ Step 1: Define Features ------------------ #
numeric_features = [
    "person_age", "person_income", "person_emp_length",
    "loan_amnt", "loan_int_rate", "loan_percent_income",
    "cb_person_cred_hist_length"
]
categorical_features = [
    "person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"
]

# ------------------ Step 2: Preprocessor ------------------ #
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ------------------ Step 3: Dummy Training Data ------------------ #
data = {
    "person_age": [25, 40, 35],
    "person_income": [50000, 80000, 60000],
    "person_emp_length": [2.0, 10.0, 5.0],
    "loan_amnt": [10000, 20000, 15000],
    "loan_int_rate": [10.5, 7.2, 12.3],
    "loan_percent_income": [0.2, 0.15, 0.18],
    "cb_person_cred_hist_length": [5, 15, 8],
    "person_home_ownership": ["RENT", "OWN", "MORTGAGE"],
    "loan_intent": ["PERSONAL", "EDUCATION", "VENTURE"],
    "loan_grade": ["A", "B", "C"],
    "cb_person_default_on_file": ["N", "Y", "N"]
}
X_train = pd.DataFrame(data)
y_train = np.array([0, 1, 0])  # 0 = low risk, 1 = high risk

# ------------------ Step 4: Final Model ------------------ #
final_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
final_model.fit(X_train, y_train)

# ------------------ Step 5: GUI ------------------ #
def predict_risk():
    try:
        user_data = {
            "person_age": int(entry_age.get()),
            "person_income": int(entry_income.get()),
            "person_emp_length": float(entry_emp_length.get()),
            "loan_amnt": int(entry_loan_amount.get()),
            "loan_int_rate": float(entry_interest_rate.get()),
            "loan_percent_income": float(entry_loan_percent_income.get()),
            "cb_person_cred_hist_length": int(entry_credit_history.get()),
            "person_home_ownership": ownership_var.get(),
            "loan_intent": intent_var.get(),
            "loan_grade": grade_var.get(),
            "cb_person_default_on_file": default_var.get()
        }
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numeric values.")
        return

    user_df = pd.DataFrame([user_data])
    risk_prediction = final_model.predict(user_df)[0]
    risk_probability = final_model.predict_proba(user_df)[0][1]

    if risk_prediction == 1:
        result = f"⚠ High Credit Risk\nProbability: {risk_probability:.2f}"
        result_label.config(text=result, foreground="red")
    else:
        result = f"✅ Low Credit Risk\nProbability: {1 - risk_probability:.2f}"
        result_label.config(text=result, foreground="green")

def cancel():
    root.destroy()

root = tk.Tk()
root.title("Credit Risk Prediction")
root.geometry("400x600")
root.configure(bg='#f9f9f9')

main_frame = ttk.Frame(root, padding="20 20 20 20")
main_frame.pack(fill='both', expand=True)

fields = [
    ("Age", "entry_age"),
    ("Income", "entry_income"),
    ("Employment Length (years)", "entry_emp_length"),
    ("Loan Amount", "entry_loan_amount"),
    ("Interest Rate", "entry_interest_rate"),
    ("Loan % of Income", "entry_loan_percent_income"),
    ("Credit History Length", "entry_credit_history")
]
entries = {}
for idx, (label, var_name) in enumerate(fields):
    ttk.Label(main_frame, text=label + ":").grid(row=idx, column=0, sticky='w', pady=6)
    entries[var_name] = ttk.Entry(main_frame, width=22)
    entries[var_name].grid(row=idx, column=1, pady=6)

entry_age = entries["entry_age"]
entry_income = entries["entry_income"]
entry_emp_length = entries["entry_emp_length"]
entry_loan_amount = entries["entry_loan_amount"]
entry_interest_rate = entries["entry_interest_rate"]
entry_loan_percent_income = entries["entry_loan_percent_income"]
entry_credit_history = entries["entry_credit_history"]

ownership_var = tk.StringVar(value="RENT")
intent_var = tk.StringVar(value="PERSONAL")
grade_var = tk.StringVar(value="A")
default_var = tk.StringVar(value="N")

ttk.Label(main_frame, text="Home Ownership:").grid(row=7, column=0, sticky='w', pady=6)
ttk.OptionMenu(main_frame, ownership_var, ownership_var.get(), "RENT", "OWN", "MORTGAGE", "OTHER").grid(row=7, column=1)

ttk.Label(main_frame, text="Loan Intent:").grid(row=8, column=0, sticky='w', pady=6)
ttk.OptionMenu(main_frame, intent_var, intent_var.get(), "PERSONAL", "EDUCATION", "VENTURE", "MEDICAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT").grid(row=8, column=1)

ttk.Label(main_frame, text="Loan Grade:").grid(row=9, column=0, sticky='w', pady=6)
ttk.OptionMenu(main_frame, grade_var, grade_var.get(), *list("ABCDEFG")).grid(row=9, column=1)

ttk.Label(main_frame, text="Default on File (Y/N):").grid(row=10, column=0, sticky='w', pady=6)
ttk.OptionMenu(main_frame, default_var, default_var.get(), "Y", "N").grid(row=10, column=1)

submit_btn = ttk.Button(main_frame, text="Predict Risk", command=predict_risk)
submit_btn.grid(row=11, column=0, pady=20)
cancel_btn = ttk.Button(main_frame, text="Cancel", command=cancel)
cancel_btn.grid(row=11, column=1, pady=20)

result_label = ttk.Label(main_frame, text="", font=("Arial", 12, "bold"))
result_label.grid(row=12, column=0, columnspan=2, pady=10)

root.mainloop()
