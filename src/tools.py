import json
import joblib
import pandas as pd
from langchain.tools import tool
from src.preprocessing import apply_preprocessing 

# Φορτώνουμε τα μοντέλα μια φορά στην αρχή για να είναι γρήγορο το εργαλείο
try:
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    print("Το μοντέλο και ο scaler φορτώθηκαν με επιτυχία στο tools.py!")
except Exception as e:
    print(f"Σφάλμα κατά τη φόρτωση των αρχείων μοντέλου/scaler: {e}")

@tool
def predict_income(input_data: str) -> str:
    """
    Predicts whether a person makes over $50K a year. 
    Input must be a JSON string with demographic and employment fields:
    age, workclass, education, marital_status, occupation, race, sex, 
    capital_gain, capital_loss, hours_per_week, native_country.
    """
    try:
        # 1. Παίρνουμε το JSON string από τον Agent και το κάνουμε DataFrame (πίνακα 1 γραμμής)
        data_dict = json.loads(input_data)
        df = pd.DataFrame([data_dict])
        
        # 2. Εφαρμόζουμε την ΠΡΟΕΠΕΞΕΡΓΑΣΙΑ 
        # Εδώ καλείται η συνάρτηση που φτιάξαμε στο preprocessing.py η οποία 
        # κάνει Encoding, Feature Engineering και ΕΥΘΥΓΡΑΜΜΙΣΗ των στηλών.
        df_processed = apply_preprocessing(df)
        
        # 3. Εφαρμόζουμε το Scaling (Ο scaler δεν παραπονιέται πλέον, γιατί οι στήλες είναι σωστές)
        X_scaled = scaler.transform(df_processed)
        
        # 4. Κάνουμε την Πρόβλεψη
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1] 
        
        # 5. Φτιάχνουμε την ανθρώπινη απάντηση για τον Agent
        label = ">50K" if prediction == 1 else "<=50K"
        prob_percent = round(probability * 100, 2)
        
        return f"Prediction: {label} (probability of earning >50K: {prob_percent}%)"

    except Exception as e:
        # Αν ο χρήστης ξεχάσει κάτι, λέμε στον Agent να του το ζητήσει ξανά
        return f"Error making prediction: {str(e)}. Please check the input values."
    
import ast
import operator

# Ένα ασφαλές λεξικό με τις επιτρεπόμενες μαθηματικές πράξεις
allowed_operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg
}

def safe_eval(node):
    """Μια ασφαλής συνάρτηση για την εκτέλεση μαθηματικών πράξεων."""
    if isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.BinOp):
        return allowed_operators[type(node.op)](safe_eval(node.left), safe_eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        return allowed_operators[type(node.op)](safe_eval(node.operand))
    else:
        raise TypeError(node)

@tool
def calculator(expression: str) -> str:
    """
    A calculator tool useful for numerical reasoning and math operations.
    Input must be a valid mathematical expression as a string (e.g., '55 * 52' or '15000 / 12').
    """
    try:
        # Αναλύουμε τη μαθηματική έκφραση με ασφάλεια
        node = ast.parse(expression, mode='eval').body
        result = safe_eval(node)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating expression: {e}"