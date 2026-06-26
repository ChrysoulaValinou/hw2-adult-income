import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
import os

def load_and_split_data(filepath='data/dataset.csv'):
    """
    Task 2.1 Φορτώνει το dataset και το χωρίζει σε Train (80%), Validation (10%) και Test (10%).
    Η διαδικασία γίνεται με stratified split βάσει της μεταβλητής-στόχου.
    """
    df = pd.read_csv(filepath)
    
    X = df.drop('income', axis=1)
    y = df['income']
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=0.10, 
        random_state=42, 
        stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=(1/9), 
        random_state=42, 
        stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def handle_missing_values(X_train, X_val, X_test):
    """
    Task 2.2: Διαχείριση ελλειπουσών τιμών. 
    Υπολογισμός στατιστικών μόνο στο X_train και εφαρμογή σε όλα τα sets.
    """
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    categorical_cols = ['workclass', 'occupation', 'native_country']

    for col in categorical_cols:
        mode_value = X_train[col].mode()[0]
        
        X_train[col] = X_train[col].fillna(mode_value)
        X_val[col] = X_val[col].fillna(mode_value)
        X_test[col] = X_test[col].fillna(mode_value)

    return X_train, X_val, X_test

def handle_outliers(X_train, X_val, X_test):
    """
    Task 2.3: Ανίχνευση και διαχείριση outliers με τη μέθοδο IQR.
    Τα όρια υπολογίζονται μόνο στο X_train και εφαρμόζονται (cap) σε όλα τα sets.
    Εξαιρούνται τα capital_gain και capital_loss λόγω sparsity, 
    ώστε να μην μηδενίζεται η χρήσιμη πληροφορία τους.
    """
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    
    # Η ΔΙΟΡΘΩΣΗ: Κρατάμε μόνο ηλικία και ώρες εργασίας
    numerical_cols = ['age', 'hours_per_week']
    
    for col in numerical_cols:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        X_train[col] = X_train[col].clip(lower=lower_bound, upper=upper_bound)
        X_val[col] = X_val[col].clip(lower=lower_bound, upper=upper_bound)
        X_test[col] = X_test[col].clip(lower=lower_bound, upper=upper_bound)
        
    return X_train, X_val, X_test

def encode_features(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Task 2.4: Μετατροπή κατηγορικών μεταβλητών σε αριθμητικές.
    """
    X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()

    le_sex = LabelEncoder()
    X_train['sex'] = le_sex.fit_transform(X_train['sex'])
    X_val['sex'] = le_sex.transform(X_val['sex'])
    X_test['sex'] = le_sex.transform(X_test['sex'])

    le_y = LabelEncoder()
    y_train = le_y.fit_transform(y_train)
    y_val = le_y.transform(y_val)
    y_test = le_y.transform(y_test)

    cat_cols = ['workclass', 'education', 'marital_status', 'occupation', 
                'relationship', 'race', 'native_country']
    
    X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_val = pd.get_dummies(X_val, columns=cat_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

    X_train, X_val = X_train.align(X_val, join='left', axis=1, fill_value=0)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    return X_train, X_val, X_test, y_train, y_val, y_test

def add_features(X_train, X_val, X_test):
    """
    Task 2.5: Δημιουργία 2 νέων features.
    """
    for df in [X_train, X_val, X_test]:
        df['capital_net'] = df['capital_gain'] - df['capital_loss']
        df['work_index'] = df['age'] * df['hours_per_week']
        
    return X_train, X_val, X_test

def scale_features(X_train, X_val, X_test):
    """
    Task 2.6: Εφαρμογή StandardScaler.
    Fit μόνο στο train, transform σε όλα τα sets.
    """
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return X_train, X_val, X_test

def run_pca_exploration(X_train, y_train):
    """
    Task 2.7: Exploratory PCA analysis.
    Αποθηκεύει το Scree Plot και το 2D Scatter Plot ως αρχεία εικόνας.
    """
    pca = PCA()
    X_pca = pca.fit_transform(X_train)
    
    plt.figure(figsize=(10, 5))
    exp_var_cum = np.cumsum(pca.explained_variance_ratio_)
    plt.bar(range(1, len(exp_var_cum)+1), pca.explained_variance_ratio_, alpha=0.5, align='center', label='Individual variance')
    plt.step(range(1, len(exp_var_cum)+1), exp_var_cum, where='mid', label='Cumulative variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.title('Scree Plot')
    plt.savefig('pca_scree_plot.png') 
    plt.close()

    loadings = pd.DataFrame(
        pca.components_.T, 
        columns=[f'PC{i+1}' for i in range(len(X_train.columns))],
        index=X_train.columns
    )

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_train, alpha=0.3)
    plt.title('PCA 2D Projection')
    plt.xlabel('First Principal Component (PC1)')
    plt.ylabel('Second Principal Component (PC2)')
    plt.savefig('pca_2d_projection.png') 
    plt.close()


# --- Main Execution Block (Διορθωμένη Σειρά) ---
if __name__ == "__main__":
    # Task 2.1
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()
    
    # Task 2.2
    X_train, X_val, X_test = handle_missing_values(X_train, X_val, X_test)
    
    # Task 2.3
    X_train, X_val, X_test = handle_outliers(X_train, X_val, X_test)
    
    # Task 2.4 (Encoding ΠΡΩΤΑ)
    X_train, X_val, X_test, y_train, y_val, y_test = encode_features(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Task 2.5 (Feature Engineering ΜΕΤΑ το Encoding)
    X_train, X_val, X_test = add_features(X_train, X_val, X_test)
    
    # Task 2.6
    X_train, X_val, X_test = scale_features(X_train, X_val, X_test)
    
    # Task 2.7
    print("\nStarting PCA exploration...")
    run_pca_exploration(X_train, y_train)
    print("\nΌλα τα βήματα προεπεξεργασίας ολοκληρώθηκαν με τη σωστή σειρά!")

# --- Συνάρτηση για το Homework 2 (Inference) ---
def apply_preprocessing(df):
    """
    Αυτή η συνάρτηση καλείται από το `tools.py` στο Homework 2.
    Εφαρμόζει Encoding -> Feature Engineering και ΕΥΘΥΓΡΑΜΜΙΖΕΙ τις στήλες.
    """
    # 1. Encoding (Απλοποιημένο για inference 1 γραμμής)
    if 'sex' in df.columns:
        df['sex'] = df['sex'].map({'Male': 1, 'Female': 0}).fillna(0) 
    
    cat_cols = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'native_country']
    df = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns])
    
    # 2. Feature Engineering (Μετά το Encoding, όπως ζήτησε ο καθηγητής)
    if all(col in df.columns for col in ['capital_gain', 'capital_loss']):
        df['capital_net'] = df['capital_gain'] - df['capital_loss']
    if all(col in df.columns for col in ['age', 'hours_per_week']):
        df['work_index'] = df['age'] * df['hours_per_week']
        
    # 3. ΕΥΘΥΓΡΑΜΜΙΣΗ ΣΤΗΛΩΝ (Το "μαγικό" βήμα για να μη σκάσει το μοντέλο)
    # Διαβάζουμε ποιες ακριβώς στήλες περιμένει ο scaler από το HW1
    try:
        scaler = joblib.load("models/scaler.pkl")
        expected_cols = scaler.feature_names_in_
        # Κρατάμε μόνο τις αναμενόμενες στήλες. Όποια κατηγορία (π.χ. άλλο επάγγελμα) λείπει, παίρνει 0.
        df = df.reindex(columns=expected_cols, fill_value=0)
    except Exception as e:
        print(f"Προσοχή κατά την ευθυγράμμιση στηλών: {e}")
        
    return df