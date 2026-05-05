import pandas as pd
import numpy as np
import pickle
import json
import os

print("Exporting data for dashboard...")
df = pd.read_csv('data/df_processed.csv')
X_full = pd.read_csv('data/X_full.csv').values

svm = pickle.load(open('outputs/models/svm_model.pkl', 'rb'))
xgb = pickle.load(open('outputs/models/xgboost_model.pkl', 'rb'))
le = pickle.load(open('outputs/models/label_encoder.pkl', 'rb'))
pca = pickle.load(open('outputs/models/pca_gmm.pkl', 'rb'))
gmm = pickle.load(open('outputs/models/gmm_model.pkl', 'rb'))

export_data = []

for i, row in df.iterrows():
    x = X_full[i].reshape(1, -1)
    
    svm_pred = le.inverse_transform([svm.predict(x)[0]])[0]
    xgb_pred = le.inverse_transform([xgb.predict(x)[0]])[0]
    
    x_pca = pca.transform(x)
    probs = gmm.predict_proba(x_pca)[0]
    entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
    
    export_data.append({
        "id": i,
        "name": f"Startup #{i+1} | {row['sector']} | {row['hq_city']}",
        "sector": str(row['sector']),
        "hq_city": str(row['hq_city']),
        "roles": str(row['top_hiring_roles']),
        "funding": float(row['total_funding_usd']),
        "employees": int(row['employee_count']),
        "culture": float(row['culture_rating']),
        "wlb": float(row['wlb_rating']),
        "salary": float(row['salary_rating']),
        "true_label": str(row['hire_quality_label']),
        "svm_pred": str(svm_pred),
        "xgb_pred": str(xgb_pred),
        "entropy": entropy
    })

# Export as a JS variable so we don't have local CORS issues
with open('dashboard/data.js', 'w') as f:
    f.write("const startupData = " + json.dumps(export_data, indent=2) + ";")

print("Successfully exported dashboard/data.js")
