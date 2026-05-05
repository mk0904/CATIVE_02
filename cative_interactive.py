import tkinter as tk
import pandas as pd
import numpy as np
import pickle
import os

class CATIVEApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CATIVE Interactive Predictor")
        self.root.geometry("1000x650")
        self.root.configure(bg="#0D1117")
        
        # Load Data
        print("Loading data and models...")
        self.df = pd.read_csv('data/df_processed.csv')
        self.X_full = pd.read_csv('data/X_full.csv').values
        
        self.svm = pickle.load(open('outputs/models/svm_model.pkl', 'rb'))
        self.xgb = pickle.load(open('outputs/models/xgboost_model.pkl', 'rb'))
        self.le = pickle.load(open('outputs/models/label_encoder.pkl', 'rb'))
        self.pca = pickle.load(open('outputs/models/pca_gmm.pkl', 'rb'))
        self.gmm = pickle.load(open('outputs/models/gmm_model.pkl', 'rb'))
        print("Done! Launching UI...")
        
        self.create_widgets()
        self.on_select(None)
        
    def create_widgets(self):
        # Top banner
        banner = tk.Frame(self.root, bg="#0D1117")
        banner.pack(fill=tk.X, pady=(20, 10), padx=20)
        tk.Label(banner, text="CATIVE Interactive Startup Explorer", 
                 font=("Helvetica", 20, "bold"), fg="#58A6FF", bg="#0D1117").pack(side=tk.LEFT)
        
        # Main layout
        main_pane = tk.Frame(self.root, bg="#0D1117")
        main_pane.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left side: List
        left_frame = tk.Frame(main_pane, width=300, bg="#0D1117")
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        tk.Label(left_frame, text="Select Startup:", font=("Helvetica", 14, "bold"), fg="#8B949E", bg="#0D1117").pack(anchor=tk.W, pady=(0, 10))
        
        # Generate display names for list
        company_names = []
        for i, row in self.df.iterrows():
            company_names.append(f"#{i+1} | {row['sector']} | {row['funding_stage']}")
            
        self.listbox = tk.Listbox(left_frame, bg="#161B22", fg="#E6EDF3", selectbackground="#58A6FF", 
                                  font=("Helvetica", 12), borderwidth=0, highlightthickness=1, highlightcolor="#30363D")
        self.listbox.pack(fill=tk.BOTH, expand=True)
        for name in company_names:
            self.listbox.insert(tk.END, name)
            
        self.listbox.bind('<<ListboxSelect>>', self.on_select)
        self.listbox.selection_set(0)
        
        # Right side: Details & Models
        right_frame = tk.Frame(main_pane, bg="#0D1117")
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Raw Data Box
        data_frame = tk.Frame(right_frame, bg="#161B22", bd=1, relief=tk.FLAT)
        data_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Grid for data
        self.lbl_sector = tk.Label(data_frame, text="-", font=("Helvetica", 14, "bold"), fg="#C586C0", bg="#161B22")
        self.lbl_roles = tk.Label(data_frame, text="-", font=("Helvetica", 14, "bold"), fg="#C586C0", bg="#161B22", wraplength=400, justify=tk.LEFT)
        self.lbl_funding = tk.Label(data_frame, text="-", font=("Helvetica", 14, "bold"), fg="#C586C0", bg="#161B22")
        self.lbl_cult = tk.Label(data_frame, text="-", font=("Helvetica", 14, "bold"), fg="#C586C0", bg="#161B22")
        
        tk.Label(data_frame, text="Sector / HQ:", font=("Helvetica", 13, "bold"), fg="#8B949E", bg="#161B22").grid(row=0, column=0, sticky=tk.W, padx=15, pady=(15,5))
        self.lbl_sector.grid(row=0, column=1, sticky=tk.W, pady=(15,5))
        
        tk.Label(data_frame, text="Hiring Roles:", font=("Helvetica", 13, "bold"), fg="#8B949E", bg="#161B22").grid(row=1, column=0, sticky=tk.W, padx=15, pady=5)
        self.lbl_roles.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        tk.Label(data_frame, text="Funding / Size:", font=("Helvetica", 13, "bold"), fg="#8B949E", bg="#161B22").grid(row=2, column=0, sticky=tk.W, padx=15, pady=5)
        self.lbl_funding.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        tk.Label(data_frame, text="Ratings (Cult/WLB):", font=("Helvetica", 13, "bold"), fg="#8B949E", bg="#161B22").grid(row=3, column=0, sticky=tk.W, padx=15, pady=(5,15))
        self.lbl_cult.grid(row=3, column=1, sticky=tk.W, pady=(5,15))
        
        # Predictions Box
        pred_frame = tk.Frame(right_frame, bg="#161B22")
        pred_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(pred_frame, text="Live Model Predictions", font=("Helvetica", 18, "bold"), fg="#58A6FF", bg="#161B22").pack(pady=15)
        
        grid_f = tk.Frame(pred_frame, bg="#161B22")
        grid_f.pack(pady=10)
        
        self.lbl_true = tk.Label(grid_f, text="-", font=("Helvetica", 18, "bold"), fg="#8B949E", bg="#161B22")
        self.lbl_svm = tk.Label(grid_f, text="-", font=("Helvetica", 18, "bold"), fg="#E07B54", bg="#161B22")
        self.lbl_xgb = tk.Label(grid_f, text="-", font=("Helvetica", 18, "bold"), fg="#5B8DB8", bg="#161B22")
        self.lbl_ent = tk.Label(grid_f, text="-", font=("Helvetica", 18, "bold"), fg="#C586C0", bg="#161B22")
        
        tk.Label(grid_f, text="Ground Truth:", font=("Helvetica", 13), fg="#E6EDF3", bg="#161B22").grid(row=0, column=0, padx=20, pady=10, sticky=tk.E)
        self.lbl_true.grid(row=0, column=1, sticky=tk.W)
        
        tk.Label(grid_f, text="SVM (Phase 1):", font=("Helvetica", 13), fg="#E6EDF3", bg="#161B22").grid(row=1, column=0, padx=20, pady=10, sticky=tk.E)
        self.lbl_svm.grid(row=1, column=1, sticky=tk.W)
        
        tk.Label(grid_f, text="XGBoost (Phase 1):", font=("Helvetica", 13), fg="#E6EDF3", bg="#161B22").grid(row=2, column=0, padx=20, pady=10, sticky=tk.E)
        self.lbl_xgb.grid(row=2, column=1, sticky=tk.W)
        
        tk.Label(grid_f, text="GMM Entropy (Phase 3 Gate):", font=("Helvetica", 13), fg="#E6EDF3", bg="#161B22").grid(row=3, column=0, padx=20, pady=10, sticky=tk.E)
        self.lbl_ent.grid(row=3, column=1, sticky=tk.W)
        
    def on_select(self, event):
        if not self.listbox.curselection(): return
        idx = self.listbox.curselection()[0]
        
        # Update raw data
        row = self.df.iloc[idx]
        self.lbl_sector.config(text=f"{row['sector']} | {row['hq_city']}")
        self.lbl_roles.config(text=f"{row['top_hiring_roles']}")
        self.lbl_funding.config(text=f"${row['total_funding_usd']:,.0f} | {row['employee_count']} emp.")
        self.lbl_cult.config(text=f"Culture: {row['culture_rating']} | WLB: {row['wlb_rating']} | Salary: {row['salary_rating']}")
        
        # Update Predictions
        x = self.X_full[idx].reshape(1, -1)
        true_label = row['hire_quality_label']
        
        # SVM Prediction
        svm_idx = self.svm.predict(x)[0]
        svm_pred = self.le.inverse_transform([svm_idx])[0]
        
        # XGB Prediction
        xgb_idx = self.xgb.predict(x)[0]
        xgb_pred = self.le.inverse_transform([xgb_idx])[0]
        
        # GMM Entropy
        x_pca = self.pca.transform(x)
        gmm_probs = self.gmm.predict_proba(x_pca)[0]
        entropy = -np.sum(gmm_probs * np.log(gmm_probs + 1e-9))
        
        def color_label(lbl, text_widget):
            if lbl == 'Emerging': text_widget.config(fg='#E07B54', text=lbl)
            elif lbl == 'Growing': text_widget.config(fg='#5B8DB8', text=lbl)
            else: text_widget.config(fg='#4CAF7D', text=lbl)
            
        color_label(true_label, self.lbl_true)
        color_label(svm_pred, self.lbl_svm)
        color_label(xgb_pred, self.lbl_xgb)
        
        ent_color = "#E07B54" if entropy > 0.8 else "#4CAF7D"
        self.lbl_ent.config(text=f"{entropy:.3f} (Max is 1.098)", fg=ent_color)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    root = tk.Tk()
    app = CATIVEApp(root)
    
    # macOS fix to bring window to front
    os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')
    
    root.mainloop()
