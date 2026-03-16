"""
Analyse Prédictive — Application ML complète
Améliorations v4 :
  • Détection automatique des types de variables (numérique continue, catégorielle,
    date/temporelle, binaire, ordinale, identifiant)
  • Analyse adaptée par type (encodage, statistiques, visualisations spécifiques)
  • Base de données enrichie : persistance de tous les résultats, schémas de données,
    corrélations, paramètres de modèles → améliore les analyses suivantes
  • Module Prédiction Future : prédiction sur une période/plage future
    avec intervalles de confiance et visualisation
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sqlite3
import hashlib
import json
import pickle
import base64
import gzip
import io
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, shapiro
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════════
#  DÉTECTEUR DE TYPES DE VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════

class VariableTypeDetector:
    """Détecte automatiquement le type sémantique de chaque colonne."""

    TYPES = {
        'date':        '📅 Date/Temporelle',
        'binary':      '🔀 Binaire (0/1)',
        'identifier':  '🔑 Identifiant',
        'categorical': '🏷️  Catégorielle',
        'ordinal':     '📊 Ordinale',
        'continuous':  '📈 Continue',
        'integer':     '🔢 Entière',
        'text':        '📝 Texte libre',
    }

    DATE_KEYWORDS = ['date', 'time', 'year', 'month', 'day', 'annee', 'mois',
                     'jour', 'periode', 'semaine', 'trimestre', 'heure']
    ID_KEYWORDS   = ['id', 'code', 'ref', 'numero', 'num', 'no', 'index',
                     'identifiant', 'matricule', 'cle']
    ORDINAL_VALS  = [
        {'faible', 'moyen', 'fort', 'élevé', 'low', 'medium', 'high'},
        {'très faible', 'faible', 'moyen', 'fort', 'très fort'},
        {'mauvais', 'passable', 'bien', 'très bien', 'excellent'},
        {'débutant', 'intermédiaire', 'avancé', 'expert'},
        {'1er', '2ème', '3ème', '4ème', '5ème'},
    ]

    def detect(self, df: pd.DataFrame) -> dict:
        """Retourne un dict {col_name: type_key} pour toutes les colonnes."""
        result = {}
        for col in df.columns:
            result[col] = self._detect_column(df[col], col)
        return result

    def _detect_column(self, series: pd.Series, name: str) -> str:
        name_low = name.lower().replace('_', ' ').replace('-', ' ')

        # 1. Essai de parsing date
        if any(k in name_low for k in self.DATE_KEYWORDS):
            try:
                pd.to_datetime(series.dropna().head(20), infer_datetime_format=True)
                return 'date'
            except Exception:
                pass

        if series.dtype in ['datetime64[ns]', 'datetime64']:
            return 'date'

        # 2. Détection par dtype
        if series.dtype == object:
            try:
                pd.to_datetime(series.dropna().head(30), infer_datetime_format=True)
                return 'date'
            except Exception:
                pass

            unique_vals = set(str(v).strip().lower() for v in series.dropna().unique())
            n_unique    = len(unique_vals)
            n_total     = len(series.dropna())

            # Binaire texte
            if unique_vals <= {'0', '1', 'oui', 'non', 'yes', 'no',
                               'true', 'false', 'vrai', 'faux', 'm', 'f',
                               'male', 'female', 'homme', 'femme'}:
                return 'binary'

            # Ordinal
            for ord_set in self.ORDINAL_VALS:
                if unique_vals & ord_set:
                    return 'ordinal'

            # Identifiant (très haute cardinalité + keyword)
            if any(k in name_low for k in self.ID_KEYWORDS) and n_unique > 0.9 * n_total:
                return 'identifier'

            # Catégorielle vs texte libre
            if n_unique <= 20 or n_unique / max(n_total, 1) < 0.05:
                return 'categorical'
            return 'text'

        if pd.api.types.is_bool_dtype(series):
            return 'binary'

        if pd.api.types.is_integer_dtype(series):
            n_unique = series.nunique()
            if n_unique == 2:
                return 'binary'
            if any(k in name_low for k in self.ID_KEYWORDS) and n_unique > 0.9 * len(series.dropna()):
                return 'identifier'
            if n_unique <= 15:
                return 'ordinal'
            return 'integer'

        if pd.api.types.is_float_dtype(series):
            n_unique = series.nunique()
            if n_unique == 2:
                return 'binary'
            return 'continuous'

        return 'categorical'

    @staticmethod
    def get_analysis_advice(var_type: str, col_name: str) -> str:
        """Retourne un conseil d'analyse selon le type."""
        advice = {
            'date':        f"'{col_name}' est temporelle → peut être décomposée (année/mois/jour) ou utilisée comme axe X pour des séries temporelles.",
            'binary':      f"'{col_name}' est binaire → idéale pour des analyses de groupe (t-test, chi²) ou comme variable cible de classification.",
            'identifier':  f"'{col_name}' est un identifiant → à exclure des modèles prédictifs (pas d'information statistique).",
            'categorical': f"'{col_name}' est catégorielle → sera encodée (One-Hot ou Label) avant modélisation.",
            'ordinal':     f"'{col_name}' est ordinale → sera encodée avec préservation de l'ordre.",
            'continuous':  f"'{col_name}' est continue → directement utilisable dans la régression. Vérifiez la distribution (normalité, outliers).",
            'integer':     f"'{col_name}' est entière → utilisable en régression ou discrétisée pour classification.",
            'text':        f"'{col_name}' est du texte libre → non exploitable directement ; à encoder (TF-IDF) ou à exclure.",
        }
        return advice.get(var_type, f"Type inconnu pour '{col_name}'.")


# ═══════════════════════════════════════════════════════════════════════════════
#  GESTIONNAIRE DE BASE DE DONNÉES ENRICHI
# ═══════════════════════════════════════════════════════════════════════════════

class DatabaseManager:

    def __init__(self, db_name="app_data.db"):
        self.db_name = db_name
        self.init_database()

    def _conn(self):
        return sqlite3.connect(self.db_name)

    def init_database(self):
        conn = self._conn()
        c    = conn.cursor()

        # ── Utilisateurs ─────────────────────────────────────────────────
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT UNIQUE NOT NULL,
            password    TEXT NOT NULL,
            email       TEXT,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')

        # ── Jeux de données (métadonnées) ────────────────────────────────
        c.execute('''CREATE TABLE IF NOT EXISTS datasets (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER,
            filename        TEXT,
            n_rows          INTEGER,
            n_cols          INTEGER,
            column_names    TEXT,   -- JSON list
            column_types    TEXT,   -- JSON dict {col: type}
            column_stats    TEXT,   -- JSON dict {col: {mean,std,min,max,nunique}}
            correlation_matrix TEXT, -- JSON
            uploaded_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )''')

        # ── Analyses / Modèles ───────────────────────────────────────────
        c.execute('''CREATE TABLE IF NOT EXISTS analyses (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER,
            dataset_id      INTEGER,
            filename        TEXT,
            model_type      TEXT,
            target_col      TEXT,
            feature_cols    TEXT,   -- JSON list
            r2_score        REAL,
            mse             REAL,
            mae             REAL,
            rmse            REAL,
            cv_r2_mean      REAL,   -- cross-validation
            cv_r2_std       REAL,
            test_size       REAL,
            n_train         INTEGER,
            n_test          INTEGER,
            model_params    TEXT,   -- JSON
            feature_importances TEXT, -- JSON {col: importance}
            scaler_data     TEXT,   -- base64 gzip pickle of scaler
            model_data      TEXT,   -- base64 gzip pickle of model
            encoders_data   TEXT,   -- base64 gzip pickle of encoders
            additional_info TEXT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id)    REFERENCES users(id),
            FOREIGN KEY (dataset_id) REFERENCES datasets(id)
        )''')

        # ── Prédictions futures sauvegardées ────────────────────────────
        c.execute('''CREATE TABLE IF NOT EXISTS future_predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id     INTEGER,
            user_id         INTEGER,
            input_values    TEXT,   -- JSON
            predicted_value REAL,
            confidence_low  REAL,
            confidence_high REAL,
            prediction_label TEXT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (analysis_id) REFERENCES analyses(id)
        )''')

        conn.commit()
        conn.close()

    # ── Auth ──────────────────────────────────────────────────────────────
    def hash_password(self, pwd):
        return hashlib.sha256(pwd.encode()).hexdigest()

    def create_user(self, username, password, email=""):
        try:
            conn = self._conn()
            conn.execute('INSERT INTO users (username,password,email) VALUES (?,?,?)',
                         (username, self.hash_password(password), email))
            conn.commit(); conn.close()
            return True, "Utilisateur créé avec succès!"
        except sqlite3.IntegrityError:
            return False, "Nom d'utilisateur déjà existant!"
        except Exception as e:
            return False, f"Erreur: {e}"

    def authenticate_user(self, username, password):
        conn = self._conn()
        c    = conn.cursor()
        c.execute('SELECT id,username FROM users WHERE username=? AND password=?',
                  (username, self.hash_password(password)))
        user = c.fetchone(); conn.close()
        return user

    # ── Dataset ───────────────────────────────────────────────────────────
    def save_dataset(self, user_id, filename, df, col_types, correlation):
        stats = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                stats[col] = {
                    'mean':   round(float(df[col].mean()), 4),
                    'std':    round(float(df[col].std()),  4),
                    'min':    round(float(df[col].min()),  4),
                    'max':    round(float(df[col].max()),  4),
                    'median': round(float(df[col].median()), 4),
                    'nunique': int(df[col].nunique()),
                    'nulls':  int(df[col].isna().sum()),
                }
            else:
                vc = df[col].value_counts()
                stats[col] = {
                    'nunique': int(df[col].nunique()),
                    'nulls':   int(df[col].isna().sum()),
                    'top3':    vc.index[:3].tolist(),
                }
        corr_json = None
        if correlation is not None:
            corr_json = correlation.round(4).to_json()

        conn = self._conn()
        c    = conn.cursor()
        c.execute('''INSERT INTO datasets
            (user_id,filename,n_rows,n_cols,column_names,column_types,column_stats,correlation_matrix)
            VALUES (?,?,?,?,?,?,?,?)''',
            (user_id, filename, len(df), len(df.columns),
             json.dumps(list(df.columns)), json.dumps(col_types),
             json.dumps(stats), corr_json))
        dataset_id = c.lastrowid
        conn.commit(); conn.close()
        return dataset_id

    def get_past_datasets(self, user_id):
        conn = self._conn()
        c    = conn.cursor()
        c.execute('''SELECT id,filename,n_rows,n_cols,column_types,column_stats,uploaded_at
                     FROM datasets WHERE user_id=? ORDER BY uploaded_at DESC LIMIT 20''',
                  (user_id,))
        rows = c.fetchall(); conn.close()
        return rows

    # ── Analyses ──────────────────────────────────────────────────────────
    def _pickle_b64(self, obj):
        buf  = io.BytesIO()
        pickle.dump(obj, buf)
        return base64.b64encode(gzip.compress(buf.getvalue())).decode()

    def _unpickle_b64(self, s):
        if not s:
            return None
        return pickle.loads(gzip.decompress(base64.b64decode(s)))

    def save_analysis(self, user_id, dataset_id, filename, model_type,
                      target_col, feature_cols, r2, mse, mae, rmse,
                      cv_r2_mean, cv_r2_std, test_size, n_train, n_test,
                      model, scaler, encoders, feature_importances,
                      additional_info=None):
        fi_json  = json.dumps({k: round(float(v), 6)
                               for k, v in feature_importances.items()}) if feature_importances else None
        conn = self._conn()
        c    = conn.cursor()
        c.execute('''INSERT INTO analyses
            (user_id,dataset_id,filename,model_type,target_col,feature_cols,
             r2_score,mse,mae,rmse,cv_r2_mean,cv_r2_std,test_size,n_train,n_test,
             model_data,scaler_data,encoders_data,feature_importances,additional_info)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (user_id, dataset_id, filename, model_type, target_col,
             json.dumps(feature_cols), r2, mse, mae, rmse,
             cv_r2_mean, cv_r2_std, test_size, n_train, n_test,
             self._pickle_b64(model),
             self._pickle_b64(scaler),
             self._pickle_b64(encoders),
             fi_json, additional_info))
        analysis_id = c.lastrowid
        conn.commit(); conn.close()
        return analysis_id

    def get_best_model_for_target(self, user_id, target_col):
        """Retrouve le meilleur modèle sauvegardé pour une variable cible."""
        conn = self._conn()
        c    = conn.cursor()
        c.execute('''SELECT id,model_type,r2_score,feature_cols,model_data,
                            scaler_data,encoders_data,feature_importances
                     FROM analyses
                     WHERE user_id=? AND target_col=?
                     ORDER BY r2_score DESC LIMIT 1''',
                  (user_id, target_col))
        row = c.fetchone(); conn.close()
        return row

    def get_user_analyses(self, user_id):
        conn = self._conn()
        c    = conn.cursor()
        c.execute('''SELECT filename,model_type,target_col,r2_score,mse,mae,
                            cv_r2_mean,n_train,n_test,created_at
                     FROM analyses WHERE user_id=?
                     ORDER BY created_at DESC''', (user_id,))
        rows = c.fetchall(); conn.close()
        return rows

    def save_future_prediction(self, analysis_id, user_id, input_values,
                                predicted, ci_low, ci_high, label):
        conn = self._conn()
        conn.execute('''INSERT INTO future_predictions
            (analysis_id,user_id,input_values,predicted_value,
             confidence_low,confidence_high,prediction_label)
            VALUES (?,?,?,?,?,?,?)''',
            (analysis_id, user_id, json.dumps(input_values),
             predicted, ci_low, ci_high, label))
        conn.commit(); conn.close()

    def get_future_predictions(self, user_id, limit=50):
        conn = self._conn()
        c    = conn.cursor()
        c.execute('''SELECT fp.prediction_label, fp.predicted_value,
                            fp.confidence_low, fp.confidence_high,
                            fp.input_values, fp.created_at,
                            a.model_type, a.target_col
                     FROM future_predictions fp
                     JOIN analyses a ON fp.analysis_id = a.id
                     WHERE fp.user_id=?
                     ORDER BY fp.created_at DESC LIMIT ?''',
                  (user_id, limit))
        rows = c.fetchall(); conn.close()
        return rows

    def get_target_history(self, user_id, target_col):
        """Retourne l'historique des r2 pour améliorer progressivement les analyses."""
        conn = self._conn()
        c    = conn.cursor()
        c.execute('''SELECT model_type, r2_score, cv_r2_mean, created_at
                     FROM analyses WHERE user_id=? AND target_col=?
                     ORDER BY created_at''', (user_id, target_col))
        rows = c.fetchall(); conn.close()
        return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  PRÉPROCESSEUR INTELLIGENT
# ═══════════════════════════════════════════════════════════════════════════════

class SmartPreprocessor:
    """Encode les variables selon leur type détecté."""

    def __init__(self, col_types: dict):
        self.col_types   = col_types
        self.encoders    = {}
        self.scaler      = StandardScaler()
        self.date_col    = None
        self.fitted      = False

    def fit_transform(self, df: pd.DataFrame, target_col: str):
        feature_cols = [c for c in df.columns
                        if c != target_col
                        and self.col_types.get(c) not in ('identifier', 'text')]
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        X = self._encode(X, fit=True)
        X_scaled = self.scaler.fit_transform(X.select_dtypes(include=[np.number]))
        self.fitted = True
        return pd.DataFrame(X_scaled, columns=X.select_dtypes(include=[np.number]).columns), y, feature_cols

    def transform(self, df: pd.DataFrame):
        X = df.copy()
        X = self._encode(X, fit=False)
        return pd.DataFrame(
            self.scaler.transform(X.select_dtypes(include=[np.number])),
            columns=X.select_dtypes(include=[np.number]).columns
        )

    def _encode(self, X: pd.DataFrame, fit: bool):
        for col in list(X.columns):
            vtype = self.col_types.get(col, 'continuous')

            if vtype == 'date':
                try:
                    dt = pd.to_datetime(X[col], infer_datetime_format=True, errors='coerce')
                    X[col + '_year']    = dt.dt.year.fillna(0).astype(int)
                    X[col + '_month']   = dt.dt.month.fillna(0).astype(int)
                    X[col + '_dayofweek'] = dt.dt.dayofweek.fillna(0).astype(int)
                    X[col + '_dayofyear'] = dt.dt.dayofyear.fillna(0).astype(int)
                    self.date_col = col
                except Exception:
                    pass
                X.drop(columns=[col], inplace=True)

            elif vtype == 'binary':
                X[col] = pd.factorize(X[col].fillna('__NA__'))[0]

            elif vtype == 'categorical':
                if fit:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].fillna('__NA__').astype(str))
                    self.encoders[col] = le
                else:
                    le = self.encoders.get(col)
                    if le:
                        def safe_transform(v):
                            try:    return le.transform([str(v)])[0]
                            except: return -1
                        X[col] = X[col].fillna('__NA__').astype(str).apply(safe_transform)
                    else:
                        X[col] = 0

            elif vtype == 'ordinal':
                if fit:
                    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    X[col] = oe.fit_transform(X[[col]].fillna('__NA__').astype(str))
                    self.encoders[col] = oe
                else:
                    oe = self.encoders.get(col)
                    if oe:
                        X[col] = oe.transform(X[[col]].fillna('__NA__').astype(str))
                    else:
                        X[col] = 0

            else:  # continuous / integer
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(X[col].median() if pd.api.types.is_numeric_dtype(X[col]) else 0)

        return X


# ═══════════════════════════════════════════════════════════════════════════════
#  FENÊTRE DE CONNEXION
# ═══════════════════════════════════════════════════════════════════════════════

class LoginWindow:

    def __init__(self, root, db_manager):
        self.root       = root
        self.db_manager = db_manager
        self.root.title("Analyse Prédictive — Connexion")
        self.root.geometry("520x620")
        self.root.configure(bg='#1a252f')
        self._center()
        self._build()

    def _center(self):
        self.root.update_idletasks()
        w, h = 520, 620
        x = (self.root.winfo_screenwidth()  - w) // 2
        y = (self.root.winfo_screenheight() - h) // 2
        self.root.geometry(f'{w}x{h}+{x}+{y}')

    def _build(self):
        main = tk.Frame(self.root, bg='#1a252f')
        main.pack(expand=True, fill='both', padx=40, pady=30)

        tk.Label(main, text="🔬 Analyse Prédictive",
                 font=('Helvetica', 26, 'bold'),
                 bg='#1a252f', fg='#ecf0f1').pack(pady=(0, 6))
        tk.Label(main, text="ML · Détection de types · Prédiction future",
                 font=('Helvetica', 10), bg='#1a252f', fg='#7f8c8d').pack(pady=(0, 30))

        box = tk.Frame(main, bg='#2c3e50')
        box.pack(fill='both', expand=True, padx=10, pady=10)

        tk.Label(box, text="Connexion", font=('Helvetica', 16, 'bold'),
                 bg='#2c3e50', fg='#ecf0f1').pack(pady=(22, 24))

        for lbl, attr, show in [
            ("Nom d'utilisateur", 'username_entry', ''),
            ("Mot de passe",      'password_entry', '●'),
        ]:
            tk.Label(box, text=lbl, font=('Helvetica', 10),
                     bg='#2c3e50', fg='#bdc3c7').pack(pady=(8, 3))
            e = tk.Entry(box, font=('Helvetica', 12), show=show,
                         bg='#ecf0f1', fg='#2c3e50', relief='flat')
            e.pack(pady=(0, 12), ipadx=10, ipady=8, fill='x', padx=30)
            setattr(self, attr, e)

        btn_row = tk.Frame(box, bg='#2c3e50')
        btn_row.pack(pady=(10, 20))

        tk.Button(btn_row, text="Se connecter", command=self.login,
                  font=('Helvetica', 11, 'bold'), bg='#27ae60', fg='white',
                  relief='flat', padx=28, pady=10, cursor='hand2').pack(side='left', padx=6)
        tk.Button(btn_row, text="Créer un compte", command=self._register_window,
                  font=('Helvetica', 11), bg='#2980b9', fg='white',
                  relief='flat', padx=28, pady=10, cursor='hand2').pack(side='left', padx=6)

        self.password_entry.bind('<Return>', lambda e: self.login())

    def login(self):
        u = self.username_entry.get().strip()
        p = self.password_entry.get()
        if not u or not p:
            messagebox.showerror("Erreur", "Veuillez remplir tous les champs!"); return
        user = self.db_manager.authenticate_user(u, p)
        if user:
            self.root.destroy()
            r = tk.Tk()
            PredictiveAnalysisApp(r, self.db_manager, user[0], user[1])
            r.mainloop()
        else:
            messagebox.showerror("Erreur", "Identifiants incorrects!")
            self.password_entry.delete(0, tk.END)

    def _register_window(self):
        win = tk.Toplevel(self.root)
        win.title("Créer un compte"); win.geometry("420x480")
        win.configure(bg='#1a252f'); win.transient(self.root); win.grab_set()

        f = tk.Frame(win, bg='#2c3e50')
        f.pack(expand=True, fill='both', padx=28, pady=28)
        tk.Label(f, text="Nouveau compte", font=('Helvetica', 18, 'bold'),
                 bg='#2c3e50', fg='#ecf0f1').pack(pady=(18, 24))

        fields = {}
        for lbl, key, show in [
            ("Nom d'utilisateur", 'u', ''),
            ("Email (optionnel)", 'e', ''),
            ("Mot de passe",      'p', '●'),
            ("Confirmer",         'c', '●'),
        ]:
            tk.Label(f, text=lbl, bg='#2c3e50', fg='#bdc3c7',
                     font=('Helvetica', 10)).pack(pady=(8, 3))
            en = tk.Entry(f, font=('Helvetica', 12), show=show,
                          bg='#ecf0f1', fg='#2c3e50', relief='flat')
            en.pack(pady=(0, 10), ipadx=10, ipady=7, fill='x', padx=18)
            fields[key] = en

        def register():
            u = fields['u'].get().strip()
            e = fields['e'].get().strip()
            p = fields['p'].get()
            c = fields['c'].get()
            if not u or not p:
                messagebox.showerror("Erreur", "Nom d'utilisateur et mot de passe requis!"); return
            if p != c:
                messagebox.showerror("Erreur", "Mots de passe différents!"); return
            if len(p) < 4:
                messagebox.showerror("Erreur", "Mot de passe trop court (min 4 car.)!"); return
            ok, msg = self.db_manager.create_user(u, p, e)
            if ok:
                messagebox.showinfo("Succès", msg); win.destroy()
            else:
                messagebox.showerror("Erreur", msg)

        tk.Button(f, text="Créer le compte", command=register,
                  font=('Helvetica', 12, 'bold'), bg='#27ae60', fg='white',
                  relief='flat', padx=40, pady=12, cursor='hand2').pack(pady=18)


# ═══════════════════════════════════════════════════════════════════════════════
#  APPLICATION PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════════

class PredictiveAnalysisApp:

    def __init__(self, root, db_manager, user_id, username):
        self.root        = root
        self.db_manager  = db_manager
        self.user_id     = user_id
        self.username    = username

        # Données courantes
        self.df          = None
        self.col_types   = {}
        self.dataset_id  = None
        self.preprocessor = None

        # Modèle courant
        self.model        = None
        self.analysis_id  = None
        self.target_col   = None
        self.feature_cols = []
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.predictions = None

        # Canvases
        self._corr_canvas   = None
        self._evalue_canvas = None

        self.detector = VariableTypeDetector()

        self.root.title(f"Analyse Prédictive v4 — {username}")
        self.root.geometry("1440x860")
        self.root.configure(bg='#ecf0f1')
        self._build_ui()

    # ──────────────────────────────────────────────────────────────────────
    #  UI SHELL
    # ──────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self.root, bg='#1a252f', height=56)
        hdr.pack(fill='x')
        tk.Label(hdr, text=f"👤 {self.username}",
                 font=('Helvetica', 11, 'bold'),
                 bg='#1a252f', fg='#ecf0f1').pack(side='left', padx=20, pady=14)

        for txt, cmd, color in [
            ("🚪 Déconnexion", self.logout,       '#e74c3c'),
            ("📊 Historique",  self.show_history,  '#2980b9'),
            ("📁 Prédictions", self.show_saved_predictions, '#8e44ad'),
        ]:
            tk.Button(hdr, text=txt, command=cmd, font=('Helvetica', 10),
                      bg=color, fg='white', relief='flat', cursor='hand2',
                      padx=14, pady=5).pack(side='right', padx=5, pady=12)

        # Notebook
        nb = ttk.Notebook(self.root)
        nb.pack(fill='both', expand=True, padx=8, pady=8)
        self.notebook = nb

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background='#ecf0f1')
        style.configure('TNotebook.Tab', padding=[18, 9], font=('Helvetica', 10))

        tabs = [
            ('tab1', '📁 Données'),
            ('tab2', '🔍 Types & Exploration'),
            ('tab3', '🤖 Modélisation'),
            ('tab4', '📈 Résultats'),
            ('tab5', '📐 Analyse Stat.'),
            ('tab6', '🔮 Prédiction Future'),
        ]
        for attr, label in tabs:
            f = tk.Frame(nb, bg='white')
            nb.add(f, text=label)
            setattr(self, attr, f)

        self._build_tab1()
        self._build_tab2()
        self._build_tab3()
        self._build_tab4()
        self._build_tab5()
        self._build_tab6()

    # ──────────────────────────────────────────────────────────────────────
    #  TAB 1 — CHARGEMENT
    # ──────────────────────────────────────────────────────────────────────
    def _build_tab1(self):
        m = tk.Frame(self.tab1, bg='white')
        m.pack(fill='both', expand=True, padx=30, pady=24)

        tk.Label(m, text="📂 Importer vos données",
                 font=('Helvetica', 18, 'bold'), bg='white', fg='#2c3e50').pack(pady=(0, 16))

        info = tk.Frame(m, bg='#eaf4fb', relief='groove', bd=1)
        info.pack(fill='x', pady=12)
        for t in ["✅ Formats acceptés : Excel (.xlsx, .xls) et CSV (.csv)",
                  "✅ La dernière colonne numérique sera proposée comme variable cible",
                  "✅ Toutes les colonnes sont analysées et typées automatiquement",
                  "✅ Les analyses précédentes améliorent les nouvelles via la base de données"]:
            tk.Label(info, text=t, font=('Helvetica', 10), bg='#eaf4fb',
                     fg='#1a5276', anchor='w').pack(anchor='w', padx=20, pady=2)
        tk.Label(info, text="", bg='#eaf4fb').pack(pady=4)

        btn_row = tk.Frame(m, bg='white')
        btn_row.pack(pady=20)
        self.load_btn = tk.Button(btn_row, text="📂 Choisir un fichier",
                                  command=self.load_data,
                                  font=('Helvetica', 14, 'bold'),
                                  bg='#2980b9', fg='white', relief='flat',
                                  cursor='hand2', padx=36, pady=14)
        self.load_btn.pack(side='left', padx=8)

        tk.Button(btn_row, text="🕘 Recharger un fichier récent",
                  command=self._load_recent,
                  font=('Helvetica', 12), bg='#7f8c8d', fg='white',
                  relief='flat', cursor='hand2', padx=20, pady=14).pack(side='left', padx=8)

        self.info_frame = tk.Frame(m, bg='white')
        self.info_frame.pack(fill='both', expand=True, pady=12)

    def _load_recent(self):
        rows = self.db_manager.get_past_datasets(self.user_id)
        if not rows:
            messagebox.showinfo("Info", "Aucun fichier récent trouvé."); return
        win = tk.Toplevel(self.root)
        win.title("Fichiers récents"); win.geometry("780x400")
        win.configure(bg='white'); win.transient(self.root); win.grab_set()
        tk.Label(win, text="Sélectionnez un fichier récent",
                 font=('Helvetica', 14, 'bold'), bg='white', fg='#2c3e50').pack(pady=14)
        tree = ttk.Treeview(win, columns=('Fichier','Lignes','Colonnes','Date'),
                            show='headings', height=12)
        for col, w in [('Fichier',280),('Lignes',80),('Colonnes',80),('Date',180)]:
            tree.heading(col, text=col); tree.column(col, width=w, anchor='center')
        for row in rows:
            tree.insert('', 'end', values=(row[1], row[2], row[3], row[6]))
        tree.pack(fill='both', expand=True, padx=16, pady=8)
        tk.Button(win, text="Charger les métadonnées",
                  command=lambda: self._restore_dataset(rows, tree, win),
                  bg='#2980b9', fg='white', font=('Helvetica', 11, 'bold'),
                  relief='flat', padx=20, pady=8).pack(pady=10)

    def _restore_dataset(self, rows, tree, win):
        sel = tree.selection()
        if not sel:
            messagebox.showwarning("Attention", "Sélectionnez un fichier."); return
        idx = tree.index(sel[0])
        row = rows[idx]
        col_types = json.loads(row[4]) if row[4] else {}
        self.col_types = col_types
        messagebox.showinfo("Info",
            f"Métadonnées de '{row[1]}' rechargées.\n"
            f"Types de variables restaurés ({len(col_types)} colonnes).\n"
            "Veuillez recharger le fichier Excel pour modéliser.")
        win.destroy()

    # ──────────────────────────────────────────────────────────────────────
    #  TAB 2 — TYPES & EXPLORATION
    # ──────────────────────────────────────────────────────────────────────
    def _build_tab2(self):
        outer = tk.Frame(self.tab2, bg='white')
        outer.pack(fill='both', expand=True)
        canvas_w = tk.Canvas(outer, bg='white')
        vsb      = tk.Scrollbar(outer, orient='vertical', command=canvas_w.yview)
        self.explore_frame = tk.Frame(canvas_w, bg='white')
        self.explore_frame.bind('<Configure>',
            lambda e: canvas_w.configure(scrollregion=canvas_w.bbox('all')))
        canvas_w.create_window((0, 0), window=self.explore_frame, anchor='nw')
        canvas_w.configure(yscrollcommand=vsb.set)
        canvas_w.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')
        tk.Label(self.explore_frame,
                 text="Les types et statistiques s'affichent après le chargement.",
                 font=('Helvetica', 12), bg='white', fg='#7f8c8d').pack(pady=50)

    def _update_tab2(self):
        for w in self.explore_frame.winfo_children():
            w.destroy()

        m = self.explore_frame
        tk.Label(m, text="🔍 Types de Variables & Exploration",
                 font=('Helvetica', 18, 'bold'), bg='white', fg='#2c3e50').pack(pady=16)

        # ── Tableau des types ─────────────────────────────────────────────
        type_frame = tk.LabelFrame(m, text="📋 Détection automatique des types",
                                   font=('Helvetica', 12, 'bold'),
                                   bg='white', fg='#2c3e50')
        type_frame.pack(fill='x', padx=20, pady=10)

        cols_t = ('Colonne', 'Type détecté', 'Valeurs uniques', 'Nulls %', 'Conseil d\'analyse')
        hsb = tk.Scrollbar(type_frame, orient='horizontal')
        vsb = tk.Scrollbar(type_frame)
        hsb.pack(side='bottom', fill='x')
        vsb.pack(side='right',  fill='y')
        tree = ttk.Treeview(type_frame, columns=cols_t, show='headings',
                            xscrollcommand=hsb.set, yscrollcommand=vsb.set,
                            height=min(14, len(self.df.columns)))
        hsb.config(command=tree.xview); vsb.config(command=tree.yview)
        for c, w in zip(cols_t, [150, 160, 110, 80, 480]):
            tree.heading(c, text=c); tree.column(c, width=w, anchor='w')

        color_map = {
            'date': '#d6eaf8', 'binary': '#d5f5e3', 'categorical': '#fef9e7',
            'ordinal': '#fdebd0', 'continuous': '#f0f0f0', 'integer': '#f0f0f0',
            'identifier': '#fadbd8', 'text': '#f5eef8',
        }
        for col in self.df.columns:
            vtype  = self.col_types.get(col, 'continuous')
            label  = VariableTypeDetector.TYPES.get(vtype, vtype)
            nuniq  = self.df[col].nunique()
            null_p = round(self.df[col].isna().mean() * 100, 1)
            advice = VariableTypeDetector().get_analysis_advice(vtype, col)
            tag    = f'tag_{vtype}'
            tree.insert('', 'end', values=(col, label, nuniq, f"{null_p}%", advice),
                        tags=(tag,))
            tree.tag_configure(tag, background=color_map.get(vtype, '#ffffff'))
        tree.pack(fill='both', expand=True, padx=10, pady=10)

        # Légende
        leg = tk.Frame(type_frame, bg='white')
        leg.pack(fill='x', padx=10, pady=(0, 8))
        for vtype, label in list(VariableTypeDetector.TYPES.items())[:4]:
            tk.Label(leg, text=f"■ {label}", bg=color_map.get(vtype,'#fff'),
                     fg='#2c3e50', font=('Helvetica', 8), padx=6).pack(side='left', padx=4)

        # ── Stats descriptives ────────────────────────────────────────────
        num_df = self.df.select_dtypes(include=[np.number])
        if not num_df.empty:
            st_frame = tk.LabelFrame(m, text="📊 Statistiques descriptives (variables numériques)",
                                     font=('Helvetica', 12, 'bold'),
                                     bg='white', fg='#2c3e50')
            st_frame.pack(fill='x', padx=20, pady=10)
            desc = num_df.describe()
            hsb2 = tk.Scrollbar(st_frame, orient='horizontal')
            vsb2 = tk.Scrollbar(st_frame)
            hsb2.pack(side='bottom', fill='x')
            vsb2.pack(side='right', fill='y')
            t2 = ttk.Treeview(st_frame, columns=['Stat'] + list(desc.columns),
                               show='headings', xscrollcommand=hsb2.set,
                               yscrollcommand=vsb2.set, height=8)
            hsb2.config(command=t2.xview); vsb2.config(command=t2.yview)
            for c in ['Stat'] + list(desc.columns):
                t2.heading(c, text=c); t2.column(c, width=110, anchor='center')
            for idx, row in desc.iterrows():
                t2.insert('', 'end', values=[idx] + [f"{v:.3f}" for v in row])
            t2.pack(fill='both', expand=True, padx=10, pady=10)

        # ── Visualisation par type ─────────────────────────────────────────
        self._draw_type_charts(m)

        # ── Matrice de corrélation + interprétation ───────────────────────
        self._draw_correlation(m)

    def _draw_type_charts(self, parent):
        num_cols  = [c for c, t in self.col_types.items() if t in ('continuous','integer') and c in self.df.columns]
        cat_cols  = [c for c, t in self.col_types.items() if t in ('categorical','ordinal','binary') and c in self.df.columns]
        date_cols = [c for c, t in self.col_types.items() if t == 'date' and c in self.df.columns]

        if num_cols:
            frm = tk.LabelFrame(parent,
                                text="📈 Distributions — Variables numériques (histogrammes)",
                                font=('Helvetica', 12, 'bold'), bg='white', fg='#2c3e50')
            frm.pack(fill='x', padx=20, pady=8)
            n  = min(len(num_cols), 6)
            nc = min(n, 3)
            nr = (n + nc - 1) // nc
            fig, axes = plt.subplots(nr, nc, figsize=(14, 3.5 * nr))
            axes = np.array(axes).reshape(-1)
            for i, col in enumerate(num_cols[:n]):
                ax = axes[i]
                self.df[col].dropna().hist(ax=ax, bins=25, color='#3498db',
                                           edgecolor='white', alpha=0.8)
                ax.set_title(col, fontsize=9, fontweight='bold')
                ax.set_xlabel(''); ax.grid(True, alpha=0.2)
                mu, sd = self.df[col].mean(), self.df[col].std()
                ax.axvline(mu, color='red', linestyle='--', lw=1.5, label=f'μ={mu:.2f}')
                ax.legend(fontsize=7)
            for j in range(n, len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout()
            c = FigureCanvasTkAgg(fig, frm)
            c.draw(); c.get_tk_widget().pack(fill='x', padx=8, pady=8)

        if cat_cols:
            frm2 = tk.LabelFrame(parent,
                                 text="🏷️  Fréquences — Variables catégorielles / binaires / ordinales",
                                 font=('Helvetica', 12, 'bold'), bg='white', fg='#2c3e50')
            frm2.pack(fill='x', padx=20, pady=8)
            n  = min(len(cat_cols), 6)
            nc = min(n, 3)
            nr = (n + nc - 1) // nc
            fig2, axes2 = plt.subplots(nr, nc, figsize=(14, 3.5 * nr))
            axes2 = np.array(axes2).reshape(-1)
            palette = sns.color_palette("Set2", 10)
            for i, col in enumerate(cat_cols[:n]):
                ax = axes2[i]
                vc = self.df[col].value_counts().head(10)
                colors = palette[:len(vc)]
                vc.plot.bar(ax=ax, color=colors, edgecolor='white', alpha=0.85)
                ax.set_title(col, fontsize=9, fontweight='bold')
                ax.set_xlabel(''); ax.tick_params(axis='x', rotation=35)
                ax.grid(True, alpha=0.2, axis='y')
            for j in range(n, len(axes2)):
                axes2[j].set_visible(False)
            plt.tight_layout()
            c2 = FigureCanvasTkAgg(fig2, frm2)
            c2.draw(); c2.get_tk_widget().pack(fill='x', padx=8, pady=8)

        if date_cols:
            frm3 = tk.LabelFrame(parent, text="📅 Séries temporelles",
                                 font=('Helvetica', 12, 'bold'), bg='white', fg='#2c3e50')
            frm3.pack(fill='x', padx=20, pady=8)
            num_numeric = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if num_numeric:
                col_d = date_cols[0]
                col_v = num_numeric[-1]
                try:
                    fig3, ax3 = plt.subplots(figsize=(14, 3.5))
                    dt = pd.to_datetime(self.df[col_d], errors='coerce')
                    tmp = pd.DataFrame({'dt': dt, 'v': self.df[col_v]}).dropna().sort_values('dt')
                    ax3.plot(tmp['dt'], tmp['v'], color='#2980b9', lw=1.5, alpha=0.85)
                    ax3.fill_between(tmp['dt'], tmp['v'], alpha=0.15, color='#2980b9')
                    ax3.set_title(f"{col_v} dans le temps ({col_d})", fontsize=10, fontweight='bold')
                    ax3.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
                    fig3.autofmt_xdate(); ax3.grid(True, alpha=0.25)
                    plt.tight_layout()
                    c3 = FigureCanvasTkAgg(fig3, frm3)
                    c3.draw(); c3.get_tk_widget().pack(fill='x', padx=8, pady=8)
                except Exception:
                    pass

    @staticmethod
    def _corr_label(r):
        a = abs(r)
        if a >= 0.9: return "🔴", "Très forte",  "#922b21"
        if a >= 0.7: return "🟠", "Forte",       "#d35400"
        if a >= 0.5: return "🟡", "Modérée",     "#d4ac0d"
        if a >= 0.3: return "🟢", "Faible",      "#1e8449"
        return "⚪", "Très faible", "#717d7e"

    def _draw_correlation(self, parent):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return
        correlation = self.df[numeric_cols].corr()

        corr_outer = tk.LabelFrame(parent, text="🗺️ Matrice de Corrélation",
                                   font=('Helvetica', 12, 'bold'), bg='white', fg='#2c3e50')
        corr_outer.pack(fill='both', expand=True, padx=20, pady=10)

        n = len(numeric_cols)
        fig, ax = plt.subplots(figsize=(max(8, n), max(6, n - 1)))
        sns.heatmap(correlation, annot=True, cmap='RdYlGn', center=0,
                    fmt='.2f', square=True, ax=ax,
                    linewidths=0.5, linecolor='#ecf0f1',
                    cbar_kws={'shrink': 0.75, 'label': 'r'},
                    vmin=-1, vmax=1, annot_kws={'size': 9, 'weight': 'bold'})
        ax.set_title('Corrélation de Pearson', fontsize=13, fontweight='bold', pad=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        plt.tight_layout()
        cv = FigureCanvasTkAgg(fig, corr_outer)
        cv.draw(); cv.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=8)

        # Tableau interprétation
        cols_list = list(numeric_cols)
        interp_frame = tk.LabelFrame(parent,
                                     text="📝 Interprétation par paire",
                                     font=('Helvetica', 12, 'bold'), bg='white', fg='#2c3e50')
        interp_frame.pack(fill='x', padx=20, pady=8)

        icols = ['Variable 1','Variable 2','r','Force','Direction','Interprétation']
        vsb   = tk.Scrollbar(interp_frame)
        vsb.pack(side='right', fill='y')
        itree = ttk.Treeview(interp_frame, columns=icols, show='headings',
                             yscrollcommand=vsb.set,
                             height=min(14, len(cols_list)*(len(cols_list)-1)//2))
        vsb.config(command=itree.yview)
        for c, w in zip(icols, [130,130,70,110,100,460]):
            itree.heading(c, text=c); itree.column(c, width=w, anchor='center')

        tag_bg = {'Très forte':'#f9ebea','Forte':'#fef5e7',
                  'Modérée':'#fdfefe','Faible':'#eafaf1','Très faible':'#f2f3f4'}
        pairs  = []
        for i in range(len(cols_list)):
            for j in range(i+1, len(cols_list)):
                c1, c2 = cols_list[i], cols_list[j]
                r      = correlation.loc[c1, c2]
                emoji, force, _ = self._corr_label(r)
                direction = "Positive ↗" if r > 0 else "Négative ↘"
                abs_r = abs(r)
                if abs_r >= 0.9:
                    detail = (f"Relation quasi-linéaire — risque de multicolinéarité si utilisées ensemble.")
                elif abs_r >= 0.7:
                    detail = f"Forte dépendance {'directe' if r>0 else 'inverse'} — bon prédicteur réciproque."
                elif abs_r >= 0.5:
                    detail = f"Relation {'positive' if r>0 else 'négative'} modérée — tendance perceptible."
                elif abs_r >= 0.3:
                    detail = f"Faible corrélation — relation existe mais peu prononcée."
                else:
                    detail = f"Indépendance statistique apparente — évolution non liée."
                itree.insert('', 'end',
                             values=(c1, c2, f"{r:+.3f}", f"{emoji} {force}",
                                     direction, detail),
                             tags=(force,))
                itree.tag_configure(force, background=tag_bg.get(force, '#fff'))
                pairs.append((abs(r), r, c1, c2))
        itree.pack(fill='x', expand=True, padx=8, pady=6)

        # Top 3 + alerte multicolinéarité
        pairs.sort(reverse=True)
        recap = tk.Frame(parent, bg='#eaf4fb', relief='groove', bd=1)
        recap.pack(fill='x', padx=20, pady=(0, 12))
        tk.Label(recap, text="🔎 Top 3 paires les plus corrélées :",
                 font=('Helvetica', 10, 'bold'), bg='#eaf4fb', fg='#154360').pack(
                     anchor='w', padx=14, pady=(8,4))
        for rank, (abs_r, r, c1, c2) in enumerate(pairs[:3], 1):
            emoji, force, _ = self._corr_label(r)
            direction = "positivement" if r > 0 else "négativement"
            tk.Label(recap, text=f"  {rank}. {c1} ↔ {c2} : r = {r:+.3f}  ({emoji} {force}, {direction})",
                     font=('Helvetica', 10), bg='#eaf4fb', fg='#1a5276').pack(anchor='w', padx=24, pady=2)
        high_mc = [(c1,c2,r) for (_,r,c1,c2) in pairs if abs(r) >= 0.8]
        if high_mc:
            warn = ("⚠️  Multicolinéarité potentielle : "
                    + " | ".join([f"{c1} & {c2} (r={r:+.2f})" for c1,c2,r in high_mc])
                    + "\n    Ces variables très corrélées peuvent déstabiliser les coefficients.")
            tk.Label(recap, text=warn, font=('Helvetica', 9,'italic'),
                     bg='#eaf4fb', fg='#922b21', justify='left',
                     wraplength=1100).pack(anchor='w', padx=14, pady=(4,10))

    # ──────────────────────────────────────────────────────────────────────
    #  TAB 3 — MODÉLISATION
    # ──────────────────────────────────────────────────────────────────────
    def _build_tab3(self):
        m = tk.Frame(self.tab3, bg='white')
        m.pack(fill='both', expand=True, padx=30, pady=24)

        tk.Label(m, text="🤖 Configuration du modèle",
                 font=('Helvetica', 18, 'bold'), bg='white', fg='#2c3e50').pack(pady=(0,20))

        # ── Sélection variable cible ──────────────────────────────────────
        tgt_frame = tk.LabelFrame(m, text="🎯 Variable cible (à prédire)",
                                  font=('Helvetica', 12,'bold'), bg='white', fg='#2c3e50',
                                  padx=18, pady=14)
        tgt_frame.pack(fill='x', pady=10)

        row1 = tk.Frame(tgt_frame, bg='white')
        row1.pack(fill='x')
        tk.Label(row1, text="Colonne cible :", bg='white',
                 font=('Helvetica', 10,'bold'), fg='#2c3e50').pack(side='left', padx=(0,10))
        self.target_var = tk.StringVar()
        self.target_cb  = ttk.Combobox(row1, textvariable=self.target_var,
                                        state='readonly', width=28)
        self.target_cb.pack(side='left', padx=6)
        self.target_info_lbl = tk.Label(row1, text="", bg='white',
                                        font=('Helvetica', 10), fg='#7f8c8d')
        self.target_info_lbl.pack(side='left', padx=14)

        # ── Historique modèles pour cette cible ──────────────────────────
        self.hist_lbl = tk.Label(tgt_frame, text="",
                                 font=('Helvetica', 9,'italic'), bg='white', fg='#2980b9')
        self.hist_lbl.pack(anchor='w', pady=(6,0))
        self.target_var.trace('w', self._on_target_change)

        # ── Choix du modèle ───────────────────────────────────────────────
        model_frame = tk.LabelFrame(m, text="🔧 Algorithme",
                                    font=('Helvetica', 12,'bold'), bg='white', fg='#2c3e50',
                                    padx=18, pady=14)
        model_frame.pack(fill='x', pady=10)
        self.model_var = tk.StringVar(value="auto")
        models = [
            ("auto",     "🧠 Auto (choisit le meilleur selon l'historique)"),
            ("linear",   "📊 Régression Linéaire — rapide, interprétable"),
            ("ridge",    "🔵 Ridge (régularisée L2) — anti-multicolinéarité"),
            ("rf",       "🌲 Random Forest — précis, robuste"),
            ("gbm",      "⚡ Gradient Boosting — haute précision"),
        ]
        for val, txt in models:
            tk.Radiobutton(model_frame, text=txt, variable=self.model_var, value=val,
                           font=('Helvetica', 10), bg='white', fg='#2c3e50',
                           selectcolor='#ecf0f1').pack(anchor='w', pady=3)

        # ── Paramètres ────────────────────────────────────────────────────
        par_frame = tk.LabelFrame(m, text="⚙️  Paramètres",
                                  font=('Helvetica', 12,'bold'), bg='white', fg='#2c3e50',
                                  padx=18, pady=14)
        par_frame.pack(fill='x', pady=10)
        row2 = tk.Frame(par_frame, bg='white'); row2.pack(fill='x')
        tk.Label(row2, text="Taille test (%) :", bg='white',
                 font=('Helvetica', 10)).grid(row=0, column=0, padx=(0,8), sticky='e')
        self.test_size_var = tk.IntVar(value=20)
        tk.Scale(row2, from_=10, to=40, orient='horizontal',
                 variable=self.test_size_var, bg='white', length=220,
                 font=('Helvetica', 9)).grid(row=0, column=1, padx=8)
        tk.Label(row2, text="Validation croisée :", bg='white',
                 font=('Helvetica', 10)).grid(row=0, column=2, padx=(20,8), sticky='e')
        self.cv_var = tk.IntVar(value=5)
        tk.Spinbox(row2, from_=3, to=10, textvariable=self.cv_var,
                   width=4, font=('Helvetica', 11)).grid(row=0, column=3, padx=8)
        tk.Label(row2, text="folds", bg='white', font=('Helvetica', 10)).grid(row=0, column=4)

        # ── Bouton train ──────────────────────────────────────────────────
        self.train_btn = tk.Button(m, text="🚀 Entraîner & Sauvegarder le modèle",
                                   command=self.train_model,
                                   font=('Helvetica', 14,'bold'), bg='#27ae60', fg='white',
                                   relief='flat', cursor='hand2', padx=50, pady=14,
                                   state='disabled')
        self.train_btn.pack(pady=24)
        self.status_lbl = tk.Label(m, text="", font=('Helvetica', 11),
                                   bg='white', fg='#7f8c8d')
        self.status_lbl.pack()

    def _on_target_change(self, *_):
        tgt = self.target_var.get()
        if not tgt:
            return
        vtype = self.col_types.get(tgt, 'continuous')
        self.target_info_lbl.config(
            text=f"Type : {VariableTypeDetector.TYPES.get(vtype, vtype)}")
        history = self.db_manager.get_target_history(self.user_id, tgt)
        if history:
            best = max(history, key=lambda x: x[1] if x[1] else 0)
            self.hist_lbl.config(
                text=f"💾 {len(history)} analyse(s) précédente(s) pour « {tgt} » — "
                     f"Meilleur R² = {best[1]:.4f} ({best[0]}) "
                     f"| Mode Auto recommande : {'Ridge' if len(history) >= 3 else 'RF'}")
        else:
            self.hist_lbl.config(text=f"🆕 Première analyse pour « {tgt} »")

    # ──────────────────────────────────────────────────────────────────────
    #  TAB 4 — RÉSULTATS
    # ──────────────────────────────────────────────────────────────────────
    def _build_tab4(self):
        outer = tk.Frame(self.tab4, bg='white')
        outer.pack(fill='both', expand=True)
        canvas_w = tk.Canvas(outer, bg='white')
        vsb      = tk.Scrollbar(outer, orient='vertical', command=canvas_w.yview)
        self.results_frame = tk.Frame(canvas_w, bg='white')
        self.results_frame.bind('<Configure>',
            lambda e: canvas_w.configure(scrollregion=canvas_w.bbox('all')))
        canvas_w.create_window((0,0), window=self.results_frame, anchor='nw')
        canvas_w.configure(yscrollcommand=vsb.set)
        canvas_w.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')
        tk.Label(self.results_frame, text="Les résultats apparaîtront après l'entraînement.",
                 font=('Helvetica', 12), bg='white', fg='#7f8c8d').pack(pady=50)

    # ──────────────────────────────────────────────────────────────────────
    #  TAB 5 — ANALYSE STATISTIQUE
    # ──────────────────────────────────────────────────────────────────────
    def _build_tab5(self):
        outer = tk.Frame(self.tab5, bg='white')
        outer.pack(fill='both', expand=True)
        cs    = tk.Canvas(outer, bg='white')
        vsb   = tk.Scrollbar(outer, orient='vertical', command=cs.yview)
        self._tab5_inner = tk.Frame(cs, bg='white')
        self._tab5_inner.bind('<Configure>',
            lambda e: cs.configure(scrollregion=cs.bbox('all')))
        cs.create_window((0,0), window=self._tab5_inner, anchor='nw')
        cs.configure(yscrollcommand=vsb.set)
        cs.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')

        m = self._tab5_inner

        hdr_f = tk.Frame(m, bg='white'); hdr_f.pack(fill='x', padx=30, pady=(20,0))
        tk.Label(hdr_f, text="📐 Analyse Statistique / Corrélation / E-value",
                 font=('Helvetica', 18,'bold'), bg='white', fg='#2c3e50').pack(side='left')
        tk.Button(hdr_f, text="🔄 Réinitialiser", command=self._reset_tab5,
                  font=('Helvetica', 10,'bold'), bg='#e74c3c', fg='white',
                  relief='flat', cursor='hand2', padx=14, pady=6).pack(side='right')

        # Corrélation ciblée
        cf = tk.LabelFrame(m, text="🔗 Corrélation Ciblée",
                           font=('Helvetica', 12,'bold'), bg='white', fg='#2c3e50',
                           padx=20, pady=15)
        cf.pack(fill='x', padx=30, pady=10)
        sr = tk.Frame(cf, bg='white'); sr.pack(fill='x', pady=5)
        tk.Label(sr, text="Variable X :", bg='white', font=('Helvetica',10,'bold')).grid(row=0, column=0, padx=(0,8), sticky='e')
        self.var1 = tk.StringVar()
        self.var1_cb = ttk.Combobox(sr, textvariable=self.var1, state='readonly', width=22)
        self.var1_cb.grid(row=0, column=1, padx=8)
        tk.Label(sr, text="Variable Y :", bg='white', font=('Helvetica',10,'bold')).grid(row=0, column=2, padx=(20,8), sticky='e')
        self.var2 = tk.StringVar()
        self.var2_cb = ttk.Combobox(sr, textvariable=self.var2, state='readonly', width=22)
        self.var2_cb.grid(row=0, column=3, padx=8)
        tk.Label(sr, text="Visualisation :", bg='white', font=('Helvetica',10,'bold')).grid(row=0, column=4, padx=(20,8), sticky='e')
        self.viz_type = tk.StringVar(value="📊 Vue complète (4 graphes)")
        viz_options = ["📊 Vue complète (4 graphes)","🔵 Nuage de points","📦 Boîtes","🌊 KDE","📉 Hexbin"]
        self._viz_map = {v: v.split()[1].lower().replace('(','').replace(')','') for v in viz_options}
        self._viz_map["📊 Vue complète (4 graphes)"] = "multi"
        self._viz_map["🔵 Nuage de points"] = "scatter"
        self._viz_map["📦 Boîtes"] = "box"
        self._viz_map["🌊 KDE"] = "kde"
        self._viz_map["📉 Hexbin"] = "hexbin"
        ttk.Combobox(sr, textvariable=self.viz_type, values=viz_options,
                     state='readonly', width=26).grid(row=0, column=5, padx=8)

        br = tk.Frame(cf, bg='white'); br.pack(pady=8)
        self.corr_btn = tk.Button(br, text="▶  Analyser", command=self._calculate_corr,
                                  bg='#2980b9', fg='white', font=('Helvetica',11,'bold'),
                                  relief='flat', cursor='hand2', padx=20, pady=8, state='disabled')
        self.corr_btn.pack(side='left', padx=6)
        tk.Button(br, text="🗑  Effacer", command=self._clear_corr_graph,
                  bg='#95a5a6', fg='white', font=('Helvetica',10), relief='flat',
                  cursor='hand2', padx=14, pady=8).pack(side='left', padx=6)

        self.corr_badge_frame = tk.Frame(cf, bg='white'); self.corr_badge_frame.pack(fill='x', pady=(4,0))
        self.corr_badge = tk.Label(self.corr_badge_frame, text="", font=('Helvetica',12,'bold'),
                                   bg='white', fg='white', relief='flat', padx=16, pady=6)
        self.corr_badge.pack(side='left')
        self.corr_interp_lbl = tk.Label(self.corr_badge_frame, text="", font=('Helvetica',10),
                                        bg='white', fg='#2c3e50', justify='left', wraplength=900)
        self.corr_interp_lbl.pack(side='left', padx=14)
        self.corr_graph_frame = tk.Frame(cf, bg='white')
        self.corr_graph_frame.pack(fill='both', expand=True, pady=8)

        # E-value
        ef = tk.LabelFrame(m, text="📊 Analyse E-value / OLS",
                           font=('Helvetica', 12,'bold'), bg='white', fg='#2c3e50',
                           padx=20, pady=15)
        ef.pack(fill='x', padx=30, pady=10)
        evr = tk.Frame(ef, bg='white'); evr.pack(fill='x', pady=5)
        tk.Label(evr, text="Variable cible (Y) :", bg='white', font=('Helvetica',10,'bold')).grid(row=0, column=0, padx=(0,8), sticky='e')
        self.target_var2 = tk.StringVar()
        self.target_cb2  = ttk.Combobox(evr, textvariable=self.target_var2, state='readonly', width=22)
        self.target_cb2.grid(row=0, column=1, padx=8)
        tk.Label(evr, text="Variables X (virgule) :", bg='white', font=('Helvetica',10,'bold')).grid(row=0, column=2, padx=(20,8), sticky='e')
        self.features_var2 = tk.StringVar()
        ttk.Entry(evr, textvariable=self.features_var2, width=40).grid(row=0, column=3, padx=8)

        evbr = tk.Frame(ef, bg='white'); evbr.pack(pady=8)
        self.calc_button = tk.Button(evbr, text="▶  Calculer E-value", command=self._calculate_evalue,
                                     bg='#e67e22', fg='white', font=('Helvetica',11,'bold'),
                                     relief='flat', cursor='hand2', padx=20, pady=8, state='disabled')
        self.calc_button.pack(side='left', padx=6)
        tk.Button(evbr, text="🗑  Effacer", command=self._clear_evalue,
                  bg='#95a5a6', fg='white', font=('Helvetica',10), relief='flat',
                  cursor='hand2', padx=14, pady=8).pack(side='left', padx=6)

        self.ols_tree_frame   = tk.Frame(ef, bg='white'); self.ols_tree_frame.pack(fill='x', pady=6)
        self.evalue_interp_lbl = tk.Label(ef, text="", font=('Helvetica',10), bg='#f0f3f4',
                                          fg='#2c3e50', justify='left', wraplength=1000,
                                          padx=12, pady=8)
        self.evalue_interp_lbl.pack(fill='x', pady=4)
        self.evalue_graph_frame = tk.Frame(ef, bg='white')
        self.evalue_graph_frame.pack(fill='both', expand=True, pady=8)

    # ──────────────────────────────────────────────────────────────────────
    #  TAB 6 — PRÉDICTION FUTURE
    # ──────────────────────────────────────────────────────────────────────
    def _build_tab6(self):
        outer = tk.Frame(self.tab6, bg='white')
        outer.pack(fill='both', expand=True)
        cs  = tk.Canvas(outer, bg='white')
        vsb = tk.Scrollbar(outer, orient='vertical', command=cs.yview)
        self._tab6_inner = tk.Frame(cs, bg='white')
        self._tab6_inner.bind('<Configure>',
            lambda e: cs.configure(scrollregion=cs.bbox('all')))
        cs.create_window((0,0), window=self._tab6_inner, anchor='nw')
        cs.configure(yscrollcommand=vsb.set)
        cs.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')

        m = self._tab6_inner
        tk.Label(m, text="🔮 Prédiction Future",
                 font=('Helvetica', 18,'bold'), bg='white', fg='#2c3e50').pack(pady=(20,4))
        tk.Label(m, text="Utilisez le modèle entraîné pour prédire des valeurs futures\n"
                         "sur une plage de données ou une période temporelle.",
                 font=('Helvetica', 10), bg='white', fg='#7f8c8d',
                 justify='center').pack(pady=(0,16))

        # ── Mode de prédiction ────────────────────────────────────────────
        mode_f = tk.LabelFrame(m, text="📌 Mode de prédiction",
                               font=('Helvetica', 12,'bold'), bg='white', fg='#2c3e50',
                               padx=20, pady=14)
        mode_f.pack(fill='x', padx=30, pady=10)
        self.pred_mode = tk.StringVar(value="manual")
        for val, txt in [
            ("manual",  "✏️  Valeurs manuelles — saisir les valeurs de chaque variable"),
            ("range",   "📊 Plage de valeurs — simuler une variable X sur un intervalle"),
            ("temporal","📅 Projection temporelle — prédire sur N périodes futures"),
        ]:
            tk.Radiobutton(mode_f, text=txt, variable=self.pred_mode, value=val,
                           font=('Helvetica', 10), bg='white', fg='#2c3e50',
                           selectcolor='#ecf0f1',
                           command=self._refresh_pred_ui).pack(anchor='w', pady=3)

        # Zone dynamique
        self.pred_input_frame = tk.LabelFrame(m, text="🎛️  Paramètres de prédiction",
                                              font=('Helvetica', 12,'bold'),
                                              bg='white', fg='#2c3e50', padx=20, pady=14)
        self.pred_input_frame.pack(fill='x', padx=30, pady=10)

        # Boutons action
        btn_row = tk.Frame(m, bg='white'); btn_row.pack(pady=10)
        tk.Button(btn_row, text="🔮 Lancer la prédiction",
                  command=self._run_prediction,
                  font=('Helvetica', 13,'bold'), bg='#8e44ad', fg='white',
                  relief='flat', cursor='hand2', padx=32, pady=12).pack(side='left', padx=8)
        tk.Button(btn_row, text="💾 Sauvegarder",
                  command=self._save_prediction,
                  font=('Helvetica', 11), bg='#2980b9', fg='white',
                  relief='flat', cursor='hand2', padx=20, pady=12).pack(side='left', padx=8)
        tk.Button(btn_row, text="🗑  Effacer",
                  command=self._clear_pred_results,
                  font=('Helvetica', 11), bg='#95a5a6', fg='white',
                  relief='flat', cursor='hand2', padx=20, pady=12).pack(side='left', padx=8)

        # Résultats
        self.pred_result_frame = tk.LabelFrame(m, text="📊 Résultats de prédiction",
                                               font=('Helvetica', 12,'bold'),
                                               bg='white', fg='#2c3e50', padx=16, pady=14)
        self.pred_result_frame.pack(fill='both', expand=True, padx=30, pady=10)
        tk.Label(self.pred_result_frame,
                 text="Entraînez d'abord un modèle, puis revenez ici pour prédire.",
                 font=('Helvetica', 11), bg='white', fg='#7f8c8d').pack(pady=30)

        self._last_pred_result = None

    def _refresh_pred_ui(self):
        for w in self.pred_input_frame.winfo_children():
            w.destroy()
        if self.model is None:
            tk.Label(self.pred_input_frame,
                     text="⚠️  Veuillez d'abord entraîner un modèle (onglet Modélisation).",
                     font=('Helvetica', 11), bg='white', fg='#e74c3c').pack(pady=10)
            return
        mode = self.pred_mode.get()
        if mode == "manual":
            self._build_manual_input()
        elif mode == "range":
            self._build_range_input()
        elif mode == "temporal":
            self._build_temporal_input()

    def _build_manual_input(self):
        m = self.pred_input_frame
        tk.Label(m, text=f"Variable cible à prédire : {self.target_col}",
                 font=('Helvetica', 11,'bold'), bg='white', fg='#8e44ad').pack(anchor='w', pady=(0,10))
        tk.Label(m, text="Renseignez les valeurs des variables explicatives :",
                 font=('Helvetica', 10), bg='white', fg='#2c3e50').pack(anchor='w')

        grid = tk.Frame(m, bg='white'); grid.pack(fill='x', pady=8)
        self._manual_entries = {}
        for i, col in enumerate(self.feature_cols):
            r, c = i // 3, i % 3
            frm = tk.Frame(grid, bg='white')
            frm.grid(row=r, column=c, padx=12, pady=5, sticky='w')
            vtype = self.col_types.get(col, 'continuous')
            icon  = {'date':'📅','binary':'🔀','categorical':'🏷️','ordinal':'📊',
                     'continuous':'📈','integer':'🔢'}.get(vtype, '📌')
            tk.Label(frm, text=f"{icon} {col}", font=('Helvetica', 9,'bold'),
                     bg='white', fg='#2c3e50').pack(anchor='w')
            if vtype in ('categorical', 'ordinal', 'binary') and self.df is not None:
                vals = list(self.df[col].dropna().unique()[:20])
                var  = tk.StringVar(value=str(vals[0]) if vals else '')
                cb   = ttk.Combobox(frm, textvariable=var, values=[str(v) for v in vals],
                                    width=16)
                cb.pack()
                self._manual_entries[col] = var
            else:
                if self.df is not None and pd.api.types.is_numeric_dtype(self.df[col]):
                    default = str(round(float(self.df[col].mean()), 3))
                else:
                    default = ''
                e = tk.Entry(frm, font=('Helvetica', 11), width=16,
                             bg='#ecf0f1', fg='#2c3e50', relief='flat')
                e.insert(0, default)
                e.pack()
                self._manual_entries[col] = e

    def _build_range_input(self):
        m = self.pred_input_frame
        num_features = [c for c in self.feature_cols
                        if self.col_types.get(c) in ('continuous','integer')]
        if not num_features:
            tk.Label(m, text="Aucune variable numérique continue dans les features.",
                     font=('Helvetica',11), bg='white', fg='#e74c3c').pack(pady=10)
            return

        tk.Label(m, text="Faites varier une variable X sur une plage, les autres restent à leur moyenne.",
                 font=('Helvetica',10), bg='white', fg='#2c3e50').pack(anchor='w', pady=(0,8))
        row = tk.Frame(m, bg='white'); row.pack(fill='x')
        tk.Label(row, text="Variable à faire varier :", bg='white',
                 font=('Helvetica',10,'bold')).grid(row=0,column=0,padx=(0,8),sticky='e')
        self.range_var = tk.StringVar(value=num_features[0])
        ttk.Combobox(row, textvariable=self.range_var, values=num_features,
                     state='readonly', width=20).grid(row=0,column=1,padx=8)
        for col, lbl, r, c, default in [
            ('range_min', 'Valeur min :', 0, 2, ''),
            ('range_max', 'Valeur max :', 0, 4, ''),
            ('range_steps','Nb points :',  0, 6, '50'),
        ]:
            tk.Label(row, text=lbl, bg='white', font=('Helvetica',10,'bold')).grid(
                row=r, column=c, padx=(16,6), sticky='e')
            e = tk.Entry(row, font=('Helvetica',11), width=10,
                         bg='#ecf0f1', fg='#2c3e50', relief='flat')
            e.insert(0, default)
            e.grid(row=r, column=c+1, padx=6)
            setattr(self, col+'_entry', e)

        # Auto-fill min/max
        def autofill(*_):
            col = self.range_var.get()
            if self.df is not None and col in self.df.columns:
                self.range_min_entry.delete(0, 'end')
                self.range_min_entry.insert(0, str(round(float(self.df[col].min()), 3)))
                self.range_max_entry.delete(0, 'end')
                self.range_max_entry.insert(0, str(round(float(self.df[col].max()), 3)))
        self.range_var.trace('w', autofill)
        autofill()

    def _build_temporal_input(self):
        m = self.pred_input_frame
        date_cols = [c for c, t in self.col_types.items() if t == 'date' and c in self.df.columns] if self.df is not None else []

        tk.Label(m, text="Projection temporelle : prédire sur N périodes futures.",
                 font=('Helvetica',10), bg='white', fg='#2c3e50').pack(anchor='w', pady=(0,8))
        row = tk.Frame(m, bg='white'); row.pack(fill='x')

        tk.Label(row, text="Nb périodes futures :", bg='white',
                 font=('Helvetica',10,'bold')).grid(row=0,column=0,padx=(0,8),sticky='e')
        self.n_future = tk.IntVar(value=12)
        tk.Spinbox(row, from_=1, to=120, textvariable=self.n_future,
                   width=6, font=('Helvetica',11)).grid(row=0,column=1,padx=8)

        tk.Label(row, text="Unité :", bg='white', font=('Helvetica',10,'bold')).grid(
            row=0,column=2,padx=(16,8),sticky='e')
        self.time_unit = tk.StringVar(value='Mois')
        ttk.Combobox(row, textvariable=self.time_unit,
                     values=['Jours','Semaines','Mois','Trimestres','Années'],
                     state='readonly', width=14).grid(row=0,column=3,padx=8)

        if date_cols:
            tk.Label(row, text="Colonne date :", bg='white', font=('Helvetica',10,'bold')).grid(
                row=0,column=4,padx=(16,8),sticky='e')
            self.date_col_var = tk.StringVar(value=date_cols[0])
            ttk.Combobox(row, textvariable=self.date_col_var,
                         values=date_cols, state='readonly', width=18).grid(row=0,column=5,padx=8)

        tk.Label(m, text="Les autres variables sont fixées à leur dernière valeur connue "
                         "(ou moyenne si non temporelles).",
                 font=('Helvetica',9,'italic'), bg='white', fg='#7f8c8d').pack(anchor='w', pady=(8,0))

    # ──────────────────────────────────────────────────────────────────────
    #  CHARGEMENT DONNÉES
    # ──────────────────────────────────────────────────────────────────────
    def load_data(self):
        filename = filedialog.askopenfilename(
            title="Sélectionner un fichier",
            filetypes=[("Excel/CSV","*.xlsx *.xls *.csv"),("All","*.*")])
        if not filename:
            return
        try:
            if filename.endswith('.csv'):
                self.df = pd.read_csv(filename)
            else:
                self.df = pd.read_excel(filename)

            # Détection des types
            self.col_types = self.detector.detect(self.df)

            # Sauvegarde dans la BD
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            correlation  = self.df[numeric_cols].corr() if len(numeric_cols) >= 2 else None
            self.dataset_id = self.db_manager.save_dataset(
                self.user_id, os.path.basename(filename),
                self.df, self.col_types, correlation)

            # Mise à jour des widgets
            self._update_info_frame(filename)
            self._update_tab2()
            self._refresh_model_tab()
            self._refresh_tab5_columns()
            self._refresh_pred_ui()

            messagebox.showinfo("Succès", f"✅ Fichier chargé : {os.path.basename(filename)}\n"
                                f"{len(self.df)} lignes × {len(self.df.columns)} colonnes\n"
                                f"Types détectés : {len(self.col_types)} colonnes")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger le fichier:\n{e}")

    def _update_info_frame(self, filename):
        for w in self.info_frame.winfo_children():
            w.destroy()
        ok = tk.Frame(self.info_frame, bg='#d4edda', relief='solid', bd=1)
        ok.pack(fill='x', pady=8)
        tk.Label(ok, text="✅ Données chargées avec succès!",
                 font=('Helvetica',12,'bold'), bg='#d4edda', fg='#155724').pack(pady=8)
        sf = tk.Frame(self.info_frame, bg='white'); sf.pack(fill='x', pady=6)
        for txt in [
            f"📊 Lignes: {len(self.df)}   |   Colonnes: {len(self.df.columns)}",
            f"📁 Fichier: {os.path.basename(filename)}",
            "🏷️  Types: " + "  ".join([f"{VariableTypeDetector.TYPES.get(t,'?')} × "
                                        f"{sum(1 for v in self.col_types.values() if v==t)}"
                                        for t in set(self.col_types.values()) if sum(1 for v in self.col_types.values() if v==t)>0]),
        ]:
            tk.Label(sf, text=txt, font=('Helvetica',11), bg='white',
                     fg='#2c3e50', anchor='w').pack(anchor='w', padx=16, pady=2)

        pf = tk.LabelFrame(self.info_frame, text="Aperçu (10 premières lignes)",
                           font=('Helvetica',11,'bold'), bg='white', fg='#2c3e50')
        pf.pack(fill='both', expand=True, pady=14)
        hsb = tk.Scrollbar(pf, orient='horizontal')
        vsb = tk.Scrollbar(pf)
        hsb.pack(side='bottom', fill='x'); vsb.pack(side='right', fill='y')
        tree = ttk.Treeview(pf, xscrollcommand=hsb.set, yscrollcommand=vsb.set, height=10)
        tree.pack(fill='both', expand=True, padx=8, pady=8)
        hsb.config(command=tree.xview); vsb.config(command=tree.yview)
        tree['columns'] = list(self.df.columns); tree['show'] = 'headings'
        for col in tree['columns']:
            tree.heading(col, text=col); tree.column(col, width=110)
        for _, row in self.df.head(10).iterrows():
            tree.insert('', 'end', values=list(row))

        self.train_btn.config(state='normal')

    def _refresh_model_tab(self):
        num_cols = [c for c, t in self.col_types.items()
                    if t in ('continuous','integer','binary') and c in self.df.columns]
        self.target_cb['values'] = num_cols
        if num_cols:
            self.target_cb.set(num_cols[-1])

    def _refresh_tab5_columns(self):
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.var1_cb['values']  = num_cols
        self.var2_cb['values']  = num_cols
        self.target_cb2['values'] = num_cols
        if num_cols:
            self.var1.set(num_cols[0])
            self.var2.set(num_cols[1] if len(num_cols)>1 else num_cols[0])
            self.target_var2.set(num_cols[0])
            self.features_var2.set(",".join(num_cols[1:]))
            self.corr_btn.config(state='normal')
            self.calc_button.config(state='normal')

    # ──────────────────────────────────────────────────────────────────────
    #  ENTRAÎNEMENT
    # ──────────────────────────────────────────────────────────────────────
    def train_model(self):
        if self.df is None:
            messagebox.showerror("Erreur","Chargez d'abord un fichier!"); return
        tgt = self.target_var.get()
        if not tgt:
            messagebox.showerror("Erreur","Sélectionnez une variable cible!"); return

        self.status_lbl.config(text="⏳ Prétraitement en cours…", fg='#f39c12')
        self.root.update()

        try:
            # Prétraitement
            prep = SmartPreprocessor(self.col_types)
            X_df, y, feature_cols = prep.fit_transform(self.df, tgt)
            self.preprocessor = prep
            self.feature_cols  = feature_cols
            self.target_col    = tgt

            X = X_df.values
            y = y.values.astype(float)

            ts = self.test_size_var.get() / 100
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=ts, random_state=42)

            # Choix du modèle
            model_key = self.model_var.get()
            if model_key == "auto":
                history = self.db_manager.get_target_history(self.user_id, tgt)
                if len(history) >= 3:
                    model_key = "ridge"
                else:
                    model_key = "rf"

            model_map = {
                "linear": (LinearRegression(),       "Régression Linéaire"),
                "ridge":  (Ridge(alpha=1.0),          "Ridge (L2)"),
                "rf":     (RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1), "Random Forest"),
                "gbm":    (GradientBoostingRegressor(n_estimators=120, random_state=42),         "Gradient Boosting"),
            }
            mdl, model_name = model_map.get(model_key, model_map["rf"])

            self.status_lbl.config(text=f"⏳ Entraînement {model_name}…", fg='#f39c12')
            self.root.update()

            mdl.fit(self.X_train, self.y_train)
            self.predictions = mdl.predict(self.X_test)
            self.model        = mdl

            r2   = r2_score(self.y_test, self.predictions)
            mse  = mean_squared_error(self.y_test, self.predictions)
            mae  = mean_absolute_error(self.y_test, self.predictions)
            rmse = np.sqrt(mse)

            # Cross-validation
            cv_scores = cross_val_score(mdl, X, y, cv=self.cv_var.get(),
                                        scoring='r2', n_jobs=-1)
            cv_mean = float(cv_scores.mean())
            cv_std  = float(cv_scores.std())

            # Feature importances
            fi = {}
            if hasattr(mdl, 'feature_importances_'):
                fi = dict(zip(X_df.columns, mdl.feature_importances_))
            elif hasattr(mdl, 'coef_'):
                fi = dict(zip(X_df.columns, np.abs(mdl.coef_)))

            # Sauvegarde BD
            self.analysis_id = self.db_manager.save_analysis(
                self.user_id, self.dataset_id,
                "Analyse", model_name, tgt, list(X_df.columns),
                r2, mse, mae, rmse, cv_mean, cv_std, ts,
                len(self.X_train), len(self.X_test),
                mdl, prep.scaler, prep.encoders, fi,
                f"CV={self.cv_var.get()} folds, test={int(ts*100)}%"
            )

            self.status_lbl.config(text=f"✅ Modèle sauvegardé (ID={self.analysis_id})", fg='#27ae60')
            self._display_results(r2, mse, mae, rmse, cv_mean, cv_std, model_name, fi, X_df.columns)
            self._refresh_pred_ui()
            self.notebook.select(3)

        except Exception as e:
            self.status_lbl.config(text="❌ Erreur", fg='#e74c3c')
            messagebox.showerror("Erreur", f"Erreur lors de l'entraînement:\n{e}")

    # ──────────────────────────────────────────────────────────────────────
    #  AFFICHAGE RÉSULTATS
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _interpret_residuals(residuals, predictions):
        lines = []
        mean_res = np.mean(residuals); std_res = np.std(residuals)
        if abs(mean_res) < 0.05 * std_res:
            lines.append("✅ Moyenne des résidus ≈ 0 — modèle non biaisé systématiquement.")
        else:
            d = "sur-estime" if mean_res < 0 else "sous-estime"
            lines.append(f"⚠️  Biais moyen = {mean_res:+.3f} — le modèle tend à {d} légèrement.")
        half = len(predictions)//2
        idx_s = np.argsort(predictions)
        r_lo, r_hi = np.std(residuals[idx_s[:half]]), np.std(residuals[idx_s[half:]])
        ratio = r_hi/r_lo if r_lo > 0 else 1
        if ratio < 1.5:
            lines.append(f"✅ Variance homogène (ratio = {ratio:.2f}) — homoscédasticité respectée.")
        else:
            lines.append(f"⚠️  Hétéroscédasticité probable (ratio = {ratio:.2f}) — variance croissante.")
        z = np.abs((residuals - mean_res)/std_res)
        no = int(np.sum(z > 3))
        lines.append(f"{'✅ Aucune' if no==0 else f'⚠️  {no}'} valeur(s) aberrante(s) (|z|>3).")
        p1 = float(np.mean(np.abs(residuals) <= std_res)*100)
        lines.append(f"📐 {p1:.1f} % des résidus dans ±1σ — "
                     + ("distribution proche de la normale ✅" if p1>=65 else "distribution irrégulière ⚠️"))
        return lines

    @staticmethod
    def _interpret_scatter(y_test, preds, r2):
        lines = []
        dev = np.abs(preds - y_test); med_dev = np.median(dev)
        rng = y_test.max() - y_test.min()
        pct = med_dev/rng*100 if rng > 0 else 0
        if pct < 5:    lines.append("✅ Points très proches de la diagonale — prédictions très fidèles.")
        elif pct < 15: lines.append("🟡 Écart modéré à la diagonale — quelques erreurs acceptables.")
        else:          lines.append("🔴 Écart significatif — erreurs notables sur certaines observations.")
        bias = np.mean(preds - y_test)
        if abs(bias) < 0.02*rng: lines.append(f"✅ Pas de biais systématique (biais = {bias:+.3f}).")
        elif bias > 0: lines.append(f"⚠️  Sur-estimation systématique (biais = +{bias:.3f}).")
        else:          lines.append(f"⚠️  Sous-estimation systématique (biais = {bias:.3f}).")
        return lines

    def _display_results(self, r2, mse, mae, rmse, cv_mean, cv_std, model_name, fi, feat_cols):
        for w in self.results_frame.winfo_children():
            w.destroy()
        residuals = self.y_test - self.predictions
        y_range   = self.y_test.max() - self.y_test.min()
        mae_pct   = mae/y_range*100 if y_range > 0 else 0

        tk.Label(self.results_frame, text=f"📈 Résultats — {model_name}",
                 font=('Helvetica',18,'bold'), bg='white', fg='#2c3e50').pack(pady=(18,4))
        tk.Label(self.results_frame,
                 text=f"Cible : {self.target_col}   |   "
                      f"Train : {len(self.y_train)}   |   Test : {len(self.y_test)}   |   "
                      f"CV : {self.cv_var.get()}-fold R² = {cv_mean:.4f} ± {cv_std:.4f}",
                 font=('Helvetica',10), bg='white', fg='#7f8c8d').pack(pady=(0,10))

        # Cartes métriques
        mf = tk.Frame(self.results_frame, bg='white'); mf.pack(fill='x', padx=20, pady=10)
        cards = [
            ("R² Score",  f"{r2:.4f}",   "#2980b9", "Variance expliquée"),
            ("CV R²",     f"{cv_mean:.4f}±{cv_std:.3f}", "#16a085", "Généralisation"),
            ("RMSE",      f"{rmse:.4f}", "#c0392b", "Erreur quadratique"),
            ("MAE",       f"{mae:.4f}",  "#d35400", "Erreur absolue"),
            ("Erreur %",  f"{mae_pct:.1f}%","#8e44ad", "MAE / plage valeurs"),
        ]
        for i, (name, val, color, sub) in enumerate(cards):
            c = tk.Frame(mf, bg=color); c.grid(row=0,column=i,padx=6,sticky='ew')
            mf.columnconfigure(i, weight=1)
            tk.Label(c, text=name, font=('Helvetica',10,'bold'), bg=color, fg='white').pack(pady=(10,2))
            tk.Label(c, text=val,  font=('Helvetica',14,'bold'), bg=color, fg='white').pack()
            tk.Label(c, text=sub,  font=('Helvetica',8), bg=color, fg='#d6eaf8').pack(pady=(2,10))

        # Interprétation globale
        if r2>=0.9:   q,qc,qi="Excellent","#1e8449","🏆"
        elif r2>=0.75: q,qc,qi="Très bon","#27ae60","✅"
        elif r2>=0.6:  q,qc,qi="Bon","#f39c12","🟡"
        elif r2>=0.4:  q,qc,qi="Moyen","#e67e22","🟠"
        else:          q,qc,qi="Insuffisant","#e74c3c","🔴"

        gf = tk.LabelFrame(self.results_frame, text="🎯 Interprétation globale",
                           font=('Helvetica',12,'bold'), bg='white', fg='#2c3e50')
        gf.pack(fill='x', padx=20, pady=10)
        bf = tk.Frame(gf, bg=qc); bf.pack(anchor='w', padx=14, pady=(10,6))
        tk.Label(bf, text=f"  {qi}  Performance : {q}   (R² = {r2:.4f})  ",
                 font=('Helvetica',13,'bold'), bg=qc, fg='white').pack(padx=4, pady=6)
        for title_t, body_t in [
            (f"R² = {r2:.4f}", f"Le modèle explique {r2*100:.1f}% de la variance de « {self.target_col} ». {'R² quasi-parfait — vérifiez le surapprentissage.' if r2>=0.95 else 'Plus R² est proche de 1, meilleure est la précision.'}"),
            (f"CV R² = {cv_mean:.4f}", f"La validation croisée ({self.cv_var.get()} folds) confirme une performance de {cv_mean*100:.1f}% ± {cv_std*100:.1f}% — {'résultat stable ✅' if cv_std < 0.05 else 'variabilité notable ⚠️, possible surapprentissage.'}"),
            (f"MAE = {mae:.4f}", f"Erreur typique de ±{mae:.4f} unités ({mae_pct:.1f}% de la plage). {'Très faible ✅' if mae_pct<5 else ('Acceptable 🟡' if mae_pct<15 else 'Élevée 🔴')}"),
        ]:
            rf = tk.Frame(gf, bg='white'); rf.pack(fill='x', padx=14, pady=3)
            tk.Label(rf, text=f"▸ {title_t} :", font=('Helvetica',10,'bold'), bg='white', fg='#2c3e50').pack(anchor='w')
            tk.Label(rf, text=f"   {body_t}", font=('Helvetica',10), bg='white', fg='#566573',
                     justify='left', wraplength=1100).pack(anchor='w')
        tk.Label(gf, text="", bg='white').pack(pady=2)

        # Feature Importances
        if fi:
            fi_frame = tk.LabelFrame(self.results_frame, text="🏆 Importance des variables",
                                     font=('Helvetica',12,'bold'), bg='white', fg='#2c3e50')
            fi_frame.pack(fill='x', padx=20, pady=10)
            sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
            n_fi = min(len(sorted_fi), 12)
            fig_fi, ax_fi = plt.subplots(figsize=(12, max(3, n_fi*0.45)))
            names  = [x[0] for x in sorted_fi[:n_fi]]
            values = [x[1] for x in sorted_fi[:n_fi]]
            colors_fi = ['#2980b9' if v > np.mean(values) else '#bdc3c7' for v in values]
            bars = ax_fi.barh(names[::-1], values[::-1], color=colors_fi[::-1],
                              edgecolor='white', linewidth=0.8)
            ax_fi.set_xlabel("Importance", fontsize=10)
            ax_fi.set_title(f"Top {n_fi} variables les plus influentes", fontsize=11, fontweight='bold')
            ax_fi.grid(True, alpha=0.2, axis='x')
            for bar, val in zip(bars, values[::-1]):
                ax_fi.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                           f"{val:.4f}", va='center', fontsize=8)
            plt.tight_layout()
            cv_fi = FigureCanvasTkAgg(fig_fi, fi_frame)
            cv_fi.draw(); cv_fi.get_tk_widget().pack(fill='x', padx=10, pady=8)

        # Graphiques
        grf = tk.LabelFrame(self.results_frame, text="📊 Graphiques d'analyse",
                            font=('Helvetica',12,'bold'), bg='white', fg='#2c3e50')
        grf.pack(fill='both', expand=True, padx=20, pady=10)
        fig, axes = plt.subplots(1, 3, figsize=(18,5))
        fig.suptitle(f"Analyse — {model_name} → {self.target_col}", fontsize=13, fontweight='bold')

        ax1 = axes[0]
        sc  = ax1.scatter(self.y_test, self.predictions, alpha=0.65,
                          c=np.abs(residuals), cmap='RdYlGn_r',
                          edgecolors='white', linewidths=0.3, s=45)
        fig.colorbar(sc, ax=ax1, label='|Résidu|', shrink=0.85)
        lims = [min(self.y_test.min(), self.predictions.min()),
                max(self.y_test.max(), self.predictions.max())]
        ax1.plot(lims, lims, 'k--', lw=1.8, label='Parfait')
        ax1.set_xlabel('Réelles', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Prédictions', fontsize=10, fontweight='bold')
        ax1.set_title(f'Prédictions vs Réalité\nR² = {r2:.4f}', fontsize=10, fontweight='bold')
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.25)

        ax2 = axes[1]
        ax2.scatter(self.predictions, residuals, alpha=0.65, color='#e74c3c',
                    edgecolors='white', linewidths=0.3, s=45)
        ax2.axhline(0, color='black', linestyle='--', lw=2)
        std_r = np.std(residuals)
        ax2.axhline(std_r,  color='#3498db', linestyle=':', lw=1.4, label=f'+1σ={std_r:+.3f}')
        ax2.axhline(-std_r, color='#3498db', linestyle=':', lw=1.4, label=f'-1σ={-std_r:+.3f}')
        ax2.fill_between([self.predictions.min(), self.predictions.max()],
                         -std_r, std_r, alpha=0.07, color='#3498db')
        ax2.set_xlabel('Prédictions', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Résidus', fontsize=10, fontweight='bold')
        ax2.set_title('Résidus vs Prédictions\n(bande ±1σ)', fontsize=10, fontweight='bold')
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.25)

        ax3 = axes[2]
        ax3.hist(residuals, bins=22, color='#9b59b6', alpha=0.6,
                 edgecolor='white', density=True)
        try:
            from scipy.stats import norm
            mu, sg = norm.fit(residuals)
            xn     = np.linspace(residuals.min(), residuals.max(), 200)
            ax3.plot(xn, norm.pdf(xn, mu, sg), 'r-', lw=2,
                     label=f'Normale (μ={mu:.2f}, σ={sg:.2f})')
        except Exception:
            pass
        ax3.axvline(0, color='black', linestyle='--', lw=1.8)
        ax3.set_xlabel('Résidus', fontsize=10, fontweight='bold')
        ax3.set_ylabel('Densité', fontsize=10, fontweight='bold')
        ax3.set_title('Distribution des résidus', fontsize=10, fontweight='bold')
        ax3.legend(fontsize=8); ax3.grid(True, alpha=0.25)
        plt.tight_layout()
        cv_grf = FigureCanvasTkAgg(fig, grf)
        cv_grf.draw(); cv_grf.get_tk_widget().pack(fill='both', expand=True, padx=8, pady=8)

        # Interprétation automatique des graphiques
        af = tk.LabelFrame(self.results_frame, text="🔍 Interprétation automatique",
                           font=('Helvetica',12,'bold'), bg='white', fg='#2c3e50')
        af.pack(fill='x', padx=20, pady=(0,20))
        tk.Label(af, text="📌 Graphe 1 — Prédictions vs Réalité :",
                 font=('Helvetica',10,'bold'), bg='white', fg='#2980b9').pack(anchor='w', padx=14, pady=(10,2))
        for line in self._interpret_scatter(self.y_test, self.predictions, r2):
            tk.Label(af, text=f"   {line}", font=('Helvetica',10), bg='white', fg='#2c3e50',
                     justify='left', wraplength=1150, anchor='w').pack(anchor='w', padx=20, pady=1)
        tk.Label(af, text="📌 Graphes 2 & 3 — Résidus :",
                 font=('Helvetica',10,'bold'), bg='white', fg='#c0392b').pack(anchor='w', padx=14, pady=(10,2))
        for line in self._interpret_residuals(residuals, self.predictions):
            tk.Label(af, text=f"   {line}", font=('Helvetica',10), bg='white', fg='#2c3e50',
                     justify='left', wraplength=1150, anchor='w').pack(anchor='w', padx=20, pady=1)
        tk.Label(af, text="", bg='white').pack(pady=4)

    # ──────────────────────────────────────────────────────────────────────
    #  PRÉDICTION FUTURE
    # ──────────────────────────────────────────────────────────────────────
    def _run_prediction(self):
        if self.model is None:
            messagebox.showerror("Erreur","Entraînez d'abord un modèle!"); return
        mode = self.pred_mode.get()
        try:
            if mode == "manual":
                self._predict_manual()
            elif mode == "range":
                self._predict_range()
            elif mode == "temporal":
                self._predict_temporal()
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur prédiction:\n{e}")

    def _build_pred_input_df(self, overrides=None):
        """Construit un DataFrame avec les valeurs moyennes/modes pour toutes les features."""
        row = {}
        for col in self.feature_cols:
            if overrides and col in overrides:
                row[col] = overrides[col]
            elif col in self.df.columns:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    row[col] = self.df[col].mean()
                else:
                    row[col] = self.df[col].mode().iloc[0] if len(self.df[col].mode()) > 0 else ''
            else:
                row[col] = 0
        return pd.DataFrame([row])

    def _confidence_interval(self, X_input):
        """Intervalle de confiance via bootstrap ou std des arbres RF."""
        pred = float(self.model.predict(X_input)[0])
        if hasattr(self.model, 'estimators_'):
            preds_trees = np.array([e.predict(X_input)[0] for e in self.model.estimators_])
            ci_lo = float(np.percentile(preds_trees, 5))
            ci_hi = float(np.percentile(preds_trees, 95))
        else:
            rmse_approx = np.sqrt(mean_squared_error(self.y_test, self.predictions))
            ci_lo = pred - 1.96 * rmse_approx
            ci_hi = pred + 1.96 * rmse_approx
        return pred, ci_lo, ci_hi

    def _predict_manual(self):
        overrides = {}
        for col, widget in self._manual_entries.items():
            val = widget.get() if hasattr(widget, 'get') else str(widget)
            overrides[col] = val

        input_df = self._build_pred_input_df(overrides)
        X_enc    = self.preprocessor.transform(input_df)
        pred, ci_lo, ci_hi = self._confidence_interval(X_enc.values)

        self._last_pred_result = {
            'mode': 'manual', 'pred': pred, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'inputs': overrides, 'label': f"Manuel — {datetime.now().strftime('%Y-%m-%d %H:%M')}"}
        self._show_single_pred(pred, ci_lo, ci_hi, overrides)

    def _show_single_pred(self, pred, ci_lo, ci_hi, inputs):
        for w in self.pred_result_frame.winfo_children():
            w.destroy()
        # Badge résultat
        res_f = tk.Frame(self.pred_result_frame, bg='#8e44ad')
        res_f.pack(fill='x', padx=10, pady=10)
        tk.Label(res_f, text=f"  🔮  {self.target_col}  =  {pred:.4f}  ",
                 font=('Helvetica',20,'bold'), bg='#8e44ad', fg='white').pack(pady=14)
        tk.Label(res_f,
                 text=f"Intervalle de confiance 90 % : [ {ci_lo:.4f} ; {ci_hi:.4f} ]",
                 font=('Helvetica',12), bg='#8e44ad', fg='#d7bde2').pack(pady=(0,14))

        # Détail des inputs
        inp_f = tk.LabelFrame(self.pred_result_frame, text="Variables utilisées",
                              font=('Helvetica',11,'bold'), bg='white', fg='#2c3e50')
        inp_f.pack(fill='x', padx=10, pady=8)
        grid = tk.Frame(inp_f, bg='white'); grid.pack(fill='x', padx=10, pady=8)
        for i, (col, val) in enumerate(inputs.items()):
            r, c = i // 4, i % 4
            tk.Label(grid, text=f"{col}: {val}", font=('Helvetica',9),
                     bg='#ecf0f1', fg='#2c3e50', padx=8, pady=4,
                     relief='groove').grid(row=r, column=c, padx=4, pady=3, sticky='ew')

        # Interprétation
        rmse_approx = np.sqrt(mean_squared_error(self.y_test, self.predictions))
        pct_ci = (ci_hi - ci_lo) / (pred if pred != 0 else 1) * 100
        interp = (
            f"📌 La valeur prédite de « {self.target_col} » est {pred:.4f}.\n"
            f"   L'intervalle de confiance à 90 % s'étend de {ci_lo:.4f} à {ci_hi:.4f} "
            f"(amplitude = {ci_hi-ci_lo:.4f}, soit ±{pct_ci/2:.1f}%).\n"
            f"   {'Prédiction précise ✅' if pct_ci < 20 else ('Incertitude modérée 🟡' if pct_ci < 50 else 'Forte incertitude ⚠️ — ajoutez plus de données pour améliorer la confiance.')}\n"
            f"   RMSE du modèle sur le jeu de test : {rmse_approx:.4f} — "
            f"{'bon indicateur de fiabilité ✅' if rmse_approx < 0.2*(self.y_test.max()-self.y_test.min()) else 'précision limitée ⚠️'}."
        )
        tk.Label(self.pred_result_frame, text=interp, font=('Helvetica',10),
                 bg='#eaf4fb', fg='#1a5276', justify='left',
                 wraplength=1100, padx=14, pady=10).pack(fill='x', padx=10)

    def _predict_range(self):
        col_x  = self.range_var.get()
        try:
            vmin   = float(self.range_min_entry.get())
            vmax   = float(self.range_max_entry.get())
            nsteps = int(self.range_steps_entry.get())
        except ValueError:
            messagebox.showerror("Erreur","Valeurs min/max/points invalides."); return

        x_vals = np.linspace(vmin, vmax, nsteps)
        preds  = []
        ci_los, ci_his = [], []
        for xv in x_vals:
            inp_df = self._build_pred_input_df({col_x: xv})
            X_enc  = self.preprocessor.transform(inp_df)
            p, lo, hi = self._confidence_interval(X_enc.values)
            preds.append(p); ci_los.append(lo); ci_his.append(hi)

        self._last_pred_result = {
            'mode': 'range', 'x_col': col_x, 'x_vals': x_vals.tolist(),
            'preds': preds, 'ci_los': ci_los, 'ci_his': ci_his,
            'label': f"Plage {col_x} — {datetime.now().strftime('%Y-%m-%d %H:%M')}"}

        for w in self.pred_result_frame.winfo_children():
            w.destroy()
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(x_vals, preds, color='#8e44ad', lw=2.5, label=f'Prédiction de « {self.target_col} »')
        ax.fill_between(x_vals, ci_los, ci_his, alpha=0.18, color='#8e44ad',
                        label='IC 90 %')
        ax.set_xlabel(col_x, fontsize=11, fontweight='bold')
        ax.set_ylabel(self.target_col, fontsize=11, fontweight='bold')
        ax.set_title(f"Simulation : {self.target_col} en fonction de {col_x}",
                     fontsize=13, fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.25)
        plt.tight_layout()
        cv = FigureCanvasTkAgg(fig, self.pred_result_frame)
        cv.draw(); cv.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

        # Interprétation
        slope = (preds[-1]-preds[0])/(x_vals[-1]-x_vals[0]) if x_vals[-1]!=x_vals[0] else 0
        trend = "croissant ↗" if slope > 0 else "décroissant ↘" if slope < 0 else "stable —"
        tk.Label(self.pred_result_frame,
                 text=(f"📌 Quand « {col_x} » passe de {vmin:.2f} à {vmax:.2f}, "
                       f"« {self.target_col} » passe de {preds[0]:.4f} à {preds[-1]:.4f} "
                       f"(tendance {trend}, pente ≈ {slope:.4f} unités/{col_x}).\n"
                       f"   La zone ombrée représente l'intervalle de confiance à 90 %."),
                 font=('Helvetica',10), bg='#eaf4fb', fg='#1a5276',
                 justify='left', wraplength=1100, padx=14, pady=10).pack(fill='x', padx=10)

    def _predict_temporal(self):
        n      = self.n_future.get()
        unit   = self.time_unit.get()
        today  = datetime.now()
        delta_map = {'Jours': timedelta(days=1), 'Semaines': timedelta(weeks=1),
                     'Mois': timedelta(days=30), 'Trimestres': timedelta(days=91),
                     'Années': timedelta(days=365)}
        delta = delta_map.get(unit, timedelta(days=30))
        future_dates = [today + delta * i for i in range(1, n+1)]

        preds, ci_los, ci_his = [], [], []
        for fd in future_dates:
            overrides = {}
            for col in self.feature_cols:
                vtype = self.col_types.get(col,'continuous')
                if vtype == 'date':
                    overrides[col] = fd.strftime('%Y-%m-%d')
                elif col.lower() in ('year','annee','an'):
                    overrides[col] = fd.year
                elif col.lower() in ('month','mois'):
                    overrides[col] = fd.month
                elif col.lower() in ('day','jour'):
                    overrides[col] = fd.day
            inp_df = self._build_pred_input_df(overrides)
            X_enc  = self.preprocessor.transform(inp_df)
            p, lo, hi = self._confidence_interval(X_enc.values)
            preds.append(p); ci_los.append(lo); ci_his.append(hi)

        self._last_pred_result = {
            'mode': 'temporal', 'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
            'preds': preds, 'ci_los': ci_los, 'ci_his': ci_his, 'unit': unit,
            'label': f"Projection {n} {unit} — {datetime.now().strftime('%Y-%m-%d %H:%M')}"}

        for w in self.pred_result_frame.winfo_children():
            w.destroy()

        # Graphique avec historique + projection
        fig, ax = plt.subplots(figsize=(14, 5))

        # Historique si colonne date disponible
        date_cols = [c for c, t in self.col_types.items() if t == 'date' and c in self.df.columns]
        num_cols  = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if date_cols and self.target_col in num_cols:
            try:
                dc  = date_cols[0]
                dt  = pd.to_datetime(self.df[dc], errors='coerce')
                tmp = pd.DataFrame({'dt': dt, 'v': self.df[self.target_col]}).dropna().sort_values('dt')
                ax.plot(tmp['dt'], tmp['v'], color='#2980b9', lw=2, alpha=0.8, label='Historique')
            except Exception:
                pass

        ax.plot(future_dates, preds, color='#8e44ad', lw=2.5, linestyle='--',
                marker='o', markersize=4, label=f'Projection ({unit})')
        ax.fill_between(future_dates, ci_los, ci_his, alpha=0.2, color='#8e44ad', label='IC 90 %')
        ax.axvline(today, color='#e74c3c', linestyle=':', lw=2, label="Aujourd'hui")
        ax.set_xlabel('Période', fontsize=11, fontweight='bold')
        ax.set_ylabel(self.target_col, fontsize=11, fontweight='bold')
        ax.set_title(f"Projection future de « {self.target_col} » — {n} {unit}",
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        fig.autofmt_xdate(); ax.grid(True, alpha=0.25)
        plt.tight_layout()
        cv = FigureCanvasTkAgg(fig, self.pred_result_frame)
        cv.draw(); cv.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

        # Tableau résumé
        tbl_f = tk.LabelFrame(self.pred_result_frame, text="Tableau des prédictions",
                              font=('Helvetica',10,'bold'), bg='white', fg='#2c3e50')
        tbl_f.pack(fill='x', padx=10, pady=8)
        hsb = tk.Scrollbar(tbl_f, orient='horizontal')
        vsb = tk.Scrollbar(tbl_f)
        hsb.pack(side='bottom', fill='x'); vsb.pack(side='right', fill='y')
        tbl_cols = ['Période', 'Date', f'{self.target_col} prédit', 'IC bas', 'IC haut']
        tbl = ttk.Treeview(tbl_f, columns=tbl_cols, show='headings',
                           xscrollcommand=hsb.set, yscrollcommand=vsb.set,
                           height=min(10, n))
        hsb.config(command=tbl.xview); vsb.config(command=tbl.yview)
        for c in tbl_cols:
            tbl.heading(c, text=c); tbl.column(c, width=140, anchor='center')
        for i, (fd, p, lo, hi) in enumerate(zip(future_dates, preds, ci_los, ci_his), 1):
            tbl.insert('', 'end', values=(i, fd.strftime('%Y-%m-%d'), f"{p:.4f}", f"{lo:.4f}", f"{hi:.4f}"))
        tbl.pack(fill='x', expand=True, padx=8, pady=6)

        # Interprétation
        trend_val = preds[-1] - preds[0]
        trend_txt = f"augmentation de {trend_val:.4f}" if trend_val > 0 else f"baisse de {abs(trend_val):.4f}"
        tk.Label(self.pred_result_frame,
                 text=(f"📌 Sur {n} {unit.lower()}, le modèle projette une {trend_txt} de "
                       f"« {self.target_col} » ({preds[0]:.4f} → {preds[-1]:.4f}).\n"
                       f"   Valeur max attendue : {max(preds):.4f}   |   "
                       f"Valeur min attendue : {min(preds):.4f}.\n"
                       f"   ⚠️  Cette projection est basée sur les patterns historiques. "
                       f"Elle ne tient pas compte des événements futurs imprévus."),
                 font=('Helvetica',10), bg='#eaf4fb', fg='#1a5276',
                 justify='left', wraplength=1100, padx=14, pady=10).pack(fill='x', padx=10)

    def _save_prediction(self):
        if not self._last_pred_result:
            messagebox.showinfo("Info","Aucune prédiction à sauvegarder."); return
        if not self.analysis_id:
            messagebox.showwarning("Attention","Associez d'abord un modèle entraîné."); return
        r = self._last_pred_result
        pred_val = r.get('pred') or (r.get('preds',[0])[0] if r.get('preds') else 0)
        ci_lo    = r.get('ci_lo') or (r.get('ci_los',[0])[0] if r.get('ci_los') else 0)
        ci_hi    = r.get('ci_hi') or (r.get('ci_his',[0])[0] if r.get('ci_his') else 0)
        self.db_manager.save_future_prediction(
            self.analysis_id, self.user_id,
            r.get('inputs') or {}, float(pred_val), float(ci_lo), float(ci_hi),
            r.get('label',''))
        messagebox.showinfo("Succès","Prédiction sauvegardée dans la base de données ✅")

    def _clear_pred_results(self):
        for w in self.pred_result_frame.winfo_children():
            w.destroy()
        tk.Label(self.pred_result_frame, text="Résultats effacés.",
                 font=('Helvetica',11), bg='white', fg='#7f8c8d').pack(pady=30)
        self._last_pred_result = None

    # ──────────────────────────────────────────────────────────────────────
    #  TAB 5 — helpers corr/evalue (conservés & simplifiés)
    # ──────────────────────────────────────────────────────────────────────
    def _reset_tab5(self):
        self._clear_corr_graph(); self._clear_evalue()
        if self.df is not None:
            self._refresh_tab5_columns()
        messagebox.showinfo("Réinitialisé","Onglet Analyse Statistique réinitialisé.")

    def _clear_corr_graph(self):
        for w in self.corr_graph_frame.winfo_children(): w.destroy()
        if self._corr_canvas: plt.close(self._corr_canvas.figure); self._corr_canvas = None
        self.corr_badge.config(text="", bg='white'); self.corr_interp_lbl.config(text="")

    def _clear_evalue(self):
        for w in self.ols_tree_frame.winfo_children():    w.destroy()
        for w in self.evalue_graph_frame.winfo_children(): w.destroy()
        if self._evalue_canvas: plt.close(self._evalue_canvas.figure); self._evalue_canvas = None
        self.evalue_interp_lbl.config(text="")

    def _calculate_corr(self):
        if self.df is None or not self.var1.get() or not self.var2.get(): return
        col_x, col_y = self.var1.get(), self.var2.get()
        x = self.df[col_x].dropna(); y = self.df[col_y].dropna()
        common = x.index.intersection(y.index)
        x, y   = x[common], y[common]
        pearson_r, p_value = pearsonr(x, y)
        spearman_r         = x.corr(y, method='spearman')
        abs_r = abs(pearson_r)
        if abs_r>=0.8: force,badge_bg="Très forte","#1a5276"
        elif abs_r>=0.6: force,badge_bg="Forte","#2980b9"
        elif abs_r>=0.4: force,badge_bg="Modérée","#f39c12"
        elif abs_r>=0.2: force,badge_bg="Faible","#e67e22"
        else: force,badge_bg="Très faible / nulle","#e74c3c"
        direction = "positive ↗" if pearson_r>0 else "négative ↘"
        sig_text  = (f"significatif (p={p_value:.4f}<0.05)" if p_value<0.05
                     else f"non significatif (p={p_value:.4f}≥0.05)")
        self.corr_badge.config(text=f"  r = {pearson_r:+.3f}  |  {force}  ",
                               bg=badge_bg, fg='white')
        self.corr_interp_lbl.config(
            text=(f"Pearson r={pearson_r:+.4f}  |  Spearman ρ={spearman_r:+.4f}  |  "
                  f"p={p_value:.4f}  →  {sig_text}\n"
                  f"Corrélation {force.lower()} et {direction}.  "
                  f"R²={pearson_r**2:.4f} → {pearson_r**2*100:.1f}% de variance expliquée."))
        for w in self.corr_graph_frame.winfo_children(): w.destroy()
        if self._corr_canvas: plt.close(self._corr_canvas.figure)

        viz_code = self._viz_map.get(self.viz_type.get(),"multi")
        if viz_code=="multi":
            fig,axes = plt.subplots(2,2,figsize=(13,8))
            fig.suptitle(f"{col_x} vs {col_y}",fontsize=13,fontweight='bold')
            ax = axes[0,0]
            ax.scatter(x,y,alpha=0.55,color='#3498db',edgecolors='white',linewidths=0.4,s=40)
            m_,b_=np.polyfit(x,y,1); xl=np.linspace(x.min(),x.max(),200)
            ax.plot(xl,m_*xl+b_,'r--',lw=2,label=f'y={m_:.2f}x{b_:+.2f}')
            ax.set_title("Nuage + régression",fontsize=9,fontweight='bold'); ax.legend(fontsize=7); ax.grid(True,alpha=0.25)
            ax.text(0.05,0.93,f"r={pearson_r:+.3f}",transform=ax.transAxes,fontsize=9,color=badge_bg,
                    bbox=dict(boxstyle='round,pad=0.3',fc='#ecf0f1',alpha=0.8))
            ax2=axes[0,1]; xn=(x-x.mean())/x.std(); yn=(y-y.mean())/y.std()
            xn.plot.kde(ax=ax2,color='#3498db',lw=2,label=f'{col_x}(std)')
            yn.plot.kde(ax=ax2,color='#e74c3c',lw=2,label=f'{col_y}(std)')
            ax2.set_title("KDE standardisé",fontsize=9,fontweight='bold'); ax2.legend(fontsize=7); ax2.grid(True,alpha=0.25)
            ax3=axes[1,0]; hb=ax3.hexbin(x,y,gridsize=22,cmap='Blues',mincnt=1)
            fig.colorbar(hb,ax=ax3,label='N')
            ax3.set_xlabel(col_x,fontsize=8); ax3.set_ylabel(col_y,fontsize=8)
            ax3.set_title("Hexbin densité",fontsize=9,fontweight='bold')
            ax4=axes[1,1]; bp=ax4.boxplot([xn,yn],labels=[f'{col_x}(std)',f'{col_y}(std)'],
                                           patch_artist=True,medianprops=dict(color='black',lw=2))
            for patch,c in zip(bp['boxes'],['#3498db','#e74c3c']):
                patch.set_facecolor(c); patch.set_alpha(0.6)
            ax4.set_title("Boîtes à moustaches",fontsize=9,fontweight='bold'); ax4.grid(True,alpha=0.25)
            plt.tight_layout()
        else:
            fig,ax=plt.subplots(figsize=(8,5))
            ax.scatter(x,y,alpha=0.6,color='#3498db',edgecolors='white',s=50)
            m_,b_=np.polyfit(x,y,1); xl=np.linspace(x.min(),x.max(),200)
            ax.plot(xl,m_*xl+b_,'r--',lw=2)
            ax.set_xlabel(col_x,fontsize=11,fontweight='bold')
            ax.set_ylabel(col_y,fontsize=11,fontweight='bold')
            ax.set_title(f"{col_x} vs {col_y} (r={pearson_r:+.3f})",fontsize=12,fontweight='bold')
            ax.grid(True,alpha=0.3); plt.tight_layout()

        self._corr_canvas = FigureCanvasTkAgg(fig, self.corr_graph_frame)
        self._corr_canvas.draw(); self._corr_canvas.get_tk_widget().pack(fill='both',expand=True,pady=6)

    def _calculate_evalue(self):
        try:
            import statsmodels.api as sm
        except ImportError:
            messagebox.showerror("Erreur","statsmodels requis: pip install statsmodels"); return
        if self.df is None: return
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        target   = self.target_var2.get()
        features = [f.strip() for f in self.features_var2.get().split(",") if f.strip() in num_cols]
        if not target or not features:
            messagebox.showerror("Erreur","Cible et variables explicatives requises."); return
        X_ols = sm.add_constant(self.df[features]); y_ols = self.df[target]
        ols   = sm.OLS(y_ols, X_ols).fit()
        rdf   = pd.DataFrame({'Variable':ols.params.index,'Coefficient':ols.params.values,
                              'Std Error':ols.bse.values,'t-stat':ols.tvalues.values,
                              'p-value':ols.pvalues.values,'E-value':1-ols.pvalues.values})
        for w in self.ols_tree_frame.winfo_children(): w.destroy()
        cols_t = list(rdf.columns)
        hsb=tk.Scrollbar(self.ols_tree_frame,orient='horizontal')
        vsb2=tk.Scrollbar(self.ols_tree_frame)
        hsb.pack(side='bottom',fill='x'); vsb2.pack(side='right',fill='y')
        t=ttk.Treeview(self.ols_tree_frame,columns=cols_t,show='headings',
                       xscrollcommand=hsb.set,yscrollcommand=vsb2.set,height=min(8,len(rdf)+1))
        hsb.config(command=t.xview); vsb2.config(command=t.yview)
        t.tag_configure('sig',background='#eafaf1',foreground='#1e8449')
        t.tag_configure('not_sig',background='#fdedec',foreground='#922b21')
        t.tag_configure('const',background='#f2f3f4',foreground='#717d7e')
        for _,row in rdf.iterrows():
            if row['Variable']=='const': tag='const'; si='—'
            elif row['p-value']<0.01:    tag='sig';   si='✅✅ p<0.01'
            elif row['p-value']<0.05:    tag='sig';   si='✅ p<0.05'
            elif row['p-value']<0.10:    tag='not_sig'; si='🟡 p<0.10'
            else:                         tag='not_sig'; si='❌ p≥0.10'
            t.insert('','end',values=(row['Variable'],f"{row['Coefficient']:+.4f}",
                     f"{row['Std Error']:.4f}",f"{row['t-stat']:+.3f}",
                     f"{row['p-value']:.4f} {si}",f"{row['E-value']:.4f}"),tags=(tag,))
        t.pack(fill='x',expand=True)

        # Corrélations entre variables explicatives
        corr_feat = self.df[features].corr()
        high_mc   = [(features[i],features[j],corr_feat.iloc[i,j])
                     for i in range(len(features)) for j in range(i+1,len(features))
                     if abs(corr_feat.iloc[i,j])>=0.7]
        feat_df   = rdf[rdf['Variable']!='const'].copy()
        sig_01    = feat_df[feat_df['p-value']<0.01]['Variable'].tolist()
        sig_05    = feat_df[(feat_df['p-value']>=0.01)&(feat_df['p-value']<0.05)]['Variable'].tolist()
        ns_10     = feat_df[feat_df['p-value']>=0.10]['Variable'].tolist()

        interp = (
            f"📊 R²={ols.rsquared:.4f} | R²adj={ols.rsquared_adj:.4f} | "
            f"F-test p={ols.f_pvalue:.4f} {'✅' if ols.f_pvalue<0.05 else '❌'} | "
            f"AIC={ols.aic:.1f}\n"
            f"➜ Variance expliquée : {ols.rsquared*100:.1f}% de « {target} »\n"
            + (f"✅✅ Très signif. (p<0.01) : {', '.join(sig_01)}\n" if sig_01 else "")
            + (f"✅ Signif. (p<0.05) : {', '.join(sig_05)}\n" if sig_05 else "")
            + (f"❌ Non signif. (p≥0.10) : {', '.join(ns_10)}\n" if ns_10 else "")
            + (f"⚠️  Multicolinéarité ({', '.join([f'{a}&{b}(r={r:+.2f})' for a,b,r in high_mc])}) — "
               "coefficients potentiellement instables.\n" if high_mc else "✅ Pas de multicolinéarité détectée (|r|<0.70).\n")
        )
        self.evalue_interp_lbl.config(text=interp)

        for w in self.evalue_graph_frame.winfo_children(): w.destroy()
        if self._evalue_canvas: plt.close(self._evalue_canvas.figure)
        colors_e = ['#27ae60' if p<0.05 else ('#f39c12' if p<0.10 else '#e74c3c') for p in feat_df['p-value']]
        fig_e,(ax1e,ax2e)=plt.subplots(1,2,figsize=(13,max(4,len(feat_df)*0.65+2)))
        bars=ax1e.barh(feat_df['Variable'],feat_df['E-value'],color=colors_e,edgecolor='white')
        ax1e.axvline(0.95,color='#2c3e50',linestyle='--',lw=1.8,label='Seuil 5%')
        ax1e.axvline(0.90,color='#7f8c8d',linestyle=':',lw=1.2,label='Seuil 10%')
        ax1e.set_xlim(0,1.05); ax1e.set_xlabel("E-value (1−p)",fontsize=10)
        ax1e.set_title("E-value par variable\n🟢p<0.05 🟡p<0.10 🔴p≥0.10",fontsize=10,fontweight='bold')
        ax1e.legend(fontsize=8); ax1e.grid(True,alpha=0.2,axis='x')
        for bar,val in zip(bars,feat_df['E-value']):
            ax1e.text(min(bar.get_width()+0.01,1.02),bar.get_y()+bar.get_height()/2,f"{val:.3f}",va='center',fontsize=8)
        ax2e.barh(feat_df['Variable'],feat_df['Coefficient'],xerr=feat_df['Std Error'],
                  color=colors_e,edgecolor='white',capsize=5,error_kw=dict(ecolor='#2c3e50',lw=1.4))
        ax2e.axvline(0,color='black',linestyle='-',lw=1.2)
        ax2e.set_xlabel("Coefficient (±Std Error)",fontsize=10)
        ax2e.set_title("Coefficients\n(flèches ↑↓ = direction de l'effet)",fontsize=10,fontweight='bold')
        ax2e.grid(True,alpha=0.2,axis='x')
        for i,(_,row) in enumerate(feat_df.iterrows()):
            sym = "↑" if row['Coefficient']>0 else "↓"
            ax2e.text(row['Coefficient']+row['Std Error']*1.1,i,sym,va='center',fontsize=10,
                      color='#1e8449' if row['Coefficient']>0 else '#c0392b')
        plt.tight_layout()
        self._evalue_canvas = FigureCanvasTkAgg(fig_e, self.evalue_graph_frame)
        self._evalue_canvas.draw(); self._evalue_canvas.get_tk_widget().pack(fill='both',expand=True,pady=6)

    # ──────────────────────────────────────────────────────────────────────
    #  HISTORIQUE & PRÉDICTIONS SAUVEGARDÉES
    # ──────────────────────────────────────────────────────────────────────
    def show_history(self):
        win = tk.Toplevel(self.root)
        win.title("Historique des Analyses"); win.geometry("1100x500")
        win.configure(bg='white')
        tk.Label(win, text="📊 Historique de vos analyses",
                 font=('Helvetica',16,'bold'), bg='white', fg='#2c3e50').pack(pady=16)
        tf = tk.Frame(win, bg='white'); tf.pack(fill='both', expand=True, padx=16, pady=8)
        hsb = tk.Scrollbar(tf, orient='horizontal'); vsb = tk.Scrollbar(tf)
        hsb.pack(side='bottom',fill='x'); vsb.pack(side='right',fill='y')
        cols = ('Fichier','Modèle','Cible','R²','MSE','MAE','CV R²','Train','Test','Date')
        tree = ttk.Treeview(tf, columns=cols, show='headings',
                            xscrollcommand=hsb.set, yscrollcommand=vsb.set)
        hsb.config(command=tree.xview); vsb.config(command=tree.yview)
        for c,w in zip(cols,[120,110,100,75,80,80,90,60,60,160]):
            tree.heading(c,text=c); tree.column(c,width=w,anchor='center')
        for row in self.db_manager.get_user_analyses(self.user_id):
            fn,mt,tc,r2,mse,mae,cv_r2,nt,nts,dt = row
            tree.insert('','end',values=(fn,mt,tc,
                f"{r2:.4f}" if r2 else '—',
                f"{mse:.4f}" if mse else '—',
                f"{mae:.4f}" if mae else '—',
                f"{cv_r2:.4f}" if cv_r2 else '—',
                nt,nts,dt))
        tree.pack(fill='both',expand=True)

    def show_saved_predictions(self):
        win = tk.Toplevel(self.root)
        win.title("Prédictions sauvegardées"); win.geometry("1050x450")
        win.configure(bg='white')
        tk.Label(win, text="📁 Prédictions futures sauvegardées",
                 font=('Helvetica',16,'bold'), bg='white', fg='#2c3e50').pack(pady=14)
        tf = tk.Frame(win, bg='white'); tf.pack(fill='both', expand=True, padx=14, pady=8)
        hsb = tk.Scrollbar(tf,orient='horizontal'); vsb = tk.Scrollbar(tf)
        hsb.pack(side='bottom',fill='x'); vsb.pack(side='right',fill='y')
        cols = ('Label','Prédit','IC bas','IC haut','Cible','Modèle','Date')
        tree = ttk.Treeview(tf,columns=cols,show='headings',xscrollcommand=hsb.set,yscrollcommand=vsb.set)
        hsb.config(command=tree.xview); vsb.config(command=tree.yview)
        for c,w in zip(cols,[240,90,90,90,100,110,160]):
            tree.heading(c,text=c); tree.column(c,width=w,anchor='center')
        for row in self.db_manager.get_future_predictions(self.user_id):
            lbl,prd,clo,chi,_,dt,mt,tc = row
            tree.insert('','end',values=(lbl,f"{prd:.4f}",f"{clo:.4f}",f"{chi:.4f}",tc,mt,dt))
        tree.pack(fill='both',expand=True)

    # ──────────────────────────────────────────────────────────────────────
    #  DÉCONNEXION
    # ──────────────────────────────────────────────────────────────────────
    def logout(self):
        if messagebox.askyesno("Déconnexion","Êtes-vous sûr ?"):
            self.root.destroy()
            r = tk.Tk()
            LoginWindow(r, self.db_manager)
            r.mainloop()


# ═══════════════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    db  = DatabaseManager()
    root = tk.Tk()
    LoginWindow(root, db)
    root.mainloop()

if __name__ == "__main__":
    main()
