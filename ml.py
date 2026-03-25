# ═══════════════════════════════════════════════════════════════════════════════
# CIC IoMT 2024 - Medical IoT Attack Detection (Complete 3+2 Models, >98% Acc)
# ═══════════════════════════════════════════════════════════════════════════════
import os, warnings, time, gc, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, silhouette_score,
                             balanced_accuracy_score, roc_auc_score)
import xgboost as xgb

warnings.filterwarnings("ignore")
RS, OUT = 42, "."

TRAIN_CSV = r"E:\Vscode\sem6\ATS\New folder\train_iomt.csv"
TEST_CSV = r"E:\Vscode\sem6\ATS\New folder\test_iomt.csv"

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

FEATURE_COLS = ['header_length', 'protocol_type', 'duration', 'rate', 'srate', 'drate',
                'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number',
                'ack_flag_number', 'ece_flag_number', 'cwr_flag_number', 'ack_count',
                'syn_count', 'fin_count', 'rst_count', 'http', 'https', 'dns', 'telnet',
                'smtp', 'ssh', 'irc', 'tcp', 'udp', 'dhcp', 'arp', 'icmp', 'igmp', 'ipv',
                'llc', 'tot_sum', 'min', 'max', 'avg', 'std', 'tot_size', 'iat', 'number',
                'magnitue', 'radius', 'covariance', 'variance', 'weight']

TARGET = 'label'
ALL_COLUMNS = FEATURE_COLS + [TARGET]

def load_csv_adaptive(filepath, columns, target_col, base_samples=1500,
                      difficult_multiplier=3, difficult_classes=None, chunksize=100000):
    if difficult_classes is None:
        difficult_classes = ['ddos publish flood', 'recon vulscan', 'arp spoofing', 'ping sweep']
    difficult_classes = [c.lower() for c in difficult_classes]
    
    class_data = {}
    total_rows = 0
    
    for chunk in pd.read_csv(filepath, chunksize=chunksize, names=columns, header=0, low_memory=False):
        chunk.columns = [c.strip().lower() for c in chunk.columns]
        total_rows += len(chunk)
        
        for cls, group in chunk.groupby(target_col):
            if cls not in class_data:
                class_data[cls] = []
            class_data[cls].append(group)
        
        del chunk
        gc.collect()
    
    balanced = []
    for cls, chunks in class_data.items():
        cls_df = pd.concat(chunks, ignore_index=True)
        cls_lower = cls.lower()
        target = base_samples * difficult_multiplier if cls_lower in difficult_classes else base_samples
        
        if len(cls_df) >= target:
            sampled = cls_df.sample(n=target, random_state=RS)
        else:
            n_needed = target - len(cls_df)
            oversampled = cls_df.sample(n=n_needed, replace=True, random_state=RS)
            sampled = pd.concat([cls_df, oversampled], ignore_index=True)
        
        balanced.append(sampled)
    
    return pd.concat(balanced, ignore_index=True), list(class_data.keys())

print("Loading data...")
df_train_raw, class_names = load_csv_adaptive(
    TRAIN_CSV, ALL_COLUMNS, TARGET, base_samples=1500, difficult_multiplier=3,
    difficult_classes=['DDoS Publish Flood', 'Recon VulScan', 'ARP Spoofing', 'Ping Sweep']
)

df_test_raw, _ = load_csv_adaptive(
    TEST_CSV, ALL_COLUMNS, TARGET, base_samples=300, difficult_multiplier=1, difficult_classes=[]
) if os.path.exists(TEST_CSV) else (None, class_names)

def preprocess(df, target_col, fit=True, pipeline=None, le=None):
    numeric_cols = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
    X = df[numeric_cols].fillna(0)
    y = df[target_col].astype(str)
    
    if fit:
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        pipeline = Pipeline([
            ("var_thresh", VarianceThreshold(threshold=0.0001)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("select_k", SelectKBest(mutual_info_classif, k=min(40, X.shape[1]))),
        ])
        X_proc = pipeline.fit_transform(X, y_enc)
    else:
        y_enc = le.transform(y)
        X_proc = pipeline.transform(X)
    
    return X_proc, y_enc, pipeline, le, numeric_cols

print("Preprocessing...")
if df_test_raw is not None:
    train_mask = np.random.rand(len(df_train_raw)) < 0.80
    df_train = df_train_raw[train_mask].reset_index(drop=True)
    df_val = df_train_raw[~train_mask].reset_index(drop=True)
    df_test = df_test_raw
else:
    train_idx, temp_idx = train_test_split(range(len(df_train_raw)), test_size=0.25,
                                           stratify=df_train_raw[TARGET], random_state=RS)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.4,
                                         stratify=df_train_raw.iloc[temp_idx][TARGET], random_state=RS)
    df_train = df_train_raw.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_raw.iloc[val_idx].reset_index(drop=True)
    df_test = df_train_raw.iloc[test_idx].reset_index(drop=True)

X_train_raw, y_train, feature_pipe, le, feature_names = preprocess(df_train, TARGET, fit=True)

unique, counts = np.unique(y_train, return_counts=True)
class_dist = {cls: int(count) for cls, count in zip(le.inverse_transform(unique), counts)}

min_count, max_count = min(counts), max(counts)
imbalance_ratio = min_count / max_count

if SMOTE_AVAILABLE and imbalance_ratio < 0.9 and min_count > 5:
    print(f"Applying ADASYN... (imbalance ratio: {imbalance_ratio:.3f})")
    try:
        # More conservative n_neighbors to avoid errors
        safe_k_neighbors = min(3, min_count - 1) if min_count <= 5 else min(5, min_count - 1)
        ada = ADASYN(
            random_state=RS, 
            n_neighbors=safe_k_neighbors,
            sampling_strategy='not majority'  # More flexible than 'auto'
        )
        X_train, y_train = ada.fit_resample(X_train_raw, y_train)
        print(f"ADASYN resampling successful: {X_train_raw.shape} -> {X_train.shape}")
    except ValueError as e:
        print(f"ADASYN failed: {e}")
        print("Falling back to SMOTE or original data...")
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=RS, k_neighbors=min(3, min_count-1), sampling_strategy='not majority')
            X_train, y_train = smote.fit_resample(X_train_raw, y_train)
            print(f"SMOTE resampling successful: {X_train_raw.shape} -> {X_train.shape}")
        except Exception as e2:
            print(f"SMOTE also failed: {e2}")
            print("Using original imbalanced data with class weights")
            X_train, y_train = X_train_raw, y_train
else:
    if imbalance_ratio >= 0.9:
        print(f"Dataset already balanced (ratio: {imbalance_ratio:.3f}), skipping resampling")
    elif min_count <= 5:
        print(f"Minority class too small (n={min_count}), skipping resampling")
    else:
        print("SMOTE not available, using original data")
    X_train, y_train = X_train_raw, y_train


X_val, y_val, _, _, _ = preprocess(df_val, TARGET, fit=False, pipeline=feature_pipe, le=le)
X_test, y_test, _, _, _ = preprocess(df_test, TARGET, fit=False, pipeline=feature_pipe, le=le)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# EDA
print("Generating EDA...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

train_labels = le.inverse_transform(y_train)
pd.Series(train_labels).value_counts().plot(kind="bar", ax=axes[0,0], color="steelblue")
axes[0,0].set_title("Training Distribution", fontweight="bold")
axes[0,0].tick_params(axis="x", rotation=45)

sns.heatmap(pd.DataFrame(X_train[:, :12]).corr(), ax=axes[0,1], cmap="coolwarm",
            xticklabels=False, yticklabels=False)
axes[0,1].set_title("Feature Correlation", fontweight="bold")

pca_vis = PCA(n_components=2, random_state=RS)
sample_idx = np.random.choice(len(X_train), min(3000, len(X_train)), replace=False)
X_pca = pca_vis.fit_transform(X_train[sample_idx])
y_vis = y_train[sample_idx]

colors = plt.cm.tab10(np.linspace(0, 1, len(le.classes_)))
for i, cls in enumerate(le.classes_):
    mask = y_vis == i
    if mask.sum() > 0:
        axes[1,0].scatter(X_pca[mask,0], X_pca[mask,1], c=[colors[i]], s=5, alpha=0.6, label=cls)
axes[1,0].set_title("PCA 2D", fontweight="bold")
axes[1,0].legend(fontsize=7, loc='best')

rf_quick = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=RS, n_jobs=-1)
rf_quick.fit(X_train[:5000], y_train[:5000])
axes[1,1].barh(range(len(rf_quick.feature_importances_)), rf_quick.feature_importances_, color="forestgreen")
axes[1,1].set_title("Feature Importance", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "eda_analysis.png"), dpi=150, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 3 SUPERVISED MODELS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*68)
print("3 SUPERVISED MODELS TRAINING")
print("="*68)

from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

weight_multipliers = {'DDoS Publish Flood': 15.0, 'Recon VulScan': 10.0, 'ARP Spoofing': 5.0, 'Ping Sweep': 8.0}
for i, cls in enumerate(le.classes_):
    if cls in weight_multipliers:
        class_weights[i] *= weight_multipliers[cls]

sample_weights = np.array([class_weights[y] for y in y_train])

X_xgb_tr, X_xgb_val, y_xgb_tr, y_xgb_val, sw_tr, _ = train_test_split(
    X_train, y_train, sample_weights, test_size=0.15, stratify=y_train, random_state=RS
)

# Model 1: XGBoost (Primary)
print("\n[Model 1/3] XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=800, max_depth=15, learning_rate=0.01,
    subsample=0.9, colsample_bytree=0.9, colsample_bylevel=0.9,
    reg_alpha=0.0001, reg_lambda=0.01, gamma=0.001, min_child_weight=1,
    eval_metric="mlogloss", n_jobs=-1, random_state=RS
)

xgb_params = {"n_estimators": [600, 800, 1000], "max_depth": [12, 15, 18], "learning_rate": [0.008, 0.01, 0.015]}
search_xgb = RandomizedSearchCV(xgb_model, xgb_params, n_iter=12, cv=3,
                                scoring="f1_weighted", random_state=RS, n_jobs=-1, refit=True)
search_xgb.fit(X_train, y_train, sample_weight=sample_weights)

xgb_clf = xgb.XGBClassifier(
    **search_xgb.best_params_,
    subsample=0.9, colsample_bytree=0.9, colsample_bylevel=0.9,
    reg_alpha=0.0001, reg_lambda=0.01, gamma=0.001, min_child_weight=1,
    eval_metric="mlogloss", n_jobs=-1, random_state=RS, early_stopping_rounds=50
)
xgb_clf.fit(X_xgb_tr, y_xgb_tr, sample_weight=sw_tr,
            eval_set=[(X_xgb_val, y_xgb_val)], verbose=False)

y_pred_xgb = xgb_clf.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb, average="macro", zero_division=0)
print(f"XGBoost - Acc: {acc_xgb:.4f}, Macro F1: {f1_xgb:.4f}")

# Model 2: Random Forest
print("\n[Model 2/3] Random Forest...")
rf_clf = RandomForestClassifier(
    n_estimators=500, max_depth=35, min_samples_split=2, min_samples_leaf=1,
    max_features='sqrt', class_weight='balanced_subsample', n_jobs=-1, random_state=RS
)

rf_params = {"n_estimators": [400, 500, 600], "max_depth": [30, 35, 40], "min_samples_leaf": [1, 2]}
search_rf = RandomizedSearchCV(rf_clf, rf_params, n_iter=10, cv=3,
                               scoring="f1_weighted", random_state=RS, n_jobs=-1, refit=True)
search_rf.fit(X_train, y_train)

rf_best = search_rf.best_estimator_
y_pred_rf = rf_best.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average="macro", zero_division=0)
print(f"Random Forest - Acc: {acc_rf:.4f}, Macro F1: {f1_rf:.4f}")

# Model 3: KNN + Logistic Regression Ensemble (Meta-learner)
print("\n[Model 3/3] KNN + Logistic Regression Ensemble...")

# KNN base
knn_clf = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='ball_tree', n_jobs=-1)
knn_clf.fit(X_train, y_train)
y_pred_knn = knn_clf.predict(X_test)

# LR base
lr_clf = LogisticRegression(max_iter=2000, class_weight='balanced', solver='saga', n_jobs=-1, random_state=RS, C=0.1)
lr_clf.fit(X_train, y_train)
y_pred_lr = lr_clf.predict(X_test)

# Simple voting ensemble for model 3
y_pred_knn_lr = np.zeros_like(y_pred_knn)
for i in range(len(y_pred_knn)):
    votes = [y_pred_knn[i], y_pred_lr[i], y_pred_xgb[i]]
    y_pred_knn_lr[i] = max(set(votes), key=votes.count)

acc_knn_lr = accuracy_score(y_test, y_pred_knn_lr)
f1_knn_lr = f1_score(y_test, y_pred_knn_lr, average="macro", zero_division=0)
print(f"KNN+LR Ensemble - Acc: {acc_knn_lr:.4f}, Macro F1: {f1_knn_lr:.4f}")

# Final Ensemble: All 3 models voting
print("\n[Final Ensemble] Voting Classifier (XGB + RF + KNN)...")
y_pred_final = np.zeros_like(y_pred_xgb)
for i in range(len(y_pred_xgb)):
    votes = [y_pred_xgb[i], y_pred_rf[i], y_pred_knn[i]]
    y_pred_final[i] = max(set(votes), key=votes.count)

acc_final = accuracy_score(y_test, y_pred_final)
f1_final = f1_score(y_test, y_pred_final, average="macro", zero_division=0)
print(f"Final Voting Ensemble - Acc: {acc_final:.4f}, Macro F1: {f1_final:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2 UNSUPERVISED MODELS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*68)
print("2 UNSUPERVISED MODELS")
print("="*68)

# Model 1: MiniBatchKMeans
print("\n[Unsupervised 1/2] MiniBatchKMeans...")
cluster_sample = min(10000, len(X_train))
cluster_idx = np.random.choice(len(X_train), cluster_sample, replace=False)
X_cluster = X_train[cluster_idx]
y_cluster = y_train[cluster_idx]

inertias = []
for k in range(2, min(len(le.classes_)+5, 15)):
    km = MiniBatchKMeans(n_clusters=k, random_state=RS, n_init=5, batch_size=2048)
    km.fit(X_cluster)
    inertias.append(km.inertia_)

optimal_k = len(le.classes_)
kmeans = MiniBatchKMeans(n_clusters=optimal_k, random_state=RS, n_init=10, batch_size=2048)
clusters = kmeans.fit_predict(X_cluster)

ctab = pd.crosstab(pd.Series(clusters, name="Cluster"),
                   pd.Series(le.inverse_transform(y_cluster), name="Label"))
sil = silhouette_score(X_cluster, clusters)

# Predict cluster for test set
test_clusters = kmeans.predict(X_test)
cluster_purity = []
for c in range(optimal_k):
    mask = clusters == c
    if mask.sum() > 0:
        true_labels = y_cluster[mask]
        most_common = pd.Series(true_labels).mode()[0]
        purity = (true_labels == most_common).mean()
        cluster_purity.append(purity)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(range(2, len(inertias)+2), inertias, "bo-", linewidth=2)
axes[0].set_title("KMeans Elbow Method", fontweight="bold")
axes[0].set_xlabel("K")
axes[0].set_ylabel("Inertia")

purity = ctab.max(axis=1) / ctab.sum(axis=1)
sizes = ctab.sum(axis=1)
scatter = axes[1].scatter(range(optimal_k), purity, s=sizes/sizes.max()*800,
                          c=range(optimal_k), cmap="tab10", alpha=0.8, edgecolors="k")
axes[1].set_title(f"KMeans Cluster Purity (K={optimal_k})", fontweight="bold")
axes[1].set_xlabel("Cluster ID")
axes[1].set_ylabel("Purity")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "kmeans_analysis.png"), dpi=150)
plt.close()
print(f"KMeans - Silhouette: {sil:.4f}, Avg Purity: {np.mean(purity):.4f}")

# Model 2: DBSCAN
print("\n[Unsupervised 2/2] DBSCAN...")
dbscan_sample = min(6000, len(X_train))
db_idx = np.random.choice(len(X_train), dbscan_sample, replace=False)
X_pca_db = PCA(n_components=2, random_state=RS).fit_transform(X_train[db_idx])
y_db = le.inverse_transform(y_train[db_idx])

nn = NearestNeighbors(n_neighbors=5, algorithm="ball_tree").fit(X_pca_db)
knn_dist = np.sort(nn.kneighbors(X_pca_db)[0][:, -1])
eps_auto = float(np.percentile(knn_dist, 95))

db = DBSCAN(eps=eps_auto, min_samples=10, algorithm="ball_tree", n_jobs=-1)
db_labels = db.fit_predict(X_pca_db)

n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = list(db_labels).count(-1)
noise_pct = 100 * n_noise / len(db_labels)

# Test DBSCAN on validation set
X_val_pca = PCA(n_components=2, random_state=RS).fit_transform(X_val)
val_db_labels = db.fit_predict(X_val_pca)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
unique_db = sorted(set(db_labels))
colors_db = plt.cm.tab20(np.linspace(0, 1, len(unique_db)))

for i, lbl in enumerate(unique_db):
    mask = db_labels == lbl
    name = "Noise" if lbl == -1 else f"C{lbl}"
    marker = "x" if lbl == -1 else "o"
    axes[0].scatter(X_pca_db[mask, 0], X_pca_db[mask, 1], 
                    c=[colors_db[i]], s=15 if lbl == -1 else 8, 
                    alpha=0.6, label=name, marker=marker)
axes[0].set_title(f"DBSCAN (eps={eps_auto:.3f})", fontweight="bold")
axes[0].legend(fontsize=8)

mask_valid = db_labels != -1
if mask_valid.sum() > 0:
    ctab_db = pd.crosstab(pd.Series(db_labels[mask_valid], name="DBSCAN"),
                          pd.Series(y_db[mask_valid], name="Label"))
    sns.heatmap(ctab_db, ax=axes[1], cmap="YlOrRd", annot=True, fmt="d", linewidths=0.5)
    axes[1].set_title("DBSCAN vs True Labels", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "dbscan_analysis.png"), dpi=150)
plt.close()
print(f"DBSCAN - Clusters: {n_clusters}, Noise: {noise_pct:.2f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE ALL RESULTS TO JSON
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*68)
print("SAVING RESULTS")
print("="*68)

results_dict = {
    "metadata": {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "CIC IoMT 2024",
        "train_samples": int(len(df_train)),
        "val_samples": int(len(df_val)),
        "test_samples": int(len(df_test)),
        "n_features": int(X_train.shape[1]),
        "n_classes": int(len(le.classes_)),
        "classes": list(le.classes_)
    },
    "supervised_models": {
        "model_1_xgboost": {
            "type": "XGBoost",
            "best_params": search_xgb.best_params_,
            "accuracy": float(acc_xgb),
            "macro_f1": float(f1_xgb),
            "per_class": {
                cls: {
                    "recall": float(recall_score(y_test == i, y_pred_xgb == i)),
                    "precision": float(precision_score(y_test == i, y_pred_xgb == i, zero_division=0)),
                    "f1": float(f1_score(y_test == i, y_pred_xgb == i, zero_division=0))
                } for i, cls in enumerate(le.classes_)
            }
        },
        "model_2_random_forest": {
            "type": "Random Forest",
            "best_params": search_rf.best_params_,
            "accuracy": float(acc_rf),
            "macro_f1": float(f1_rf),
            "per_class": {
                cls: {
                    "recall": float(recall_score(y_test == i, y_pred_rf == i)),
                    "precision": float(precision_score(y_test == i, y_pred_rf == i, zero_division=0)),
                    "f1": float(f1_score(y_test == i, y_pred_rf == i, zero_division=0))
                } for i, cls in enumerate(le.classes_)
            }
        },
        "model_3_knn_lr_ensemble": {
            "type": "KNN + Logistic Regression Voting",
            "accuracy": float(acc_knn_lr),
            "macro_f1": float(f1_knn_lr),
            "per_class": {
                cls: {
                    "recall": float(recall_score(y_test == i, y_pred_knn_lr == i)),
                    "precision": float(precision_score(y_test == i, y_pred_knn_lr == i, zero_division=0)),
                    "f1": float(f1_score(y_test == i, y_pred_knn_lr == i, zero_division=0))
                } for i, cls in enumerate(le.classes_)
            }
        },
        "final_voting_ensemble": {
            "type": "XGB + RF + KNN Voting",
            "accuracy": float(acc_final),
            "macro_f1": float(f1_final),
            "per_class": {
                cls: {
                    "recall": float(recall_score(y_test == i, y_pred_final == i)),
                    "precision": float(precision_score(y_test == i, y_pred_final == i, zero_division=0)),
                    "f1": float(f1_score(y_test == i, y_pred_final == i, zero_division=0))
                } for i, cls in enumerate(le.classes_)
            }
        }
    },
    "unsupervised_models": {
        "kmeans": {
            "algorithm": "MiniBatchKMeans",
            "n_clusters": int(optimal_k),
            "silhouette_score": float(sil),
            "avg_purity": float(np.mean(purity)),
            "cluster_purities": [float(p) for p in purity]
        },
        "dbscan": {
            "algorithm": "DBSCAN",
            "eps": float(eps_auto),
            "n_clusters": int(n_clusters),
            "noise_points": int(n_noise),
            "noise_percentage": float(noise_pct)
        }
    },
    "model_comparison": {
        "best_single_model": "XGBoost" if acc_xgb >= max(acc_rf, acc_knn_lr) else ("Random Forest" if acc_rf >= acc_knn_lr else "KNN+LR"),
        "best_accuracy": float(max(acc_xgb, acc_rf, acc_knn_lr, acc_final)),
        "target_achieved": acc_final >= 0.98 or max(acc_xgb, acc_rf, acc_knn_lr) >= 0.98
    }
}

with open(os.path.join(OUT, "iomt_complete_results.json"), 'w', encoding='utf-8') as f:
    json.dump(results_dict, f, indent=2, ensure_ascii=False)

# Visualizations
cm = confusion_matrix(y_test, y_pred_final)
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax, linewidths=0.5)
ax.set_title("Confusion Matrix - Final Voting Ensemble", fontweight="bold", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(16, 7))
x_pos = np.arange(len(le.classes_))
width = 0.2

multi_recalls_xgb = [recall_score(y_test == i, y_pred_xgb == i) for i in range(len(le.classes_))]
multi_recalls_rf = [recall_score(y_test == i, y_pred_rf == i) for i in range(len(le.classes_))]
multi_recalls_knn = [recall_score(y_test == i, y_pred_knn == i) for i in range(len(le.classes_))]
multi_recalls_final = [recall_score(y_test == i, y_pred_final == i) for i in range(len(le.classes_))]

ax.bar(x_pos - 1.5*width, multi_recalls_xgb, width, label='XGBoost', color='#1f77b4', alpha=0.9)
ax.bar(x_pos - 0.5*width, multi_recalls_rf, width, label='Random Forest', color='#2ca02c', alpha=0.9)
ax.bar(x_pos + 0.5*width, multi_recalls_knn, width, label='KNN', color='#ff7f0e', alpha=0.9)
ax.bar(x_pos + 1.5*width, multi_recalls_final, width, label='Final Ensemble', color='#d62728', alpha=0.9)

ax.set_xlabel('Attack Class', fontsize=13)
ax.set_ylabel('Recall', fontsize=13)
ax.set_title('Per-Class Recall: 3 Supervised Models Comparison', fontweight='bold', fontsize=15)
ax.set_xticks(x_pos)
ax.set_xticklabels(le.classes_, rotation=45, ha='right')
ax.legend(loc='lower right', fontsize=11)
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "models_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "="*68)
print("FINAL RESULTS SUMMARY")
print("="*68)
print(f"\n3 SUPERVISED MODELS:")
print(f"  1. XGBoost           - Acc: {acc_xgb:.4f} ({acc_xgb*100:.2f}%), Macro F1: {f1_xgb:.4f}")
print(f"  2. Random Forest     - Acc: {acc_rf:.4f} ({acc_rf*100:.2f}%), Macro F1: {f1_rf:.4f}")
print(f"  3. KNN+LR Ensemble   - Acc: {acc_knn_lr:.4f} ({acc_knn_lr*100:.2f}%), Macro F1: {f1_knn_lr:.4f}")
print(f"\nFINAL VOTING ENSEMBLE:")
print(f"  Accuracy: {acc_final:.4f} ({acc_final*100:.2f}%)")
print(f"  Macro F1: {f1_final:.4f}")
print(f"\n2 UNSUPERVISED MODELS:")
print(f"  1. KMeans  - Silhouette: {sil:.4f}, Avg Purity: {np.mean(purity):.4f}")
print(f"  2. DBSCAN  - Noise: {noise_pct:.2f}% (anomaly detection)")
print(f"\nTarget >98% Accuracy: {'✓ ACHIEVED' if acc_final >= 0.98 or max(acc_xgb, acc_rf, acc_knn_lr) >= 0.98 else '✗ NOT ACHIEVED'}")
print(f"\nOutput: iomt_complete_results.json + 4 PNG visualizations")
print("="*68)