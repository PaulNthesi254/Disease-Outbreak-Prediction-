# SDG3 — Disease Outbreak Prediction Project
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



CSV_PATH = 'health_data.csv'


def make_synthetic(n_countries=10, months=60):
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for c in range(n_countries):
        country = f"C{c+1}"
        pop = rng.integers(0.5e6, 50e6)
        health_spend = rng.uniform(50, 500)  # USD per capita approx
        for m in range(months):
            date = pd.Timestamp('2019-01-01') + pd.DateOffset(months=m)
            # seasonality + noise
            temp = 20 + 10 * np.sin(2 * np.pi * (m % 12) / 12) + rng.normal(0,2)
            precip = max(0, 100 * (1 + 0.5 * np.sin(2 * np.pi * (m % 12) / 12 + 1)) + rng.normal(0,20))
            mobility = max(0, 100 + rng.normal(0,15))
            base_cases = 5 + 0.00001 * pop + 2 * (temp > 28)
            # occasional outbreaks
            outbreak = rng.random() < 0.05 + 0.02 * (precip > 120)
            cases = int(base_cases * (1 + 10 * outbreak) + rng.poisson(3))
            rows.append({'country':country, 'date':date, 'cases':cases, 'population':pop,
                         'temp':temp, 'precip':precip, 'health_spend':health_spend, 'mobility':mobility})
    return pd.DataFrame(rows)

if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH, parse_dates=['date'])
else:
    print("No CSV found at", CSV_PATH, "— generating synthetic dataset for demo.")
    df = make_synthetic(n_countries=15, months=72)

# Quick peek
print(df.head())


df = df.sort_values(['country','date']).reset_index(drop=True)


thresholds = df.groupby('country')['cases'].quantile(0.90).to_dict()



df['cases_next'] = df.groupby('country')['cases'].shift(-1)
df['outbreak_next'] = df.apply(lambda r: int(r['cases_next'] > thresholds[r['country']]) if pd.notna(r['cases_next']) else np.nan, axis=1)

# drop final rows without target
ndf = df.dropna(subset=['outbreak_next']).copy()

# add lag features: previous 1-3 months cases, rolling mean
ndf['cases_lag1'] = ndf.groupby('country')['cases'].shift(1)
ndf['cases_lag3_mean'] = ndf.groupby('country')['cases'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)

# temporal features
ndf['month'] = ndf['date'].dt.month

# simple per-capita feature
ndf['cases_per_100k'] = ndf['cases'] / (ndf['population'] / 100_000)

# drop rows with NaN in features
ndf = ndf.dropna()

# select features
FEATURES = ['cases', 'cases_lag1', 'cases_lag3_mean', 'cases_per_100k', 'temp', 'precip', 'mobility', 'health_spend', 'month']
X = ndf[FEATURES]
y = ndf['outbreak_next'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=ndf['country'])


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
clf.fit(X_train_scaled, y_train)

# Save scaler & model
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(clf, 'rf_outbreak_model.joblib')



y_pred = clf.predict(X_test_scaled)
y_proba = clf.predict_proba(X_test_scaled)[:,1]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {acc:.3f}, F1: {f1:.3f}, ROC-AUC: {roc:.3f}")

print('\nClassification report:\n', classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix:\n', cm)


imp = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print('\nFeature importances:\n', imp)

plt.figure(figsize=(8,4))
imp.plot(kind='bar')
plt.title('Feature importances — Random Forest')
plt.tight_layout()
plt.show()

# ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],'--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

eval_df = X_test.copy()
eval_df['y_true'] = y_test
eval_df['y_pred'] = y_pred
eval_df['y_proba'] = y_proba
eval_df.to_csv('model_evaluation.csv', index=False)

print('Saved: rf_outbreak_model.joblib, scaler.joblib, model_evaluation.csv')
