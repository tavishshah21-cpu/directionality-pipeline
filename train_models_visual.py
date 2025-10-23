import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import random
import matplotlib.pyplot as plt # Added for plotting

PAIR_DATA_PATH = "pair_data.csv"

# ========= 1. Load data =========
df = pd.read_csv(PAIR_DATA_PATH)
print(f"Loaded {len(df)} site pairs across {df['EQID'].nunique()} earthquakes.")

TARGET = "delta_theta_target"
FEATURES = ["distance_km", "magnitude", "delta_vs30", "avg_pga_pair"]

# ========= 2. Event-based split =========
unique_eqids = df["EQID"].unique()
random.seed(42)
random.shuffle(unique_eqids)

split_idx = int(0.8 * len(unique_eqids))
train_eqids = unique_eqids[:split_idx]
test_eqids = unique_eqids[split_idx:]

train_df = df[df["EQID"].isin(train_eqids)]
test_df = df[df["EQID"].isin(test_eqids)]

print(f"Training earthquakes: {len(train_eqids)}  |  Test earthquakes: {len(test_eqids)}")
print(f"Training pairs: {len(train_df)}  |  Test pairs: {len(test_df)}")

X_train = train_df[FEATURES]
y_train = train_df[TARGET]
X_test = test_df[FEATURES]
y_test = test_df[TARGET]

# ========= 3. Scaling =========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========= 4. Evaluation Helper =========
def evaluate_model(name, model, X_test_data, y_test_data):
    preds = model.predict(X_test_data)
    r2 = r2_score(y_test_data, preds)
    mae = mean_absolute_error(y_test_data, preds)
    print(f"{name:25s} | R²={r2:.3f} | MAE={mae:.2f}")
    return {"Model": name, "R²": r2, "MAE": mae}

results = []

# ========= 5. Train and Test =========
# Benchmark (Linear)
benchmark = LinearRegression()
benchmark.fit(X_train[["distance_km"]], y_train)
results.append(evaluate_model("Benchmark (Linear)", benchmark, X_test[["distance_km"]], y_test))

# Ridge Regression
ridge = Ridge(alpha=1.5)
ridge.fit(X_train_scaled, y_train)
results.append(evaluate_model("Ridge Regression", ridge, X_test_scaled, y_test))

# Neural Network
nn = MLPRegressor(hidden_layer_sizes=(400,)*5, activation='relu',
                  solver='adam', alpha=0.1, max_iter=500,
                  early_stopping=True, random_state=42)
nn.fit(X_train_scaled, y_train)
results.append(evaluate_model("Neural Network", nn, X_test_scaled, y_test))

# ========= 6. Summary =========
summary = pd.DataFrame(results)
print("\n=== Performance on Unseen Earthquake Events ===")
print(summary.to_string(index=False))

# ========= 7. Visualization (NEW SECTION) =========
print("\n📈 Generating performance plots...")

# --- 7.1 Predicted vs True Plots ---
trained_models = {
    "Benchmark (Linear)": benchmark,
    "Ridge Regression": ridge,
    "Neural Network": nn
}

for index, row in summary.iterrows():
    model_name = row["Model"]
    model = trained_models[model_name]
    r2 = row["R²"]
    mae = row["MAE"]

    # Use the correct test data depending on the model
    if model_name == "Benchmark (Linear)":
        X_test_data = X_test[["distance_km"]]
    else:
        X_test_data = X_test_scaled
    
    preds = model.predict(X_test_data)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.6, edgecolors='k', linewidths=0.5)
    plt.plot([0, 90], [0, 90], 'r--', lw=2, label="Perfect Prediction")
    plt.xlabel("True Δθ (°)")
    plt.ylabel("Predicted Δθ (°)")
    plt.title(f"{model_name}\nPredicted vs True Δθ (R²={r2:.2f}, MAE={mae:.1f})")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 90)
    plt.ylim(0, 90)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()

# --- 7.2 Feature Correlation Heatmap ---
plt.figure(figsize=(8, 6))
corr = df[FEATURES + [TARGET]].corr()
im = plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
plt.colorbar(im, fraction=0.046, label="Correlation Coefficient")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()

# Display all generated plots
plt.show()
