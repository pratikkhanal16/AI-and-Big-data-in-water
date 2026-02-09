import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math # Added for calculating axis maximum
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import tensorflow as tf
from pathlib import Path
from sklearn.linear_model import LinearRegression # Needed for scatter plot trendline
from matplotlib.ticker import MultipleLocator # Added for specific tick placement

# --- 1. SETUP AND DATA LOADING ---
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Ensure figure output directory exists
FIG_DIR = Path("artifacts/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

date_cols = ["Date"]
# File path reverted to local path for environment execution
try:
    df = pd.read_csv("madi river data.csv", parse_dates=date_cols)
except FileNotFoundError:
    print("Error: 'madi river data.csv' not found. Cannot proceed.")
    # Create dummy data to prevent further errors for demonstration if needed, but here we halt.
    raise

df = df.sort_values("Date").set_index("Date")
feature_names = df.columns.tolist()

# Define the split index
split_index = 6902

# --- 2. DATA PREPROCESSING (as per original script) ---
df1 = df.iloc[:split_index].copy()
df2 = df.iloc[split_index:].copy()
X_train = df.iloc[:split_index, :]
X_test = df.iloc[split_index:, :]
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        end_ix = i + seq_length
        if end_ix >= len(data):
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y[0])
    return np.array(X), np.array(y)

sequence_length = 7
n_features = len(feature_names)

# Create training and test sequences
X_train, y_train = create_sequences(X_train_scaled, sequence_length)
X_test, y_test = create_sequences(X_test_scaled, sequence_length)

# Reshape X_train and X_test to ensure they have the correct shape
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))

# --- 3. LSTM MODEL BUILDING AND TRAINING ---
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model = Sequential([
    LSTM(
        units=128,
        activation="relu",
        return_sequences=True,
        input_shape=(sequence_length, n_features)
    ),
    Dropout(0.1),
    LSTM(units=96, activation="relu"),
    Dropout(0.1),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")

print("Starting model training...")
history = model.fit(
    X_train,
    y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=0 # Changed to 0 for cleaner output
)
print("Training complete.")

# --- 4. PREDICTION AND INVERSE TRANSFORM ---
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Inverse transform helper setup
y_train_pred_full = np.zeros((y_train_pred.shape[0], n_features))
y_train_full = np.zeros((y_train.shape[0], n_features))
y_test_pred_full = np.zeros((y_test_pred.shape[0], n_features))
y_test_full = np.zeros((y_test.shape[0], n_features))

y_train_pred_full[:, 0] = y_train_pred.flatten()
y_train_full[:, 0] = y_train
y_test_pred_full[:, 0] = y_test_pred.flatten()
y_test_full[:, 0] = y_test

# Inverse transform
y_train_pred_inv = scaler.inverse_transform(y_train_pred_full)[:, 0]
y_train_inv = scaler.inverse_transform(y_train_full)[:, 0]
y_test_pred_inv = scaler.inverse_transform(y_test_pred_full)[:, 0]
y_test_inv = scaler.inverse_transform(y_test_full)[:, 0]

# --- 5. METRICS (as per original script) ---
EPS = np.finfo(float).eps

def nash_sutcliffe_efficiency(obs, sim):
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom == 0:
        return np.nan
    return 1 - np.sum((obs - sim) ** 2) / denom

def kling_gupta_efficiency(obs, sim):
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    obs_std = np.std(obs)
    sim_std = np.std(sim)
    if obs_std == 0 or sim_std == 0:
        return np.nan
    # Ensure obs and sim are 1D arrays for corrcoef
    r = np.corrcoef(obs.flatten(), sim.flatten())[0, 1]
    if np.isnan(r):
        return np.nan
    alpha = sim_std / (obs_std + EPS)
    beta = np.mean(sim) / (np.mean(obs) + EPS)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

def root_mean_square_error(obs, sim):
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    return np.sqrt(np.mean((obs - sim) ** 2))

# --- 6. PLOTTING SETUP ---

# Correct indices for training data plots
df1_dates = df.index[sequence_length:split_index]
df1_precipitation = df.iloc[sequence_length:split_index]['Precipitation'].values

# Correct indices for test data plots
df2_dates = df.index[split_index + sequence_length:]
df2_precipitation = df.iloc[split_index + sequence_length:]['Precipitation'].values

# Define the explicit precipitation ticks and limit
PRECIP_TICKS = [0, 200, 400, 600, 800, 1000]
MAX_PRECIP_LIM = 1000

# Create the figure with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# --- Panel A: Training Time Series (Discharge & Precipitation) ---
ax_a = axs[0, 0]
ax_a.plot(df1_dates, y_train_inv, label='Observed', color='red', linestyle='solid', linewidth=1)
ax_a.plot(df1_dates, y_train_pred_inv, label='Predicted', color='blue', linestyle='--', linewidth=1)
ax_a.set_ylabel('Discharge ($m^3/s$)')

# UPDATED: Move legend inside the plot box to the upper right corner, and 1cm down
ax_a.legend(loc='upper right', bbox_to_anchor=(1, 0.95))

ax_a.set_title('A: Training Data Time Series', loc='left')

# Secondary y-axis for Precipitation (High scale, Blue color)
ax_a_twin = ax_a.twinx()
ax_a_twin.bar(df1_dates, df1_precipitation, color='blue', alpha=0.7, width=1, label='Precipitation')
ax_a_twin.set_ylabel('Precipitation (mm)', color='blue')
ax_a_twin.tick_params(axis='y', labelcolor='blue')
ax_a_twin.invert_yaxis()

# Set fixed Y-limits and explicit ticks
ax_a_twin.set_ylim(MAX_PRECIP_LIM, 0) 
ax_a_twin.set_yticks(PRECIP_TICKS)

# --- Panel B: Training Scatter Plot ---
ax_b = axs[0, 1]
ax_b.scatter(y_train_inv, y_train_pred_inv, color='red', alpha=0.5, s=10)
ax_b.set_xlabel('Observed Discharge ($m^3/s$)')
ax_b.set_ylabel('Predicted Discharge ($m^3/s$)')
ax_b.set_title('B: Training Data Scatter', loc='left')

# Add 1:1 line and regression/R2
min_val_b = min(y_train_inv.min(), y_train_pred_inv.min())
max_val_b = max(y_train_inv.max(), y_train_pred_inv.max())
plot_range_b = [min_val_b - 0.05 * (max_val_b - min_val_b), max_val_b + 0.05 * (max_val_b - min_val_b)]
ax_b.plot(plot_range_b, plot_range_b, color='black', linestyle='--', linewidth=1, label='1:1 Line')
ax_b.set_aspect('equal', adjustable='box')
ax_b.set_xlim(plot_range_b)
ax_b.set_ylim(plot_range_b)

reg_b = LinearRegression()
reg_b.fit(y_train_inv.reshape(-1, 1), y_train_pred_inv)
y_pred_reg_b = reg_b.predict(y_train_inv.reshape(-1, 1))
r2_b = r2_score(y_train_inv, y_train_pred_inv)
ax_b.plot(y_train_inv, y_pred_reg_b, color='black', linestyle='-', linewidth=1, alpha=0.7)
ax_b.text(0.05, 0.95, f'y = {reg_b.coef_[0]:.2f}x + {reg_b.intercept_:.2f}\nR² = {r2_b:.2f}',
         transform=ax_b.transAxes, fontsize=10, verticalalignment='top')


# --- Panel C: Test Time Series (Discharge & Precipitation) ---
ax_c = axs[1, 0]
ax_c.plot(df2_dates, y_test_inv, label='Observed', color='red', linestyle='solid', linewidth=1)
ax_c.plot(df2_dates, y_test_pred_inv, label='Predicted', color='blue', linestyle='--', linewidth=1)
ax_c.set_xlabel('Date')
ax_c.set_ylabel('Discharge ($m^3/s$)')

# UPDATED: Move legend inside the plot box to the upper right corner, and 1cm down
ax_c.legend(loc='upper right', bbox_to_anchor=(1, 0.95))

ax_c.set_title('C: Testing Data Time Series', loc='left')

# Secondary y-axis for Precipitation (High scale, Blue color)
ax_c_twin = ax_c.twinx()
ax_c_twin.bar(df2_dates, df2_precipitation, color='blue', alpha=0.7, width=1, label='Precipitation')
ax_c_twin.set_ylabel('Precipitation (mm)', color='blue')
ax_c_twin.tick_params(axis='y', labelcolor='blue')
ax_c_twin.invert_yaxis()

# Set fixed Y-limits and explicit ticks
ax_c_twin.set_ylim(MAX_PRECIP_LIM, 0) 
ax_c_twin.set_yticks(PRECIP_TICKS)

# --- Panel D: Test Scatter Plot ---
ax_d = axs[1, 1]
ax_d.scatter(y_test_inv, y_test_pred_inv, color='red', alpha=0.5, s=10)
ax_d.set_xlabel('Observed Discharge ($m^3/s$)')
ax_d.set_ylabel('Predicted Discharge ($m^3/s$)')
ax_d.set_title('D: Testing Data Scatter', loc='left')

# Add 1:1 line
min_val_d = min(y_test_inv.min(), y_test_pred_inv.min())
max_val_d = max(y_test_inv.max(), y_test_pred_inv.max())

# Calculate the next multiple of 100 for the axis limit maximum
plot_max = math.ceil(max_val_d / 100) * 100
plot_range_d = [0, plot_max]

ax_d.plot(plot_range_d, plot_range_d, color='black', linestyle='--', linewidth=1, label='1:1 Line')
ax_d.set_aspect('equal', adjustable='box')

# Set X and Y limits to start at 0 and end at the calculated multiple of 100
ax_d.set_xlim(plot_range_d)
ax_d.set_ylim(plot_range_d)

# Set major ticks to be at multiples of 100 (0, 100, 200, ...)
ax_d.xaxis.set_major_locator(MultipleLocator(100))
ax_d.yaxis.set_major_locator(MultipleLocator(100))

# Add regression line and R2
reg_d = LinearRegression()
reg_d.fit(y_test_inv.reshape(-1, 1), y_test_pred_inv)
y_pred_reg_d = reg_d.predict(y_test_inv.reshape(-1, 1))
r2_d = r2_score(y_test_inv, y_test_pred_inv)
ax_d.plot(y_test_inv, y_pred_reg_d, color='black', linestyle='-', linewidth=1, alpha=0.7)
ax_d.text(0.05, 0.95, f'y = {reg_d.coef_[0]:.2f}x + {reg_d.intercept_:.2f}\nR² = {r2_d:.2f}',
         transform=ax_d.transAxes, fontsize=10, verticalalignment='top')

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(FIG_DIR / 'combined_discharge_prediction_plots.png')
plt.show()


# Print metrics as requested originally
baseline_metrics = {
    "RMSE": root_mean_square_error(y_test_inv, y_test_pred_inv),
    "NSE": nash_sutcliffe_efficiency(y_test_inv, y_test_pred_inv),
    "KGE": kling_gupta_efficiency(y_test_inv, y_test_pred_inv),
    "R2": r2_score(y_test_inv, y_test_pred_inv)
}

print("\nPerformance Metrics (Deterministic):")
for metric, value in baseline_metrics.items():
    print(f"{metric}: {value:.3f}")