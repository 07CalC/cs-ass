import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Import Base Models
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.neural_network import MLPRegressor # Represents ANN/RNN in the paper
from sklearn.ensemble import GradientBoostingRegressor # Represents LSBoost

# Import Meta Model (GAM)
# Note: You may need to run 'pip install pygam'
from pygam import LinearGAM, s

# ==========================================
# 1. DATA PREPARATION
# ==========================================
# ==========================================
# 1. DATA PREPARATION (REAL MERGED DATA)
# ==========================================
def load_data():
    print("Reading CSV files...")
    # Load both datasets
    weather = pd.read_csv('Plant_1_Weather_Sensor_Data.csv')
    gen = pd.read_csv('Plant_1_Generation_Data.csv')

    # Convert time columns to standard datetime format so we can merge them
    weather['DATE_TIME'] = pd.to_datetime(weather['DATE_TIME'])
    gen['DATE_TIME'] = pd.to_datetime(gen['DATE_TIME'])

    # The Generation file has data for many separate inverters. 
    # We will group them to get the TOTAL power for the whole plant at each timestamp.
    gen_grouped = gen.groupby('DATE_TIME').sum().reset_index()

    # Merge the files together based on the timestamp
    df = pd.merge(weather, gen_grouped, on='DATE_TIME')

    # Select the columns we need for the paper
    # Inputs (from Weather file)
    data = pd.DataFrame()
    data['DNI'] = df['IRRADIATION']
    data['Temperature'] = df['AMBIENT_TEMPERATURE']
    data['Module_Temp'] = df['MODULE_TEMPERATURE']
    
    # Output (from Generation file)
    # The paper predicts "Generated Power". We use AC_POWER as the target.
    data['GP'] = df['AC_POWER']

    # Remove rows where Power is 0 (Night time), otherwise results get skewed
    data = data[data['GP'] > 0]

    return data

print("Loading Data...")
df = load_data()
X = df.drop(columns=['GP']) # Inputs
y = df['GP'] # Output (Target)

# Split Data 70% Training / 30% Testing as per paper [cite: 360]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize Data (Critical for Neural Networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 2. BASE MODELS IMPLEMENTATION (Stage 1)
# ==========================================
print("Training Base Models...")

# Model 1: GPR (Gaussian Process Regression) 
# Paper uses Ardexponential kernel, we use RationalQuadratic as a close approximation in sklearn
gpr = GaussianProcessRegressor(kernel=RationalQuadratic(), alpha=0.1, random_state=42)
gpr.fit(X_train_scaled, y_train)
pred_gpr = gpr.predict(X_test_scaled)

# Model 2: RNN/ANN (Neural Networks) [cite: 493, 817]
# The paper distinguishes ANN and RNN, but in regression tasks, they are structurally similar (MLPs).
# We will use MLPRegressor with different architectures to represent them.
rnn = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', max_iter=2000, random_state=42)
rnn.fit(X_train_scaled, y_train)
pred_rnn = rnn.predict(X_test_scaled)

ann = MLPRegressor(hidden_layer_sizes=(150,), activation='relu', max_iter=2000, random_state=42)
ann.fit(X_train_scaled, y_train)
pred_ann = ann.predict(X_test_scaled)

# Model 3: LSBoost (Least Squares Boosting) 
# Implemented via GradientBoostingRegressor
lsboost = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
lsboost.fit(X_train_scaled, y_train)
pred_lsboost = lsboost.predict(X_test_scaled)

# ==========================================
# 3. ENSEMBLE CONSTRUCTION (Stage 2 - EnsGAM)
# ==========================================
print("Training EnsGAM Meta-Model...")

# Prepare the "Level 1" dataset for the Meta Model
# The input to GAM is the PREDICTIONS of the base models [cite: 391]

# We need predictions on the TRAINING set to train the GAM
train_pred_gpr = gpr.predict(X_train_scaled)
train_pred_rnn = rnn.predict(X_train_scaled)
train_pred_ann = ann.predict(X_train_scaled)
train_pred_lsboost = lsboost.predict(X_train_scaled)

# Stack them into a matrix
X_train_meta = np.column_stack((train_pred_gpr, train_pred_rnn, train_pred_ann, train_pred_lsboost))

# Prepare the TEST set for the Meta Model
X_test_meta = np.column_stack((pred_gpr, pred_rnn, pred_ann, pred_lsboost))

# Initialize GAM (Generalized Additive Model) [cite: 393]
# s(0) means a spline term for the 0th feature, etc.
gam = LinearGAM(s(0) + s(1) + s(2) + s(3))
gam.fit(X_train_meta, y_train)

# Final Prediction using EnsGAM
final_predictions = gam.predict(X_test_meta)

# ==========================================
# 4. EVALUATION & METRICS
# ==========================================
print("\n=== Results ===")

def calculate_metrics(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r_val = np.corrcoef(y_true, y_pred)[0, 1]
    print(f"{model_name} -> RMSE: {rmse:.4f}, R: {r_val:.4f}")

calculate_metrics(y_test, pred_gpr, "GPR")
calculate_metrics(y_test, pred_rnn, "RNN")
calculate_metrics(y_test, pred_lsboost, "LSBoost")
calculate_metrics(y_test, final_predictions, "**EnsGAM (Proposed)**")

# ==========================================
# 5. VISUALIZATION (Comparison Grid)
# ==========================================
print("Generating comparison graphs...")

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Observed vs Predicted Power Generation (Comparison)', fontsize=16)

# Function to plot each model
def plot_model(ax, y_true, y_pred, title, color):
    ax.scatter(y_true, y_pred, color=color, alpha=0.5, s=10)
    # Draw the perfect diagonal line
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    ax.set_title(title)
    ax.set_xlabel('Observed Power (kW)')
    ax.set_ylabel('Predicted Power (kW)')
    ax.grid(True)

# Plot 1: GPR
plot_model(axs[0, 0], y_test, pred_gpr, 'GPR Model', 'green')

# Plot 2: RNN
plot_model(axs[0, 1], y_test, pred_rnn, 'RNN Model', 'purple')

# Plot 3: LSBoost
plot_model(axs[1, 0], y_test, pred_lsboost, 'LSBoost Model', 'orange')

# Plot 4: EnsGAM (Proposed)
plot_model(axs[1, 1], y_test, final_predictions, 'EnsGAM (Proposed)', 'red')

plt.tight_layout()

# Save the combined comparison image
plt.savefig('comparison_graph.png')
print("Graph saved as 'comparison_graph.png'")