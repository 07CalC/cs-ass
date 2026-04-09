import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor

from pygam import LinearGAM, s


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
DATASET_DIR = Path(os.environ["DATASET_DIR"]).expanduser() if "DATASET_DIR" in os.environ else None

if DATASET_DIR is not None and not DATASET_DIR.is_absolute():
    DATASET_DIR = BASE_DIR / DATASET_DIR


def resolve_data_file(filename):
    candidates = []

    if DATASET_DIR is not None:
        candidates.append(DATASET_DIR / filename)

    candidates.extend((
        BASE_DIR / "dataset" / filename,
        BASE_DIR / filename,
    ))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    exact_matches = sorted(BASE_DIR.rglob(filename))
    if exact_matches:
        return exact_matches[0]

    lowered_name = filename.lower()
    csv_matches = sorted(
        path for path in BASE_DIR.rglob("*.csv")
        if path.name.lower() == lowered_name
    )
    if csv_matches:
        return csv_matches[0]

    searched = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        f"Could not find {filename}. Checked:\n{searched}\n"
        f"You can also set DATASET_DIR to the folder containing the CSV files."
    )


def load_data():
    weather = pd.read_csv(resolve_data_file("Plant_1_Weather_Sensor_Data.csv"))
    gen = pd.read_csv(resolve_data_file("Plant_1_Generation_Data.csv"))

    weather['DATE_TIME'] = pd.to_datetime(weather['DATE_TIME'], format="%Y-%m-%d %H:%M:%S")
    gen['DATE_TIME'] = pd.to_datetime(gen['DATE_TIME'], format="%d-%m-%Y %H:%M")

    gen_grouped = gen.groupby('DATE_TIME').sum().reset_index()
    df = pd.merge(weather, gen_grouped, on='DATE_TIME')

    data = pd.DataFrame()
    
    data['DATE_TIME'] = df['DATE_TIME']

    data['DNI'] = df['IRRADIATION']
    data['Temperature'] = df['AMBIENT_TEMPERATURE']
    data['Module_Temp'] = df['MODULE_TEMPERATURE']

    data['Temp_Diff'] = df['MODULE_TEMPERATURE'] - df['AMBIENT_TEMPERATURE']
    data['Irr_Temp_Ratio'] = df['IRRADIATION'] / (df['AMBIENT_TEMPERATURE'] + 1e-5)

    data['GP'] = df['AC_POWER']

    data = data[data['GP'] > 0]

    return data


def evaluate(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r = np.corrcoef(y_true, y_pred)[0, 1]
    print(f"{name} -> RMSE: {rmse:.4f}, R: {r:.4f}")

def plot_model(ax, y_true, y_pred, title):
    ax.scatter(y_true, y_pred, alpha=0.5, s=10)
    ax.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        'k--'
    )
    ax.set_title(title)
    ax.set_xlabel("Observed (kW)")
    ax.set_ylabel("Predicted (kW)")
    ax.grid(True)


def save_figure(fig, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path.relative_to(BASE_DIR)}")


def main():
    print("Loading Data...")
    df = load_data()

    X = df.drop(columns=['GP', 'DATE_TIME'])
    y = df['GP']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    gpr = GaussianProcessRegressor(
        kernel=RationalQuadratic(), alpha=0.1, random_state=42
    )

    rnn = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        max_iter=2000,
        random_state=42
    )

    ann = MLPRegressor(
        hidden_layer_sizes=(150,),
        activation='relu',
        max_iter=2000,
        random_state=42
    )

    lsboost = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    print("Generating Meta Features with K-Fold...")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    meta_features = np.zeros((X_train_scaled.shape[0], 4))

    for train_idx, val_idx in kf.split(X_train_scaled):

        X_tr = X_train_scaled[train_idx]
        X_val = X_train_scaled[val_idx]

        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        gpr.fit(X_tr, y_tr)
        rnn.fit(X_tr, y_tr)
        ann.fit(X_tr, y_tr)
        lsboost.fit(X_tr, y_tr)

        meta_features[val_idx, 0] = gpr.predict(X_val)
        meta_features[val_idx, 1] = rnn.predict(X_val)
        meta_features[val_idx, 2] = ann.predict(X_val)
        meta_features[val_idx, 3] = lsboost.predict(X_val)

    gpr.fit(X_train_scaled, y_train)
    rnn.fit(X_train_scaled, y_train)
    ann.fit(X_train_scaled, y_train)
    lsboost.fit(X_train_scaled, y_train)

    pred_gpr = gpr.predict(X_test_scaled)
    pred_rnn = rnn.predict(X_test_scaled)
    pred_ann = ann.predict(X_test_scaled)
    pred_lsboost = lsboost.predict(X_test_scaled)

    X_test_meta = np.column_stack((
        pred_gpr,
        pred_rnn,
        pred_ann,
        pred_lsboost
    ))

    print("Training EnsGAM...")

    gam = LinearGAM(
        s(0) + s(1) + s(2) + s(3)
    )

    gam.fit(meta_features, y_train)
    final_predictions = gam.predict(X_test_meta)

    print("\n=== RESULTS ===")
    evaluate(y_test, pred_gpr, "GPR")
    evaluate(y_test, pred_rnn, "RNN")
    evaluate(y_test, pred_lsboost, "LSBoost")
    evaluate(y_test, final_predictions, "EnsGAM-Improved")

    print("\nGenerating Visualizations...")

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Observed vs Predicted Power Generation", fontsize=16)

    plot_model(axs[0, 0], y_test, pred_gpr, "GPR")
    plot_model(axs[0, 1], y_test, pred_rnn, "RNN")
    plot_model(axs[1, 0], y_test, pred_lsboost, "LSBoost")
    plot_model(axs[1, 1], y_test, final_predictions, "EnsGAM Improved")

    plt.tight_layout()
    save_figure(
        fig,
        "figure_01_observed_vs_predicted_ac_power_comparison.png"
    )

    fig = plt.figure(figsize=(10, 8))
    cols_to_corr = ['DNI', 'Temperature', 'Module_Temp', 'Temp_Diff', 'Irr_Temp_Ratio', 'GP']
    corr_matrix = df[cols_to_corr].corr()

    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap", fontsize=16)
    plt.tight_layout()
    save_figure(
        fig,
        "figure_02_correlation_matrix_of_engineered_features.png"
    )

    fig = plt.figure(figsize=(10, 6))
    df['Hour'] = df['DATE_TIME'].dt.hour
    hourly_irradiation = df.groupby('Hour')['DNI'].mean()

    plt.plot(hourly_irradiation.index, hourly_irradiation.values, marker='o', color='orange', linewidth=2)
    plt.title("Diurnal Variations of Solar Irradiation", fontsize=16)
    plt.xlabel("Hour of the Day", fontsize=12)
    plt.ylabel("Average Direct Normal Irradiance (W/m^2)", fontsize=12)
    plt.xticks(np.arange(0, 24, 2))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_figure(
        fig,
        "figure_03_diurnal_profile_of_direct_normal_irradiance.png"
    )


if __name__ == "__main__":
    main()
