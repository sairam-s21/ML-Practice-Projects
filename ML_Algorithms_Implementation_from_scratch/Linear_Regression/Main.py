import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Linear_Regression import LinearRegression as CustomLinearRegression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    # 1. Generate Synthetic Dataset using Pandas
    np.random.seed(42)
    num_samples = 200

    x_data = np.random.uniform(-3, 3, num_samples)
    y_data = 2.5 * x_data + 4.0 + np.random.normal(0, 1.2, num_samples)

    df = pd.DataFrame({"Feature": x_data, "Target": y_data})

    X = df[["Feature"]].values
    y = df["Target"].values

    # 2. Train-Test Split & Feature Scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Fit Custom NumPy Linear Regression Model
    custom_model = CustomLinearRegression(learning_rate=0.05, n_iters=1000)
    custom_model.fit(X_train_scaled, y_train)
    y_pred_custom = custom_model.predict(X_test_scaled)

    # 4. Fit Scikit-Learn Linear Regression Model
    sklearn_model = SklearnLinearRegression()
    sklearn_model.fit(X_train_scaled, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test_scaled)

    # 5. Print Learned Parameters & Comprehensive Metrics Comparison
    print("==================================================")
    print("          LEARNED PARAMETERS COMPARISON          ")
    print("==================================================")
    print(f"Custom Weights:  {np.round(custom_model.w, 4)}")
    print(f"Sklearn Weights: {np.round(sklearn_model.coef_, 4)}")
    print("--------------------------------------------------")
    print(f"Custom Bias:     {custom_model.b:.4f}")
    print(f"Sklearn Bias:    {sklearn_model.intercept_:.4f}")

    print("\n==================================================")
    print("          EVALUATION METRICS COMPARISON          ")
    print("==================================================")
    print(
        f"MSE   -> Custom: {custom_model.mse(y_pred_custom, y_test):.4f} | Sklearn: {mean_squared_error(y_test, y_pred_sklearn):.4f}"
    )
    print(
        f"RMSE  -> Custom: {custom_model.rmse(y_pred_custom, y_test):.4f} | Sklearn: {np.sqrt(mean_squared_error(y_test, y_pred_sklearn)):.4f}"
    )
    print(
        f"MAE   -> Custom: {custom_model.mae(y_pred_custom, y_test):.4f} | Sklearn: {mean_absolute_error(y_test, y_pred_sklearn):.4f}"
    )
    print(
        f"R2    -> Custom: {custom_model.r2_score(y_pred_custom, y_test):.4f} | Sklearn: {r2_score(y_test, y_pred_sklearn):.4f}"
    )
    print("==================================================")

    # 6. Plot Curves in Two Separate Subplots and Save PNG
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Sort test points for a clean continuous regression line
    sort_idx = np.argsort(X_test_scaled.flatten())
    X_test_sorted = X_test_scaled[sort_idx]

    # Subplot 1: Custom Model
    axes[0].scatter(
        X_test_scaled, y_test, color="blue", alpha=0.6, label="Actual Data"
    )
    axes[0].plot(
        X_test_sorted,
        custom_model.predict(X_test_sorted),
        color="red",
        linewidth=2,
        label="Custom Fit",
    )
    axes[0].set_title("Custom Linear Regression (NumPy)")
    axes[0].set_xlabel("Scaled Feature")
    axes[0].set_ylabel("Target")
    axes[0].legend()
    axes[0].grid(True)

    # Subplot 2: Sklearn Model
    axes[1].scatter(
        X_test_scaled, y_test, color="green", alpha=0.6, label="Actual Data"
    )
    axes[1].plot(
        X_test_sorted,
        sklearn_model.predict(X_test_sorted),
        color="orange",
        linewidth=2,
        label="Sklearn Fit",
    )
    axes[1].set_title("Scikit-Learn Linear Regression")
    axes[1].set_xlabel("Scaled Feature")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    output_filename = "linear_regression_comparison.png"
    plt.savefig(output_filename, dpi=300)
    plt.close()

    print(f"\nExecution complete! Plot saved as '{output_filename}'")


if __name__ == "__main__":
    main()