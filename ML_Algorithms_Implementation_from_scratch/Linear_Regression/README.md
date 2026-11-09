# Linear Regression from Scratch using NumPy

This repository contains a pure **NumPy implementation of Linear Regression using Batch Gradient Descent**, designed to demonstrate low-level algorithmic understanding.

The implementation features modular Python architecture, vectorized matrix operations, core evaluation metrics, and benchmarking against `scikit-learn`'s analytical Ordinary Least Squares (OLS) solver.

## 🧮 1. Mathematical Formulation

Linear Regression models the relationship between a dependent continuous target variable $y$ and one or more independent feature variables $X$.

### Predictive Model

For a dataset with $m$ samples and $n$ features:

* Feature matrix: $X \in \mathbb{R}^{m \times n}$

* Target vector: $y \in \mathbb{R}^{m}$

* Weight vector: $w \in \mathbb{R}^{n}$

* Bias scalar: $b \in \mathbb{R}$

The vectorized hypothesis line/hyperplane is defined as:

$$
\hat{y} = Xw + b
$$

### Cost Function (Mean Squared Error)

To measure how well our model fits the data, we compute the Mean Squared Error (MSE) loss $J(w, b)$ over all $m$ training examples:

$$
J(w, b) = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2 = \frac{1}{m} (Xw + b - y)^T (Xw + b - y)
$$

### Gradient Descent & Parameter Update Equations

To find the optimal parameters $(w, b)$ that minimize $J(w, b)$, we apply partial derivatives of the cost function with respect to $w$ and $b$:

1. **Partial Derivative w.r.t Weights (**$\frac{\partial J}{\partial w}$**):**
   

   $$
   \frac{\partial J}{\partial w} = \frac{2}{m} X^T (\hat{y} - y) \quad \xrightarrow{\text{scaling constant factor}} \quad dw = \frac{1}{m} X^T (\hat{y} - y)
   $$

2. **Partial Derivative w.r.t Bias (**$\frac{\partial J}{\partial b}$**):**
   

   $$
   \frac{\partial J}{\partial b} = \frac{2}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i) \quad \xrightarrow{\text{scaling constant factor}} \quad db = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)
   $$

3. **Parameter Update Rules:**
   Using learning rate $\alpha$ (`learning_rate`):
   

   $$
   w := w - \alpha \cdot dw
   $$

   $$
   b := b - \alpha \cdot db
   $$

## 📊 2. Evaluation Metrics

Four core regression evaluation metrics are built into the custom class to measure model precision:

1. **Mean Squared Error (MSE):**
   

   $$
   \text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2
   $$

2. **Root Mean Squared Error (RMSE):**
   

   $$
   \text{RMSE} = \sqrt{\text{MSE}}
   $$

3. **Mean Absolute Error (MAE):**
   

   $$
   \text{MAE} = \frac{1}{m} \sum_{i=1}^{m} \vert{}y_i - \hat{y}_i\vert{}
   $$

4. **Coefficient of Determination (**$R^2$ **Score):**
   

   $$
   R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
   $$

   
   *(where* $\bar{y}$ *is the empirical mean of the actual target values)*

## 🛠️ 3. Code Implementation & Modularity

The codebase follows professional engineering guidelines, separating the algorithm package, package initialization, and benchmarking pipelines.

### Folder Structure

```
Linear_Regression/
│
├── __init__.py                 # Exposes LinearRegression as an importable package module
├── Linear_Regression.py        # Vectorized NumPy implementation & evaluation methods
├── Main.py                     # Synthetic data pipeline, training, comparison & plotting script
├── linear_regression_comparison.png  # Matplotlib dual-subplot visual output
└── README.md                   # Technical documentation

```

### Key Implementation Snippet (`Linear_Regression.py`)

```
import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_rows, n_cols = X.shape
        self.w = np.zeros(n_cols)
        self.b = 0.0

        for _ in range(self.n_iters):
            y_pred = self.predict(X)
            dw = np.dot(X.T, y_pred - y) / n_rows
            db = np.sum(y_pred - y) / n_rows
            
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b
            def mse(self,y_pred,y):
        return np.sum((y_pred-y)**2)/len(y)

    def rmse(self,y_pred,y):
        return np.sqrt(self.mse(y_pred,y))

    def mae(self,y_pred,y):
        return np.sum(np.abs(y_pred-y))/len(y)

    def r2_score(self,y_pred,y):
        ss_res=np.sum((y-y_pred)**2)
        ss_tot=np.sum((y-np.mean(y))**2)
        return 1-ss_res/ss_tot

```

## ⚖️ 4. Comparison: Custom Model vs. Scikit-Learn

While both models aim to minimize the Mean Squared Error, their internal mechanics differ completely:

| Feature / Metric | Custom `LinearRegression` | Scikit-Learn `LinearRegression` | 
| ----- | ----- | ----- | 
| **Optimization Method** | **Iterative Batch Gradient Descent** | **Analytical Ordinary Least Squares (OLS)** | 
| **Algorithmic Mechanics** | Solves step-by-step using learning rate $\alpha$ over $N$ iterations | Solves analytically using Singular Value Decomposition (SVD) on $X$ | 
| **Hyperparameters** | Requires tuning `learning_rate` and `n_iters` | **No hyperparameters** required | 
| **Time Complexity** | $\mathcal{O}(\text{n\_iters} \cdot m \cdot n)$ | $\mathcal{O}(m \cdot n^2 + n^3)$ | 
| **Numerical Equivalence** | Converges **asymptotically** close to optimal weights | Computes **exact mathematical minimum** in one pass | 

### Execution & Output

Running `python Main.py` generates side-by-side metric tables in the console and outputs the comparison plot:



### For Visual Output, open (`linear_regression_comparison.png`)