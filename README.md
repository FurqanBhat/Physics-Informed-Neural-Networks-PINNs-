# Physics-Informed Neural Network (PINN) for Projectile Trajectory

This repository contains a Jupyter Notebook demonstrating the application of Physics-Informed Neural Networks (PINNs) to model the trajectory of a projectile. It compares the performance of a PINN, which incorporates physical laws into its loss function, against a traditional Neural Network (NN) trained solely on data.

## Project Description

The goal of this project is to model the height (`h`) of a projectile over time (`t`) under constant gravitational acceleration. The true solution follows the kinematic equation: `h(t) = h0 + v0*t + 0.5*g*(t**2)`, where `h0` is the initial height, `v0` is the initial velocity, and `g` is the acceleration due to gravity.

We generate synthetic noisy data points to train both models and evaluate how well each model can learn the underlying physics and predict the trajectory.

## Key Components

### 1. Data Generation

Synthetic data for `h(t)` is generated using the true solution equation with added Gaussian noise to simulate real-world measurement inaccuracies.

- **Parameters:**
  - `g = -9.8` (acceleration due to gravity)
  - `h0 = 1.0` (initial height)
  - `v0 = 10.0` (initial velocity)
- **Data Range:** `t` from `0.0` to `2.0` seconds with `10` data points.
- **Noise:** `noise_level = 0.9` added to `h_data_exact`.

### 2. Neural Network Architecture (`PINN` class)

Both the PINN and the Normal NN use the same basic feed-forward neural network architecture:

- Input Layer: 1 neuron (for `t`)
- Hidden Layers: 2 layers with `20` neurons each, using `Tanh` activation functions.
- Output Layer: 1 neuron (for `h`)

### 3. Loss Functions

#### a) Data Loss

Measures the difference between the model's prediction and the observed noisy data points.

`loss_data = Mean((h_pred - h_data)^2)`

#### b) Physics Loss (ODE Loss)

Encodes the governing differential equation into the loss function. For projectile motion, the velocity `dh/dt = v0 + g*t`.

`loss_ode = Mean(((dh_dt_true) - (dh_dt_pred))^2)`

#### c) Initial Condition Loss

Ensures that the model respects the initial height at `t=0`.

`loss_ic = Mean((h0 - h_pred(t=0))^2)`

### 4. Model Training

Two models are trained:

- **PINN (model1):** Incorporates all three loss components (data, ODE, initial condition) with equal weights (`lambda_data=1.0`, `lambda_ode=1.0`, `lambda_ic=1.0`).
- **Normal NN (model2):** Only uses the data loss (`lambda_data=2.0`, `lambda_ode=0.0`, `lambda_ic=0.0`).

Both models are optimized using `Adam` with a learning rate of `0.01` for `1000` epochs.

## Results and Comparison

The notebook visualizes the predictions of both the PINN and the Normal NN against the noisy data points and the exact true solution. The plots demonstrate:

- **PINN Performance:** The PINN, by incorporating physics constraints, tends to produce a smoother and more physically consistent trajectory, often capturing the general trend better even with sparse or noisy data.
- **Normal NN Performance:** The Normal NN, relying solely on data, might overfit to the noise or fail to generalize well outside the immediate vicinity of the training data points, especially if the data is sparse.

### Visualizations

The notebook generates two main plots:

1.  **PINN Prediction Plot:** Shows the noisy data, exact solution, and the PINN's prediction.
2.  **Normal NN Prediction Plot:** Shows the noisy data, exact solution, and the Normal NN's prediction.
3.  **Combined Comparison Plot:** Overlays all three (exact, PINN, Normal NN) for a direct comparison.

