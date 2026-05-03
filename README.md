# option-pricing
Spatio-Temporal Convolutional Neural Networks for Data-Driven Option Pricing:. I am  interested in exploring the viability of a Convolutional Neural Network (CNN) architecture for options (derivatives ) pricing 

While CNNs are typically used for spatial feature extraction in images, this research treats the Local Volatility Surface and its temporal evolution as a multi-channel "image" or "video" stream.

The objective is to predict the Normalized Time Value of European Call options on the NIFTY index by learning spatio-temporal dynamics that traditional closed-form solutions, such as the Black-Scholes-Merton (BSM) model, fail to capture due to rigid assumptions (e.g., constant volatility and log-normal distribution).

2. Methodology: Finance as Computer Vision

To adapt a Computer Vision framework for option pricing, we define the following mapping:

Spatial Dimension ($W$): Defined by Moneyness ($K/S$). We use a grid of 11 strikes centered around the At-The-Money (ATM) price.

Temporal Dimension ($H$): Defined by a Lookback Window. We use a 20-day historical window.

Channels ($C$): 12 financial feature channels (Close price, High-Low range, Volume, Implied Volatility, Realized Volatility, etc.) act as the "color channels" of the input tensor.

Input Tensor Shape: [Batch, 20 (Time), 11 (Strikes), 12 (Features)].

Key Innovations

My project translate the financial semi-martingale assumptions into a computer vision task.
This repo contains just a faction of my experiments of using various neural networks and ml algorithms for option pricing and option hedging in real world portfolios in real derivative markets . 
This project uses simulated data but I also have versions where we have used real nse market data along with data aquisition stratergies . 

Normalized Targets: Instead of predicting absolute prices (which vary wildly by spot level), the model predicts the Normalized Time Value ($(Price - Intrinsic) / Spot$). This ensures the model is scale-invariant and generalizes across different market regimes.

Squeeze-and-Excitation (SE) Blocks: We utilize SE blocks to perform channel-wise feature recalibration. This allows the model to prioritize specific financial indicators (like Implied Volatility) depending on the market context.

Metadata Embedding: A separate MLP processes the target option's specific characteristics (Strike, TTE) and fuses this with the CNN's latent representation before the final prediction.

3. Architecture Structure

The model follows a hierarchical residual architecture:

Input Batch Normalization: Standardizes raw financial features across the temporal-spatial grid.

Residual Backbone: Three layers of 2D Convolutions with GELU activation and Batch Normalization.

Global Average Pooling: Collapses the spatial-temporal grid into a latent feature vector.

Fusion Head: Concatenates CNN features with a 32-dimensional embedding of the target contract's metadata.

Sigmoid Bounding: The output is constrained to $[0, 0.25]$ of the spot price, preventing the prediction of negative time values or unrealistic premiums.

4. Directory Structure

Following the mandatory submission format:

project_student_name/
├── checkpoints/
│   └── final_weights.pth      # Trained model state dictionary
├── data/
│   ├── img01.npy              # 3D Tensor samples of market states
│   └── ... img10.npy
├── config.py                  # Hyperparameters and feature definitions
├── dataset.py                 # Data generation, engineering, and PyTorch Dataset
├── model.py                   # CNN Architecture (Residual + SE)
├── train.py                   # Training loop logic
├── predict.py                 # Inference and absolute price reconstruction
├── interface.py               # Standardized grading interface
└── README.md                  # This file


5. Usage Instructions

Training

To generate the synthetic NIFTY dataset and train the model from scratch:

python train.py


This script will:

Simulate a Heston-like stochastic volatility environment.

Build the $20 \times 11 \times 12$ tensors.

Export 10 sample tensors to the data/ directory.

Save the optimized weights to checkpoints/final_weights.pth.

Inference

To run a prediction on a saved market state:

import torch
import numpy as np
from predict import load_and_predict

# Load a sample market state
sample = np.load('data/img01.npy', allow_pickle=True).item()
price = load_and_predict('checkpoints/final_weights.pth', sample)
print(f"Predicted Option Premium: {price[0]:.2f}")


6. Financial Evaluation Metrics

Unlike standard classification tasks, this project evaluates performance using:

MAE (Mean Absolute Error): Absolute deviation from market price in Rupees.

EM (Error Metric): A binned metric used in academic literature (e.g., Tanksale) to measure deviation in discrete price intervals.

Arbitrage Violation Rate: Percentage of predictions that violate lower bounds ($C \geq S-K$) or monotonicity.

Course: Image and Video Processing with Deep Learning 
Term: jan 2026
