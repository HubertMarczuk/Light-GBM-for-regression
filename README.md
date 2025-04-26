# Light-GBM-for-regression
Light GBM for regression

This repository contains two machine learning scripts for regression tasks using **LightGBM** and **XGBoost** algorithms.  
The scripts are written in **Python** and use the following technologies:  
`pandas`, `numpy`, `matplotlib`, `lightgbm`, `xgboost`, `scikit-learn`.

The models are trained and validated on a dataset containing concrete mixture data and attempt to predict the **concrete strength** based on its composition.

Both algorithms are compared using built-in `scikit-learn` metrics such as:
- Mean Squared Error (MSE)
- R² Score
- Metric plots
- Learning curves
- Residual histograms

## Technologies Used
- Python 3.x
- pandas
- numpy
- matplotlib
- lightgbm
- xgboost
- scikit-learn

## How to Run
1. Clone the repository.
2. Install the required libraries by running:
   ```bash
   pip install pandas numpy matplotlib lightgbm xgboost scikit-learn
3. Run one or both scripts depending on your preference:
  ```bash
  python LightGBM.py
  ```
  ```bash
  python XGBoost.py
  ```

## Project Structure
LightGBM.py — script for training and validating a LightGBM regression model.

XGBoost.py — script for training and validating an XGBoost regression model.

Each script loads the dataset, trains the model, evaluates the performance, and visualizes key metrics.
