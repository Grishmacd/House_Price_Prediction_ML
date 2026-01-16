# House_Price_Prediction_ML

This project builds a Machine Learning model to predict **house prices** based on property features. It follows a complete ML workflow:
**Problem Statement → Selection of Data → Collection of Data → EDA → Train/Test Split → Model Selection → Evaluation Metrics**

---

## Problem Statement
Predict the **price of a house** using key input features such as:
- Area / size
- Number of bedrooms (BHK)
- Bathrooms
- Location-related variables (if available)
- Other numeric property features

**Output:** Predicted House Price (Regression)

---

## Selection of Data
**Dataset chosen:** House Price dataset (tabular regression dataset)

**Why this dataset?**
- Real-world regression problem
- Helps practice end-to-end ML pipeline
- Works well for learning preprocessing + regression models

---

## Collection of Data
The dataset is loaded inside the notebook (`houseprice.ipynb`) using Python libraries (commonly via `pandas.read_csv()`).

Typical sources include:
- Kaggle / CSV file / local dataset folder

---

## Main Libraries Used (and why)

1. `pandas`  
   - Loads dataset (CSV), handles DataFrame operations, cleaning, and basic analysis.

2. `numpy`  
   - Numerical calculations, array operations, and transformations.

3. `matplotlib.pyplot`  
   - Basic plotting for EDA (histograms, scatter plots).

4. `seaborn`  
   - Better statistical visualizations (correlation heatmap, distribution plots).

5. `sklearn.model_selection.train_test_split`  
   - Splits the dataset into training and testing sets.

6. `sklearn.preprocessing` (example: `StandardScaler`, `LabelEncoder`, `OneHotEncoder`)  
   - Scaling numeric columns and encoding categorical columns.

7. `sklearn.metrics`  
   - Regression evaluation metrics (MAE, MSE, RMSE, R2 score).

8. Regression Model (from `sklearn`)  
   Common options:
   - `LinearRegression`
   - `RandomForestRegressor`
   - `DecisionTreeRegressor`
   - `GradientBoostingRegressor`

(Your notebook uses one of these models for training.)

---

## Overall Project Flow (Step-by-step)

### 1) Import Libraries
Import all required libraries for:
- Data loading + cleaning
- Visual EDA
- Training ML model
- Evaluating results

### 2) Load the Dataset
- Load CSV into a pandas DataFrame
- Inspect:
  - Shape
  - Column names
  - Data types

### 3) EDA (Exploratory Data Analysis)
Goal: understand data patterns before training.
- Check missing values
- Identify categorical vs numeric columns
- Visualize price distribution
- Check correlation between features and target (price)
- Detect outliers (optional)

### 4) Data Preprocessing
- Handle missing values (drop or fill)
- Encode categorical columns (if present)
- Scale numeric features (if needed)

### 5) Define X and y
- `X` = input features (all columns except price)
- `y` = target column (price)

### 6) Divide into Training and Testing
Use `train_test_split` to create:
- Training set (model learns from this)
- Testing set (final evaluation on unseen data)

### 7) Model Selection
Choose a regression model based on:
- Simplicity and baseline (Linear Regression)
- Better accuracy on complex patterns (Random Forest / Gradient Boosting)

### 8) Train the Model
- Fit the model:
  - `model.fit(X_train, y_train)`

### 9) Predict House Prices
- Predict on test set:
  - `y_pred = model.predict(X_test)`

### 10) Evaluation Metrics (Evaluation Matrix)
Regression models are evaluated using:

1. **MAE (Mean Absolute Error)**
- Average absolute difference between actual and predicted prices

2. **MSE (Mean Squared Error)**
- Penalizes large errors more

3. **RMSE (Root Mean Squared Error)**
- Square root of MSE (in same unit as price)

4. **R² Score**
- Measures how well the model explains variance in house prices

---

## Model Used (Important)
Your notebook trains a regression model (example: Linear Regression / Random Forest / Decision Tree).
Update this line with your exact model name from the notebook:
- **Model:** `<Your Model Name Here>`

---

## How to Run Locally

### Option 1: Run in Jupyter Notebook
1. Download this repository
2. Open `houseprice.ipynb` in Jupyter Notebook / VS Code / Google Colab
3. Run all cells from top to bottom

### Option 2: Run in Google Colab
1. Upload `houseprice.ipynb`
2. Upload dataset file (if required)
3. Run all cells

---

House-Price-Prediction/
1.  houseprice.ipynb
2. dataset.csv # if used
3. README.md

## Developer
Grishma C.D
