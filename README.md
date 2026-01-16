# House Price Prediction Using Linear Regression (Machine Learning)

This project builds a **regression** model to predict **house prices** based on basic property features like **Area, Bedrooms, and Bathrooms**. It follows the standard ML workflow:
**Problem Statement → Selection of Data → Collection of Data → EDA → Train/Test Split → Model Selection → Evaluation Metric**

---

## Problem Statement
Predict the **house price** from property features so buyers/sellers can estimate pricing quickly.

**Input Features:**
- Area (sq ft)
- Bedrooms
- Bathrooms

**Output:**
- Predicted House Price (Lakhs)

---

## Selection of Data
**Dataset Type Used:** Structured tabular dataset (numeric features)

Why this dataset is suitable:
- Simple numeric features for regression learning
- Helps understand how features impact price
- Good beginner project for supervised ML

---

## Collection of Data
In this project, the dataset is created inside the code as a sample dictionary and converted into a DataFrame using `pandas`.  
(The same pipeline works with real CSV datasets loaded using `pd.read_csv()`.)

---

## EDA (Exploratory Data Analysis)
EDA is kept simple to confirm the data is valid before training:
- Preview the dataset using `df.head()`
- Confirm feature columns and target column (`Price`)
- Check basic understanding of value ranges (Area, rooms, price)

---

## Dividing Training and Testing
The dataset is split using `train_test_split`:
- Training set: model learns relationships
- Testing set: model is evaluated on unseen data

Used in code:
- `test_size=0.2` (80% train, 20% test)
- `random_state=42` (reproducible split)

---

## Model Selection
**Model used:** Linear Regression (`sklearn.linear_model.LinearRegression`)

Why Linear Regression:
- Simple, fast baseline model for regression
- Easy to interpret how inputs affect price
- Works well for linear relationships between features and target

---

## Evaluation Metric (Used in this Project)
This project uses **R² Score** (`r2_score`) to evaluate accuracy.

Simple meaning:
- R² tells how well the model explains house price variation
- Higher R² means predictions fit the real prices better

Used in code:
- `r2_score(y_test, y_pred)`

---

## Main Libraries Used (and why)

1. `pandas`  
   - Creates and manages the dataset as a DataFrame.

2. `numpy`  
   - Supports numerical operations and array handling.

3. `sklearn.model_selection.train_test_split`  
   - Splits data into train and test sets.

4. `sklearn.linear_model.LinearRegression`  
   - Trains the regression model.

5. `sklearn.metrics.r2_score`  
   - Evaluates model performance using R² score.

---

## Output
- Printed **R² Score** for model performance
- Predicted price for a new house input (example: `[1200, 3, 2]`)
  
---

## Developer
Grishma C.D
