# House Price Prediction (Regression Project)

## Project Summary

This project aims to build a machine learning model capable of predicting the sale price for residential houses in Ames, Iowa. The dataset is obtained from a Kaggle competition, includes 80 features describing nearly every aspect of a home, from material quality and area size to proximity to amenities. (accessible through "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data")

The primary challenge of this project are **Data Cleaning** (handling widespread missing data) and **Feature Engineering** to transform complex, raw data into a format ready for a machine learning model.

This project utilizes a **Linear Regression** model as a baseline and compares it against a **Random Forest** model to find the most accurate predictive model.

---

## Project Workflow

1.  **Data Loading:** Load the `train.csv` (Ames Housing) dataset directly from a URL or upload the dataset to Google Colab.
2.  **Data Cleaning (Missing Data Handling):**
    * **"NA" means "None":** For categorical features like `PoolQC`, `Alley`, `Fence`, and `FireplaceQu`, `NA` (missing) values were filled with the string `"None"` to indicate the house *lacks* the feature.
    * **"NA" means "0":** For numerical features like `MasVnrArea` or `GarageArea`, `NA` was filled with `0`, as the absence of the feature implies a value of zero.
    * **Other Imputation:**
        * `LotFrontage`: `NA` values were filled with the **median** `LotFrontage` of their corresponding `Neighborhood`.
        * `Electrical`: `NA` value was filled with the **mode** (most common value).
3.  **Exploratory Data Analysis (EDA):**
    * **Target Analysis (`SalePrice`):** The `SalePrice` was heavily right-skewed (Skewness $\approx$ 1.88).
    * **Target Transformation:** Applied a **Log Transform** (`np.log1p()`) to `SalePrice` to normalize its distribution.
    * **Correlation Analysis:** Created a heatmap to find the top numerical features correlated with `SalePrice_log`, such as `OverallQual`, `GrLivArea`, and `GarageCars`.
    * **Outlier Removal:** Identified and removed 2 extreme outliers (houses with `GrLivArea` > 4000 but inexplicably low prices) that would have skewed the regression model.
4.  **Feature Engineering:**
    * **One-Hot Encoding:** Applied `pd.get_dummies()` to convert *all* categorical features (e.g., `Neighborhood`, `Street`, `KitchenQual`) into numerical (0/1) columns. This expanded the dataset from $\approx$ 80 features to 290+ columns.
5.  **Modeling & Scaling:**
    * **Train/Test Split:** Split the data into 80% training and 20% validation/test sets.
    * **StandardScaler:** Applied `StandardScaler` to the feature sets (X_train, X_test) to ensure all features operate on a similar scale.
    * **Model 1 (Baseline):** Trained a `LinearRegression` model.
    * **Model 2 (Complex):** Trained a `RandomForestRegressor` model.
6.  **Model Evaluation:**
    * Evaluated both models using **Root Mean Squared Error (RMSE)** on the dollar amount (after converting log predictions back to dollars).

---

## Results & Key Findings

### Model Comparison

The final models were evaluated based on their ability to predict the actual sale price, with the error measured in dollars.

| Model | RMSE (in Dollars) | Conclusion |
| :--- | :--- | :--- |
| **Linear Regression** | **$\approx$ $20,866.7** | **The Lower RMSE, Better Model** |
| **Random Forest** | $\approx$ $25,560.6 | *The RMSE is higher than Linear Regression*. |

The simpler `LinearRegression` model was the clear winner, achieving an **average prediction error of $\approx$ $20,867**. This is a very strong baseline result.

### Most Important Price Drivers (from Linear Model)

By analyzing the coefficients (`model.coef_`) of the winning Linear Regression model, the most significant factors driving price are:

**Top Factors (Increasing Price):**
1.  **`GrLivArea`:** The above-ground living area size.
2.  **`YearBuilt`:** The year the house was built (newer houses are more expensive).
3.  **`PoolArea`:** Having a pool (large positive impact).
4.  **`OverallQual`:** The overall quality of materials and finish.

**Top Factors (Decreasing Price):**
1.  **`GarageCond_None`:** A feature that means **"No Garage"**.
2.  **`GarageQual_None`:** Also indicates **"No Garage"**.
3.  **`Condition2_RRAe`:** A feature indicating the house is **near an East-West Railroad**.
