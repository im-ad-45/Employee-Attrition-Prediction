# Employee Attrition Prediction using Decision Tree and Random Forest Classification

**Author:** Aditya Shukla

**Registration Number:** 23BAI10155

**Application Number:** IN26011099

**Batch Number:** 1(A)

**Email ID:** aditya.23bai10155@vitbhopal.ac.in

## Objective
The objective of this project is to build Decision Tree and Random Forest classification models to predict employee attrition based on demographic, professional, and work-related attributes, and to compare their performance.

## Dataset Link
- [Kaggle: IBM HR Analytics Employee Attrition & Performance Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

## Libraries Used
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Methodology
1. **Data Understanding**: Loaded the dataset (1470 employees, 35 columns) and identified numerical features, categorical features, and `Attrition` as the target variable. Inspected structure and summary statistics using `.info()` and `.describe()`.
2. **Data Preprocessing**:
   - Checked for missing values (none found).
   - Dropped `EmployeeCount`, `StandardHours`, and `Over18` (constant for every row) and `EmployeeNumber` (just an ID), since none carry predictive signal.
   - Encoded the target variable (`Yes` -> 1, `No` -> 0).
   - One-hot encoded the remaining categorical features (`BusinessTravel`, `Department`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`, `OverTime`).
   - Split the dataset into 80% training and 20% testing using a stratified `train_test_split`.
3. **Model Development**:
   - **Model 1**: Trained a `DecisionTreeClassifier`.
   - **Model 2**: Trained a `RandomForestClassifier` with 100 estimators.
   - Both models were trained on the same training data and used to predict attrition on the test set.
4. **Model Evaluation**: Evaluated both models using Accuracy, Precision, Recall, and F1-Score, visualized both with Confusion Matrix heatmaps, and plotted the top 10 feature importances from the Random Forest model.

## Results

| Metric      | Decision Tree | Random Forest |
|-------------|---------------|---------------|
| Accuracy    | 0.7653        | 0.8333        |
| Precision   | 0.3103        | 0.4167        |
| Recall      | 0.3830        | 0.1064        |
| F1-Score    | 0.3429        | 0.1695        |

## Model Comparison
Random Forest achieved higher accuracy and precision, but the Decision Tree achieved noticeably higher recall and F1-score. This isn't the usual "Random Forest always wins" story — it's a direct effect of class imbalance: only about 16% of employees in this dataset actually left the company, so both models lean toward predicting the majority class ("No"), and Random Forest leans harder in that direction than a single tree does here. The Decision Tree caught more of the true attrition cases (higher recall), at the cost of more false positives. Feature importance from the Random Forest ranked `OverTime`, `MonthlyIncome`, and `Age` among the strongest predictors, consistent with common HR attrition intuition.

## Conclusion
This project compared a Decision Tree and a Random Forest (100 estimators) classifier for predicting employee attrition. Which model "performed better" depends on the metric: Random Forest achieved higher accuracy and precision, but the Decision Tree achieved higher recall and F1-score, catching more of the employees who actually left at the cost of more false alarms. This is a useful reminder that Random Forest does not automatically outperform a single tree on every metric, especially on an imbalanced dataset like this one. In general, Random Forest tends to outperform a single Decision Tree because it builds many trees on bootstrapped samples and random feature subsets, then averages their predictions, which cancels out the noise any one tree picks up from its specific training split. A key limitation of Decision Trees is that they are prone to overfitting, memorizing quirks of the training data rather than learning generalizable patterns. A key limitation of Random Forest is reduced interpretability and higher computational cost, and, as seen here, its majority-vote behavior can make it less sensitive to a minority class unless that imbalance is explicitly addressed.
