# Titanic Survival Prediction

This project applies a supervised learning flow on the **Titanic dataset**.  
The task is to predict whether passengers survived (1) or not (0) based on features such as age, sex, ticket class, fare, and family size.  

---

## Workflow Summary
- **Dataset**: 891 passengers in training set, 418 in test set.  
- **EDA**: Looked at survival by gender, passenger class, and an engineered feature (FamilySize).  
- **Modeling**:  
  - Built a pipeline with preprocessing (scaling + one-hot encoding).  
  - Trained Logistic Regression with multiple hyperparameters.  
  - Used GridSearchCV with Stratified 5-Fold CV for model selection.  
- **Final Model**: Logistic Regression with the best configuration, retrained on the full training set.  
- **Evaluation**: Tested on the hold-out set, reported F1-score, and inspected the first 5 predictions.  

---

## Results
- Logistic Regression gave strong performance on both validation and test sets.  
- Gender and passenger class were the most important predictors.  
- Adding **FamilySize** improved the model’s understanding of survival chances.  

---

## Requirements
- Python 3.8+  
- pandas, numpy  
- scikit-learn  
- matplotlib, seaborn  


