# ModelsforEvaluatingMetrices
**Problem Statement**
- Implement multiple classification models
- Build an interactive Streamlit web application to demonstrate models

**Dataset**
- Source : Kaggle
- Size : 955 KB
- Features : 21 customer dimensions (including 1 unique identifier customerID, 19 predictive traits, and 1 binary   classification target Churn).
- Rows (Observations): 7,043 unique customer profiles.
- Class Distribution : 2
  (Customers who stayed with the company - 73.46%)
  (Customers who left the company (Minority Class) - 26.54%)

  **Github Repository Link**
  https://github.com/Arundhati-Bose/ModelsforEvaluatingMetrices
  
  **Models to be Implemented**
  1. Logistic Regression
  2. Decision Tree Classifier
  3. K-Nearest Neighbor Classifier
  4. Naive Bayes Classifier - Gaussian or Multinomial
  5. Ensemble Model - Random Forest

  **Evaluation Metrices for each model**
  1. Accuracy
  2. AUC Score
  3. Precision
  4. Recall
  5. F1 Score
  6. Matthews Correlation Coefficient (MCC Score)
 
  **Observation Table**
 -Logistic Regression
  1. Accuracy - 0.807
  2. AUC Score - 0.8416
  3. Precision - 0.6584
  4. Recall - 0.5668
  5. F1 Score - 0.6092
  6. Matthews Correlation Coefficient (MCC Score) - 0.4843
 
  - Decision Tree Classifier
   1. Accuracy - 0.7346
   2. AUC Score - 0.8308
   3. Precision - 0.5
   4. Recall - 0.8075
   5. F1 Score - 0.6176
   6. Matthews Correlation Coefficient (MCC Score) - 0.4601
 
  -K-Nearest Neighbor Classifier
   1. Accuracy - 0.7736
   2. AUC Score - 0.8067
   3. Precision - 0.577
   4. Recall - 0.5508
   5. F1 Score - 0.5636
   6. Matthews Correlation Coefficient (MCC Score) - 0.4111
 
  -Naive Bayes Classifier - Gaussian or Multinomial
   1. Accuracy - 0.6558
   2. AUC Score - 0.8093
   3. Precision - 0.4269
   4. Recall - 0.8663
   5. F1 Score - 0.5719
   6. Matthews Correlation Coefficient (MCC Score) - 0.3951
 
  -Ensemble Model - Random Forest
   1. Accuracy - 0.7622
   2. AUC Score - 0.8436
   3. Precision - 0.5358
   4. Recall - 0.7807
   5. F1 Score - 0.6355
   6. Matthews Correlation Coefficient (MCC Score) - 0.4863
 
  **Overall Winner**
  Random Forest
  - Highest Predictive Balance: It achieves the highest F1 Score (0.6355) and the highest MCC Score (0.4863). MCC is the   most reliable metric for imbalanced data like this customer churn dataset.
  - Top Discriminative Power: It shares the highest AUC Score (0.8436) with Logistic Regression, meaning it is excellent at separating churners from non-churners.
  - Strong Catch Rate: It catches 78.07% of churners (Recall) while maintaining a decent Accuracy (0.7622).




  
