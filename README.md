
# Credit Card Fraud Detection using Isolation Forest

This project implements an **unsupervised machine learning approach** to detect credit card fraud using the **Isolation Forest** algorithm. The dataset used contains real-world credit card transactions, where the goal is to detect the rare and unusual fraudulent activities hidden among the legitimate ones.
##  Dataset

The dataset used is the popular [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle, which contains:
* 284,807 transactions
* 492 frauds (only \~0.172% of all transactions)
* Features are numerical results of a PCA transformation, except for `Time` and `Amount`
##  Objective

To build an **unsupervised anomaly detection system** that can:
* Learn the patterns of normal transactions
* Detect potential fraudulent transactions without needing labeled training data
## Model: Isolation Forest

* **Algorithm**: [Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
* **Why?**: It's designed for anomaly detection and works well when fraud data is highly imbalanced
* **Contamination**: Set to 0.001 (based on the proportion of frauds in the dataset)
##  Steps Performed
1. **Preprocessing**

   * Standardized `Amount`
   * Dropped `Time` feature (not relevant for unsupervised training)

2. **Model Training**

   * Isolation Forest trained on all features (unsupervised)

3. **Evaluation**

   * Converted model predictions to binary (fraud = 1, normal = 0)
   * Evaluated using confusion matrix and classification report

4. **Visualization**

   * Heatmap of confusion matrix
   * KDE plot comparing log-transformed transaction amounts of fraud vs. normal transactions
## Results

* **Evaluation Metrics**: Precision, Recall, F1-score
* **Visualizations**:

  * Confusion matrix for performance overview
  * Distribution plots showing how fraud transactions differ in amount pattern
##  Output

### 📋 **Classification Report**

```
              precision    recall  f1-score   support

           0       1.00      0.99      1.00    284315
           1       0.10      0.84      0.18       492

    accuracy                           0.99    284807
   macro avg       0.55      0.92      0.59    284807
weighted avg       1.00      0.99      1.00    284807
```

> ✅ The model achieves **high recall (84%)** on fraud cases, which is important for identifying fraudulent transactions. However, due to the unsupervised nature and class imbalance, the **precision is low (10%)**, meaning there are false positives — but fewer missed frauds.

---

### 📊 **Confusion Matrix**

|                   | Predicted Normal | Predicted Fraud |
| ----------------- | ---------------- | --------------- |
| **Actual Normal** | 281857           | 2458            |
| **Actual Fraud**  | 79               | 413             |

* **True Positives (Frauds correctly identified)**: 413
* **False Positives (Legitimate flagged as fraud)**: 2458
* **False Negatives (Fraud missed)**: 79
* **True Negatives (Legitimate correctly identified)**: 281857

---

###  **Transaction Amount Distribution (Log Transformed)**


* Fraud transactions tend to cluster in **lower amount ranges**, while normal transactions are spread more widely.
* The log transformation helps visualize this clearly despite large value differences.

---

If you need help generating or saving the actual confusion matrix and KDE plot images (`conf_matrix.png`, `amount_kde.png`) to include in your GitHub repo, let me know—I can provide code for that too.



##  Libraries Used

* `pandas`, `numpy` – Data manipulation
* `scikit-learn` – Machine learning and preprocessing
* `matplotlib`, `seaborn` – Data visualization

## 📁 File Structure

```
credit-card-fraud-detection/
│
├── creditcard.csv                # Dataset
├── fraud_detection_isolation.py  # Main script
├── README.md                     # Project readme
└── plots/                        # Generated plots
```

