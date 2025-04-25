# 🩺 Disease to Symptoms Detector

## Introduction

The **Disease to Symptoms Detector** is a machine learning-based application designed to predict potential diseases based on user-reported symptoms. Utilizing a multi-label classification approach, the system can identify multiple diseases simultaneously, providing users with preliminary insights into possible health conditions.

## Approach

The project follows a systematic pipeline:

1. **Data Preprocessing**: Cleaning and preparing the dataset for analysis.
2. **Feature Engineering**: Transforming categorical disease labels into binary format for multi-label classification.
3. **Model Training**: Employing decision tree-based classifiers to learn patterns from the data.
4. **Evaluation**: Assessing model performance using appropriate metrics.

## Methods

### Data Preprocessing

The dataset underwent several preprocessing steps:

- **Duplicate Removal**: Eliminated duplicate columns to ensure data integrity.

  ```python
  df = df.loc[:, ~df.columns.duplicated()]
  ```

- **Handling Missing Values**: Addressed missing data by filling with mean values for numerical features.

  ```python
  df.fillna(df.mean(), inplace=True)
  ```

- **Binary Encoding**: Converted disease labels into binary columns to facilitate multi-label classification.

  ```python
  from sklearn.preprocessing import MultiLabelBinarizer

  mlb = MultiLabelBinarizer()
  y = mlb.fit_transform(df['disease_column'])
  ```

### Feature Selection

Selected relevant symptom features as input variables (`X`) and the binary-encoded diseases as target variables (`y`).

```python
X = df.drop('disease_column', axis=1)
```

### Train-Test Split

Divided the dataset into training and testing subsets to evaluate model performance.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Model Training

Utilized a Decision Tree Classifier with parameters tuned to prevent overfitting.

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42)
model.fit(X_train, y_train)
```

### Evaluation

Assessed the model using accuracy and a detailed classification report.

```python
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))
```

### Overfitting Prevention

Implemented pruning techniques and parameter tuning to enhance model generalization.

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(ccp_alpha=0.01, random_state=42)
model.fit(X_train, y_train)
```

## Results

The Decision Tree Classifier achieved an accuracy of **0.90** on the test set. The classification report indicated satisfactory precision and recall across multiple disease categories, demonstrating the model's capability in multi-label disease prediction.



## Conclusion

The **Disease to Symptoms Detector** effectively leverages machine learning techniques to predict potential diseases based on symptoms. Through careful preprocessing, feature engineering, and model tuning, the system provides a valuable tool for preliminary health assessment. Future enhancements may include integrating more complex models, expanding the dataset, and developing a user-friendly interface for broader accessibility.

