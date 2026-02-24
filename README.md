#  Mushroom Classification using Decision Tree

##  Project Overview

This project builds a **Machine Learning classification model** to determine whether a mushroom is **edible or poisonous** using the UCI Mushroom Dataset.

The goal is to demonstrate a complete ML workflow including data preprocessing, feature engineering, correlation analysis, and model training using a Decision Tree classifier.

---

##  Problem Statement

Incorrect mushroom identification can be dangerous.
This project applies supervised machine learning to automatically classify mushrooms based on their physical characteristics.

---

##  Dataset

* Source: UCI Machine Learning Repository
* Samples: 8,124 mushrooms
* Features: 22 categorical attributes
* Classes:

  * `e` → edible
  * `p` → poisonous

Dataset link:
https://archive.ics.uci.edu/dataset/73/mushroom

---

##  Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

---

##  Machine Learning Pipeline

1. Data loading and inspection
2. Handling categorical features
3. One-Hot Encoding (`pd.get_dummies`)
4. Correlation analysis (heatmap)
5. Removal of collinear features
6. Train/Test split
7. Decision Tree training (Entropy criterion)
8. Model evaluation

---

##  Model

**Algorithm:** Decision Tree Classifier

Parameters:

* Criterion: Entropy
* Max depth: 3
* Minimum samples per leaf: 5

---

##  Results

* High classification accuracy achieved on test data.
* Confusion matrix used for evaluation.
* Feature correlation analysis improved model robustness.

---

##  Feature Correlation Heatmap

![Heatmap](images/heatmap.png)

---

##  How to Run the Project

### 1️ Clone repository

```bash
git clone https://github.com/kianaabdollahzadeh/decision-tree.git
cd decision-tree
```

### 2️ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️ Run the model

```bash
python src/main.py
```

---

##  Project Structure

```
decision-tree/
│
├── src/
│   └── main.py
├── data/
│   └── agaricus-lepiota.data
├── images/
│   └── heatmap.png
├── requirements.txt
└── README.md
```

---

##  Learning Outcomes

* Practical ML preprocessing workflow
* Feature engineering for categorical datasets
* Detecting multicollinearity
* Model evaluation using accuracy and confusion matrix
* End-to-end ML project structuring

---

##  Author

**Kiana Abdollahzadeh**
Computer Science Student | Aspiring AI Researcher

---

##  Future Improvements

* Compare multiple classifiers (Random Forest, Logistic Regression)
* Hyperparameter tuning
* Cross-validation
* Model visualization
