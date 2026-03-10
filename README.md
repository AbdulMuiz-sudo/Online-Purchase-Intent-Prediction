# 🛒 Online Purchase Intent Prediction – Machine Learning Classifier

Welcome to the **Online Purchase Intent Prediction System** – a machine learning project designed to predict whether a visitor to an e-commerce website is likely to complete a purchase or leave without buying.

This model analyzes user browsing behavior, session statistics, and website interaction metrics to classify visitors into **buyers or non-buyers**, helping businesses optimize marketing strategies and improve conversion rates.

---

# 📌 Project Overview

Understanding customer intent is crucial for modern e-commerce platforms. Most visitors browse products but do not complete purchases. This project helps solve that problem by predicting **purchase intent based on user session data**.

This project focuses on:

* Analyzing customer browsing behavior from session data
* Performing **data preprocessing and feature engineering**
* Training multiple **machine learning classification models**
* Comparing model performance to identify the most accurate predictor
* Evaluating models using standard metrics like accuracy, precision, recall, and F1-score

The dataset used contains **12,330 user sessions**, where each row represents a single visit to an e-commerce website. Only about **15% of sessions end in a purchase**, making the dataset imbalanced. ([GitHub][1])

---

# ⚙️ Powered By

Python, Pandas, NumPy, Scikit-learn, Matplotlib, and Seaborn.

---

# 🖼️ Project Workflow

### 1️⃣ Data Exploration (EDA)

Exploratory Data Analysis was performed to understand the behavior of online shoppers.

Insights include:

* Distribution of purchasing vs non-purchasing sessions
* Relationship between **page visits, session duration, and purchase likelihood**
* Correlation between numerical features like bounce rate, exit rate, and page value
* Behavioral differences between **new and returning visitors**

---

### 2️⃣ Data Preprocessing

The dataset underwent several preprocessing steps before model training:

* Handling missing values
* Encoding categorical variables
* Normalizing numerical features
* Removing redundant or highly correlated features
* Splitting dataset into training and testing sets

---

# 🧠 Machine Learning Models Used

Several classification algorithms were trained and compared to determine the best-performing model:

* Logistic Regression
* Naive Bayes
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Random Forest
* Gradient Boosting
* AdaBoost

These models analyze session features such as:

* Page visit counts
* Time spent on different pages
* Bounce and exit rates
* Page value
* Visitor type
* Traffic source

to predict whether the **Revenue (purchase) variable will be True or False**. ([GitHub][1])

---

# 📊 Features Used

| Feature                 | Description                                           |
| ----------------------- | ----------------------------------------------------- |
| Administrative          | Number of administrative pages visited                |
| Informational           | Number of informational pages visited                 |
| ProductRelated          | Number of product-related pages viewed                |
| Administrative_Duration | Time spent on admin pages                             |
| Informational_Duration  | Time spent on informational pages                     |
| ProductRelated_Duration | Time spent on product pages                           |
| BounceRates             | Percentage of visitors leaving after viewing one page |
| ExitRates               | Percentage of exits from pages                        |
| PageValues              | Average value of pages visited                        |
| SpecialDay              | Closeness of visit date to special events             |
| VisitorType             | New or returning visitor                              |
| Weekend                 | Whether the visit occurred on a weekend               |

---

# 🧮 Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC Curve Analysis

The best-performing models typically included **ensemble methods such as Random Forest and Gradient Boosting**, which are effective for structured behavioral datasets.

---

# 🛠️ Tech Stack

* Python 3.x
* Pandas & NumPy
* Scikit-learn
* Matplotlib & Seaborn
* Jupyter Notebook
* Git & GitHub

---

# 🔍 How It Works

1. Load and explore the **Online Shoppers Intention Dataset**
2. Clean and preprocess the dataset
3. Encode categorical features and normalize numerical features
4. Split the dataset into **training and testing sets**
5. Train multiple classification models
6. Evaluate model performance using metrics
7. Select the best model to predict **purchase intent**

---

# 📦 Dataset

The dataset used in this project is the **Online Shoppers Purchasing Intention Dataset** from the **UCI Machine Learning Repository**.

It contains:

* **12,330 user sessions**
* **18 behavioral features**
* Binary target variable **Revenue (purchase or no purchase)**. ([GitHub][1])

---

# 📥 Installation

```bash
git clone https://github.com/AbdulMuiz-sudo/Online-Purchase-Intent-Prediction.git

cd Online-Purchase-Intent-Prediction
```

Open the notebook and run the cells to reproduce the analysis and model training.

