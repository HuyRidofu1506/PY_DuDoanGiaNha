# 🧠 PY_DuDoanGiaNha
## 📘 Overview
This project aims to build an AI-powered system for **predicting house prices** in **Quy Nhon City, Vietnam**. The system analyzes key features such as area, number of rooms, and location to estimate property values.

It uses machine learning and deep learning models to **classify** and **predict prices** based on real estate data collected in **Quy Nhon**. The goal is to provide a helpful tool for buyers, sellers, and real estate professionals to make informed decisions.

---

## 📂 Dataset
- **Language**: Vietnamese
- **Source**: Collected form real estate websites, surrounding area.
- **Download link**: [House Prices Dataset Quy Nhon City (Google Drive)](https://drive.google.com/drive/folders/1r47m7rB3b6fo-LszlC3QasA-H6M8bxsC?usp=sharing)

---

## 🤖 Supported Model
| Model               | Type              | Accuracy (Classification) | MAPE (Regression) |
|---------------------|-------------------|----------------------------|--------------------|
| Linear Regression   | Traditional ML    | 81.08%                     | 18.92%             |
| Random Forest       | Traditional ML    | 80.92%                     | 19.08%             |
| XGBoost             | Traditional ML    | 80.88%                     | 19.12%             |

> **Note:** Accuracy reflects classification over price tiers (e.g., low/medium/high), while **MAPE** evaluates the regression model’s prediction error in percentage.

---

## 🛠️ Technologies Used
- Programming Language: Python
- Key Libraries:
  -  `pandas`, `numpy` for data manipulation and analysis 
  -  `scikit-learn` for Linear Regression and Random Forest
  -  `xgboost` for XGBoost
  -  `Streamlit` for building a simple interactive web app (optional)
## ⚙️ Training Pipeline

1. **Data Collection & Cleaning**  
   - Scrape and aggregate listing data from multiple real estate websites in Quy Nhơn.  
   - Handle missing values, outliers, and inconsistent formatting.

2. **Feature Engineering**  
   - Extract numerical features:  
     - Area (m²)  
     - Number of bedrooms  
     - Number of bathrooms  
     - Construction year (e.g., 2015, 2020)  
   - Encode categorical features:  
     - Location (ward/district)

3. **Train/Test Split**  
   - Split the dataset into training (e.g., 80%) and validation/test (20%) sets.  
   - Ensure geographic distribution is balanced across splits so that the model generalizes across different neighborhoods of Quy Nhơn.

4. **Model Training**  
   - **Linear Regression**  
     - Fit a linear model on training features and continuous target (price).  
   - **Random Forest Regressor**  
     - Train an ensemble of decision trees with bootstrap sampling and averaged predictions.  
   - **XGBoost Regressor**  
     - Train a gradient-boosted tree ensemble to minimize squared error.

5. **Evaluation Metrics**  
   - Classification accuracy for price ranges (tiers)  
   - Regression metrics:
     - **MAPE** (Mean Absolute Percentage Error)
     - MAE (Mean Absolute Error)
     - RMSE (Root Mean Squared Error)

6. **Optional: Streamlit Web App**  
   - Build a simple Streamlit interface where users can input property features (area, number of rooms, location) and get an instant predicted price range or continuous estimate.  
   - Deploy locally or host on a simple cloud service for demonstration.

---

## 📊 Results & Conclusion
- **Linear Regression** yielded the best overall results with:
  - **MAPE**: 18.92%
  - **Accuracy**: 81.08%
- **Random Forest** followed closely with:
  - **MAPE**: 19.08%
  - **Accuracy**: 80.92%
- **XGBoost** showed acceptable baseline performance at
  - **MAPE**: 19.12% 
  - **Accuracy**: 80.88%
