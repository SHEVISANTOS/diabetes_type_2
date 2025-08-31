# Run this in your terminal
cat > README.md << 'EOL'
# Type 2 Diabetes Prediction

A machine learning model to predict Type 2 Diabetes using clinical and lifestyle data.

## 📊 Dataset
- Features: gender, age, BMI, HbA1c, blood glucose, smoking history
- Target: diabetes (0 = No, 1 = Yes)
- Source: Provided dataset (~1000+ patients)

## 🧪 Model Performance
| Model          | Accuracy | ROC-AUC |
|----------------|----------|--------|
| XGBoost        | 0.87     | 0.90   |

## 🚀 How to Run
```bash
git clone https://github.com/SHEVISANTOS/diabetes_type_2.git
pip install -r requirements.txt
jupyter notebook