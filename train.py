import argparse
import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    print("Egitim basliyor...")
    
    # Okuma
    train_data = pd.read_csv("/opt/ml/input/data/train/train.csv", header=None)
    val_data = pd.read_csv("/opt/ml/input/data/validation/test.csv", header=None)
    
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    X_val = val_data.iloc[:, :-1]
    y_val = val_data.iloc[:, -1]
    
    # Eğitim
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='error')
    model.fit(X_train, y_train)
    
    # Kayıt (XGBoost native format - Hata vermez)
    model.get_booster().save_model("/opt/ml/model/xgboost-model")
    print("✅ Model kaydedildi.")
