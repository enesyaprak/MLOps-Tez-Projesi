import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    print("Veri isleme basliyor...")
    base_dir = "/opt/ml/processing"
    
    # Klasörleri oluştur
    os.makedirs(f"{base_dir}/train", exist_ok=True)
    os.makedirs(f"{base_dir}/test", exist_ok=True)
    
    # Veriyi oku (Boşlukları temizleyerek)
    df = pd.read_csv(f"{base_dir}/input/adult.csv", header=None, skipinitialspace=True)
    
    # Sütun isimleri
    columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
               "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
               "hours-per-week", "native-country", "income"]
    df.columns = columns

    # Hedef değişkeni temizle
    df['income'] = df['income'].astype(str).str.strip().apply(lambda x: 1 if '>50K' in x else 0)
    
    # Sadece sayısal sütunları al
    df = df.replace('?', np.nan).dropna()
    df = df.select_dtypes(include=['int64', 'float64'])
    
    # Train/Test ayır
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    
    # Kaydet
    train.to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    test.to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
    print(f"✅ Tamamlandı. Train: {len(train)}, Test: {len(test)}")
