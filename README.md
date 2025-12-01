# AWS SageMaker ile Uçtan Uca MLOps Pipeline Projesi

## 🎯 Projenin Amacı
Bu proje, bulut tabanlı bir makine öğrenmesi sisteminin, veri değişikliği (Data Drift) durumunda kendini otomatik olarak nasıl iyileştirebileceğini (Self-Healing) simüle eder.

## 🛠 Kullanılan Teknolojiler
* **Orchestration:** AWS SageMaker Pipelines
* **Compute:** AWS SageMaker Processing & Training Jobs
* **Model:** XGBoost (Binary Classification)
* **Infrastructure:** Python SDK (Boto3, Sagemaker)
* **Storage:** Amazon S3

## 🔄 Sistem Mimarisi
1.  **Veri İşleme:** Ham veri S3'ten alınır, temizlenir ve Train/Test olarak ayrılır.
2.  **Eğitim:** XGBoost modeli eğitilir.
3.  **Kayıt:** Eğitilen model S3'e ve Model Registry'ye kaydedilir.
4.  **Dağıtım (Deployment):** Model canlı bir Endpoint sunucusuna yüklenir.
