import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import os
import mlflow
import mlflow.xgboost
import joblib
import logging
import time

# Variables
file_path = 'stroke_dataset_encoded.csv'
model_name = "XGBoost"

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configura MLflow para usar un directorio local para el seguimiento
mlflow.set_tracking_uri("mlruns")

def load_and_preprocess_data(file_path):
    try:
        logging.info(f"Cargando y preprocesando el dataset desde {file_path}")
        data = pd.read_csv(file_path)
        X = data.drop(columns=['stroke'])
        y = data['stroke']

        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = dict(zip(np.unique(y), class_weights))

        logging.info(f"Distribución de clases: {dict(y.value_counts())}")
        logging.info(f"Pesos calculados para las clases: {class_weight_dict}")

        return X, y, X.columns, class_weight_dict
    except Exception as e:
        logging.error(f"Error en la carga y preprocesamiento de datos: {str(e)}")
        raise

def apply_smote(X_train, y_train):
    try:
        start_time = time.time()
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        end_time = time.time()
        logging.info(f"Distribución de clases después de SMOTE: {dict(pd.Series(y_resampled).value_counts())}")
        execution_time = end_time - start_time
        print(f"Tiempo de ejecución de SMOTE: {execution_time:.2f} segundos")
        return X_resampled, y_resampled
    except Exception as e:
        logging.error(f"Error al aplicar SMOTE: {str(e)}")
        raise

def train_and_evaluate_model(X_train, X_test, y_train, y_test, class_weight_dict):
    try:
        params = {
            'max_depth': 5,
            'learning_rate': 0.01,
            'gamma': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': class_weight_dict[1] / class_weight_dict[0],  # Invertir para clase minoritaria
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }

        model = XGBClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        best_threshold = find_best_threshold(y_test, y_pred_proba)
        y_pred_adjusted = (y_pred_proba >= best_threshold).astype(int)

        metrics = calculate_metrics(y_test, y_pred_adjusted, y_pred_proba)

        return model, metrics, y_pred_adjusted, y_pred_proba, params, best_threshold
    except Exception as e:
        logging.error(f"Error en el entrenamiento y evaluación del modelo: {str(e)}")
        raise

def find_best_threshold(y_true, y_pred_proba):
    thresholds = np.arange(0, 1, 0.01)
    f1_scores = [f1_score(y_true, (y_pred_proba >= threshold).astype(int)) for threshold in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    logging.info(f"Mejor umbral encontrado: {best_threshold}")
    return best_threshold

def calculate_metrics(y_true, y_pred, y_pred_proba):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_pred_proba)
    }

def log_mlflow(model, metrics, model_name, params):
    try:
        with mlflow.start_run():
            mlflow.log_params(params)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            mlflow.xgboost.log_model(model, model_name)
        logging.info(f"Métricas del modelo {model_name} registradas en MLflow.")
    except Exception as e:
        logging.error(f"Error al registrar en MLflow: {str(e)}")
        raise

def plot_feature_importance(model, feature_names):
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_features = feature_names[indices][:10]
        top_importances = importances[indices][:10]
        
        plt.figure(figsize=(10, 6))
        plt.title("Top 10 Características Más Importantes")
        plt.bar(range(10), top_importances)
        plt.xticks(range(10), top_features, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('reports/figures/feature_importance.png')
        plt.close()
        logging.info("Gráfico de importancia de características guardado.")
    except Exception as e:
        logging.error(f"Error al generar el gráfico de importancia de características: {str(e)}")
        raise

if __name__ == "__main__":
    # Cargar y preprocesar los datos
    X, y, feature_names, class_weight_dict = load_and_preprocess_data(file_path)

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Aplicar SMOTE solo en el conjunto de entrenamiento
    X_resampled, y_resampled = apply_smote(X_train, y_train)

    # Entrenar y evaluar el modelo
    model, metrics, y_pred_adjusted, y_pred_proba, params, best_threshold = train_and_evaluate_model(X_resampled, X_test, y_resampled, y_test, class_weight_dict)

    # Guardar modelo
    model_dir = f'models/{model_name.lower().replace(" ", "_")}'
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f'{model_dir}/model_{model_name.lower().replace(" ", "_")}.pkl')
    print(f"Modelo guardado en '{model_dir}'.")

    # Registrar los resultados en MLflow
    log_mlflow(model, metrics, model_name, params)

    # Graficar la importancia de las características
    plot_feature_importance(model, feature_names)

    # Imprimir métricas
    print("Métricas del modelo:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name.capitalize()}: {metric_value:.4f}")
