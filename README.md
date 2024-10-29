![image](https://github.com/user-attachments/assets/ed6d1dc5-602d-46f7-ac90-c90ae1fc04cd)

# Stroke Risk Predictor & CT Image Classifier

# Estructura de Carpetas del Proyecto

```plaintext
stroke_prediction_project/
├── models/
├── notebooks/
├── static/
├── stroke_app/
│   ├── migrations/
│   ├── templates/stroke_app/
│   │   ├── base.html
│   │   ├── prediction_form.html
│   │   ├── reports.html
│   │   ├── success.html
│   │   ├── upload_image.html
│   │   └── upload.html
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── forms.py
│   ├── models.py
│   ├── tests.py
│   ├── urls.py
│   └── views.py
├── stroke_project/
│   ├── __init__.py
│   ├── asgi.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── .dockerignore
├── .env
├── docker-compose.yml
├── dockerfile
├── manage.py
├── .gitignore
├── README.md
└── requirements.txt
```

# Stroke Prediction and Image Classification App
Aplicación integral para predecir el riesgo de ictus y clasificar imágenes de tomografías computarizadas (CT) mediante algoritmos de regresión logística y redes neuronales convolucionales (CNN). Diseñada para facilitar el análisis médico, ofrece predicciones personalizadas y clasificación de imágenes con alta precisión, además de informes detallados y acceso controlado para profesionales de la salud.

## Descripción General
Esta aplicación está diseñada para la predicción de riesgo de ictus y la clasificación de imágenes médicas. Utiliza algoritmos de machine learning avanzados, incluyendo regresión logística y redes neuronales convolucionales (CNN), proporcionando un entorno integral para profesionales de la salud que desean realizar predicciones rápidas y clasificar imágenes médicas con precisión.

## Funcionalidades Principales

### 1. Predicción de Ictus

<img align="right" width="250" alt="image" src="https://github.com/user-attachments/assets/7bf25396-564e-4f8b-be81-925926ef55fb">
La aplicación permite predecir el riesgo de ictus en pacientes basándose en 10 variables clave ingresadas a través de un formulario. Estas variables incluyen:

- **age**: Edad del paciente
- **avg_glucose_level**: Nivel promedio de glucosa
- **bmi**: Índice de masa corporal
- **hypertension**: Presencia de hipertensión
- **heart_disease**: Historial de enfermedad cardíaca
- **gender**: Género del paciente
- **ever_married**: Estado civil
- **work_type**: Tipo de trabajo
- **Residence_type**: Tipo de residencia
- **smoking_status**: Estado de tabaquismo

Al hacer clic en el botón de "Predict", se muestra un modal con el resultado de riesgo de ictus (alto o bajo), junto con la probabilidad estimada y algunas recomendaciones según el resultado obtenido.

### 2. Clasificación de Imágenes
Utilizando las librerías **TensorFlow** y **Keras**, la aplicación permite la clasificación de imágenes de tomografías computarizadas (TC). Está entrenada en un conjunto de datos de 2500 imágenes, distribuidas en:
- 1500 imágenes sin indicios de ictus.
- 1000 imágenes con indicios de ictus.

El modelo de CNN ha alcanzado un **95% de precisión** en el conjunto de imágenes de prueba, lo que garantiza un rendimiento confiable para la clasificación de imágenes médicas.

### 3. Carga Masiva
La aplicación permite la carga masiva de datos, lo cual facilita el procesamiento de múltiples registros a la vez, optimizando el tiempo de carga y análisis para los usuarios.

### 4. Informes
Incluye una sección de informes en la que se muestran las métricas del modelo de **regresión logística** utilizado para la predicción de ictus. En esta sección se presentan diferentes gráficos y métricas del modelo, como la matriz de confusión, precisión y sensibilidad. Este enlace de informes está protegido por **control de acceso basado en roles** (RBAC), implementado con Django, de modo que solo usuarios autorizados pueden acceder a información relevante para su rol profesional.

### 5. Autenticación de Usuario
La aplicación incluye funcionalidades de **login/logout** que permiten a los usuarios registrados acceder a secciones protegidas, como los informes detallados de las predicciones y clasificación. Los usuarios no autenticados solo podrán acceder a funcionalidades básicas de la aplicación.

## Requerimientos
- **Python >= 3.7**
- Librerías: `scikit-learn`, `tensorflow`, `keras`, `django`, `numpy`, `pandas`, `matplotlib`, entre otras.

## Instalación y Ejecución
1. Clonar el repositorio.
   ```bash
   git clone <repo-url>
   cd stroke_prediction_project

## Instalar las dependencias:
`pip install -r requirements.txt`

## Ejecutar la aplicación
`python manage.py runserver`
