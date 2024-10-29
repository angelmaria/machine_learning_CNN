from django.shortcuts import render, redirect
from stroke_app.forms import StrokePredictionForm
from stroke_app.models import StrokePrediction
from django.conf import settings
from django.utils import timezone
import pandas as pd
import joblib
import numpy as np
from django.core.exceptions import ValidationError
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import threading
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
import io

def load_model():
    model_path = settings.MODEL_PATH
    try:
        pipeline = joblib.load(model_path)
        return pipeline['model']  # Extraemos solo el modelo del diccionario
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

def predict_stroke(request):
    form = StrokePredictionForm()
    context = {'form': form}
    
    if request.method == 'POST':
        print(request.POST)
        form = StrokePredictionForm(request.POST)
        if form.is_valid():
            try:
                # Convertir Yes/No a valores booleanos para la base de datos
                hypertension_bool = form.cleaned_data['hypertension'] == 'Yes'
                heart_disease_bool = form.cleaned_data['heart_disease'] == 'Yes'
                ever_married_bool = form.cleaned_data['ever_married'] == 'Yes'
                
                # Preparar datos para el modelo
                input_data = pd.DataFrame([{
                    'gender': form.cleaned_data['gender'],
                    'age': form.cleaned_data['age'],
                    'hypertension': 1 if hypertension_bool else 0,  # El modelo espera 1/0
                    'heart_disease': 1 if heart_disease_bool else 0,  # El modelo espera 1/0
                    'ever_married': form.cleaned_data['ever_married'],
                    'work_type': form.cleaned_data['work_type'],
                    'Residence_type': form.cleaned_data['Residence_type'],
                    'avg_glucose_level': form.cleaned_data['avg_glucose_level'],
                    'bmi': form.cleaned_data['bmi'],
                    'smoking_status': form.cleaned_data['smoking_status']
                }])

                try:
                    # Realizar predicción
                    prediction_proba = model.predict_proba(input_data)[0]
                    prediction = model.predict(input_data)[0]
                except Exception as e:
                    print(f"Error en la predicción: {e}")
                    context['error'] = str(e)
                
                # Determinar el riesgo y la probabilidad
                stroke_risk = 'High' if prediction == 1 else 'Low'
                risk_probability = f"{prediction_proba[1]:.2%}"
                
                # Guardar en la base de datos
                stroke_prediction = StrokePrediction.objects.create(
                    age=form.cleaned_data['age'],
                    hypertension=hypertension_bool,  # Guardamos el booleano
                    heart_disease=heart_disease_bool,  # Guardamos el booleano
                    avg_glucose_level=form.cleaned_data['avg_glucose_level'],
                    bmi=form.cleaned_data['bmi'],
                    gender=form.cleaned_data['gender'],
                    ever_married=ever_married_bool,  # Guardamos el booleano
                    work_type=form.cleaned_data['work_type'],
                    Residence_type=form.cleaned_data['Residence_type'],
                    smoking_status=form.cleaned_data['smoking_status'],
                    stroke_risk=stroke_risk,
                    date_submitted=timezone.now()
                )
                
                # Actualizar contexto con los resultados
                context.update({
                    'risk': stroke_risk,
                    'probability': risk_probability,
                    'show_result': True
                })
                
                print("Predicción realizada exitosamente")
                print(f"Riesgo: {stroke_risk}")
                print(f"Probabilidad: {risk_probability}")
                print("Datos guardados en la base de datos")
                
            except Exception as e:
                print(f"Error detallado: {e}")
                context['error'] = str(e)
        else:
            print("Errores en el formulario:", form.errors)
            context['error'] = "Por favor, corrija los errores en el formulario."
    
    return render(request, 'stroke_app/prediction_form.html', context)

# Función para la carga masiva de registros en un csv
def upload_csv_and_predict(request):
    if request.method == 'POST' and request.FILES['file']:
        file = request.FILES['file']
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return render(request, 'stroke_app/upload.html', {'error': f'Error al cargar el archivo CSV: {e}'})

        required_columns = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
        if all(col in df.columns for col in required_columns):
            # Asegúrate de que las columnas estén en el mismo orden y tipo que el modelo espera
            input_data = df[required_columns]  # Asegúrate de que solo las columnas necesarias estén presentes
            
             # Convierte "Yes" y "No" a valores booleanos en la columna 'ever_married'
            df['ever_married'] = df['ever_married'].map({'Yes': True, 'No': False})
            
            predictions = model.predict(input_data)

            # Agregar predicciones al DataFrame
            df['stroke_risk'] = predictions
            
            # Almacenar los resultados en la base de datos
            for _, row in df.iterrows():
                StrokePrediction.objects.create(
                    gender=row['gender'],
                    age=row['age'],
                    hypertension=row['hypertension'],
                    heart_disease=row['heart_disease'],
                    ever_married=row['ever_married'],
                    work_type=row['work_type'],
                    Residence_type=row['Residence_type'],
                    avg_glucose_level=row['avg_glucose_level'],
                    bmi=row['bmi'],
                    smoking_status=row['smoking_status'],
                    stroke_risk='High' if row['stroke_risk'] == 1 else 'Low',
                    date_submitted=timezone.now()
                )

            return redirect('success_page')
        else:
            return render(request, 'stroke_app/upload.html', {'error': 'Faltan columnas en el archivo CSV'})

    return render(request, 'stroke_app/upload.html')

# Función para la creación de los gráficos
def create_plots(data, plots, total_records, stroke_cases):
    # Gráfico 1: Distribución de Stroke por Rango de Edad
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data[data['stroke_risk'] == 'High'], x='age', bins=20, kde=True, color='skyblue')
    plt.title("Cantidad de Pacientes con Alto Riesgo de Ictus por Edad", fontweight='bold')
    plt.xlabel("Edad")
    plt.ylabel("Cantidad de Casos de Ictus")
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plots['age_stroke_plot'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    # Gráfico 2: Distribución de Stroke por Nivel de Glucosa
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data[data['stroke_risk'] == 'High'], x='avg_glucose_level', bins=20, kde=True, color='skyblue')
    plt.title("Cantidad de Pacientes con Alto Riesgo de Ictus por Nivel de Glucosa", fontweight='bold')
    plt.xlabel("Nivel de Glucosa Promedio")
    plt.ylabel("Cantidad de Casos de Ictus")
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plots['glucose_stroke_plot'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    # Gráfico 3: Total de Registros vs Casos de Ictus
    plt.figure(figsize=(8, 5))
    x_labels = ['Total Registros', 'Casos de Ictus']
    y_values = [total_records, stroke_cases]
    sns.barplot(x=x_labels, y=y_values, palette=['#4a90e2', '#d9534f'])
    plt.title("Total de Registros y Casos de Ictus", fontweight='bold')
    plt.ylabel("Cantidad")
    plt.xlabel("Categorías")
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plots['stroke_percentage_plot'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    # Gráfico 4: Distribución de Stroke por Hipertensión
    plt.figure(figsize=(8, 5))
    sns.countplot(data=data[data['stroke_risk'] == 'High'], x='hypertension', palette='Blues')
    plt.title("Cantidad de Pacientes con Alto Riesgo de Ictus y Hipertensión", fontweight='bold')
    plt.xlabel("Hipertensión (0 = No, 1 = Sí)")
    plt.ylabel("Cantidad de Casos de Ictus")
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plots['hypertension_stroke_plot'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    # Gráfico 5: Distribución de Stroke por Enfermedad Cardíaca
    plt.figure(figsize=(8, 5))
    sns.countplot(data=data[data['stroke_risk'] == 'High'], x='heart_disease', palette='Blues')
    plt.title("Cantidad de Pacientes con Alto Riesgo de Ictus y Enfermedad Cardíaca", fontweight='bold')
    plt.xlabel("Enfermedad Cardíaca (0 = No, 1 = Sí)")
    plt.ylabel("Cantidad de Casos de Ictus")
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plots['heart_disease_stroke_plot'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

# Función para la visualización de los gráficos una vez subida la carga masiva
def success_view(request):
    # Carga los datos desde el modelo
    predictions = StrokePrediction.objects.all()
    
    # Convertimos los datos a un DataFrame
    data = pd.DataFrame(list(predictions.values()))
    
    # Filtrar solo las categorías que deseas
    data = data[data['stroke_risk'].isin(['High', 'Low'])]

    # Calcular total de registros y casos de ictus
    total_records = len(data)
    stroke_cases = data[data['stroke_risk'] == 'High'].shape[0]
    
    # Configura los gráficos
    plots = {}

    # Crear los gráficos en un hilo separado
    plot_thread = threading.Thread(target=create_plots, args=(data, plots, total_records, stroke_cases))
    plot_thread.start()
    plot_thread.join()

    return render(request, 'stroke_app/success.html', {'plots': plots})

###### CNN Model ######
# Cargar el modelo preentrenado de imágenes
def load_image_model():
    model_path = settings.IMAGE_MODEL_PATH  # Configura la ruta en settings.py
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading image model: {e}")
        return None

image_model = load_image_model()

# Nueva vista para predecir el ictus en una imagen
def predict_image_stroke(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        
        # Procesar la imagen para que sea compatible con el modelo
        try:
            image = Image.open(image_file)
            image = image.convert('RGB')  # Asegurarse de que la imagen sea RGB
            image = image.resize((224, 224))  # Tamaño de entrada del modelo
            image = np.array(image) / 255.0  # Normalización de la imagen
            image = np.expand_dims(image, axis=0)  # Añadir dimensión batch
            
            # Realizar predicción con el modelo
            prediction = image_model.predict(image)[0][0]
            
            # Aquí puedes ajustar el umbral o añadir una lógica más elaborada
            if prediction >= 0.5:
                stroke_prediction = 'Stroke Detected'
            elif prediction >= 0.3:  # Un umbral menor para "No seguro"
                stroke_prediction = 'Unsure, not a clear stroke indication'
            else:
                stroke_prediction = 'No Stroke Detected'
            
            # Añadir resultados al contexto
            context = {
                'stroke_prediction': stroke_prediction,
                'prediction_confidence': f"{prediction:.2%}"
            }
            return render(request, 'stroke_app/upload_image.html', context)

        except Exception as e:
            print(f"Error processing image: {e}")
            return render(request, 'stroke_app/upload_image.html', {'error': 'Error processing the image.'})
    
    return render(request, 'stroke_app/upload_image.html')

#### Reporte de métricas ####
# def train_stroke_model: Function that handles all the model training and evaluation logic
def train_stroke_model():
    """
    Trains the stroke prediction model and returns the model metrics and visualizations
    """
    # Load the dataset
    df = pd.read_csv(settings.DATASET_PATH)
    
    # Define columns
    cat_cols = ['gender', 'ever_married', 'Residence_type', 'work_type', 'smoking_status']
    num_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
    
    # Apply LabelEncoder to categorical columns
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols)
        ],
        remainder='passthrough'
    )
    
    # Prepare features and target
    X = df[cat_cols + num_cols]
    y = df['stroke']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train pipeline
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(
            solver='liblinear', 
            penalty='l1', 
            C=0.012557614443376395, 
            random_state=42
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Generate confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save confusion matrix plot to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    cm_image = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    ## Determine feature importance ##
    # Extraer el clasificador del pipeline
    classifier = pipeline.named_steps['classifier']
    X_train_transformed = pipeline.named_steps['preprocessor'].transform(X_train)
    
    # Obtener los nombres de las características después del preprocesamiento
    feature_names = num_cols + cat_cols

    feature_importance = pd.DataFrame(columns=['feature', 'importance'])  # Inicialización
    feature_image = ''  # Imagen de importancia de características inicializada
    
    # Comprobar si el clasificador tiene atributo de importancia de características
    if hasattr(classifier, 'feature_importances_'):
        # For classifiers like RandomForest, XGBoost, LightGBM
        importances = classifier.feature_importances_
    elif hasattr(classifier, 'coef_'):
        # For linear classifiers like Logistic Regression
        importances = np.abs(classifier.coef_[0])
    else:
        print("Feature importance not available for this classifier.")
        importances = None

    # Generate feature importance plot if available
    if importances is not None:
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance_df[:10])
        plt.title('Top 10 Feature Importance')
        plt.xlabel('Importance')
        
        # Save feature importance plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        feature_image = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
    else:
        feature_importance_df = pd.DataFrame(columns=['feature', 'importance'])
        feature_image = ''  # Imagen vacía si no hay importancia de características
        
    # Save the model
    model_path = os.path.join(settings.BASE_DIR, 'models', 'stroke_prediction_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    
    return {
        'metrics': metrics,
        'confusion_matrix': cm_image,
        'feature_importance': feature_image,
        'top_features': feature_importance_df.to_dict('records')  # Asegurar que contiene los datos correctos
    }

# The actual view function that calls the training function and renders the template
def stroke_report(request):
    """
    View function to display the stroke prediction model report
    """
    results = train_stroke_model()
    
    context = {
        'metrics': results['metrics'],
        'confusion_matrix': results['confusion_matrix'],
        'feature_importance': results['feature_importance'],
        'top_features': results['top_features']
    }
    
    return render(request, 'stroke_app/reports.html', context)