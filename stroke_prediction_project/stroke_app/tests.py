from django.test import TestCase
from django.urls import reverse
from stroke_app.forms import StrokePredictionForm
from stroke_app.models import StrokePrediction

# Tests para comprobar que el formulario es válido y que los datos introducidos son correctos.
class StrokePredictionFormTest(TestCase):
    def test_form_is_valid(self):
        form_data = {
            'age': 45,
            'avg_glucose_level': 150.5,
            'bmi': 25.3,
            'hypertension': True,  # Cambiado a booleano
            'heart_disease': False,  # Cambiado a booleano
            'gender': 'Male',
            'ever_married': True,  # Cambiado a booleano
            'work_type': 'Private',
            'Residence_type': 'Urban',
            'smoking_status': 'never smoked'
        }
        form = StrokePredictionForm(data=form_data)
        self.assertTrue(form.is_valid(), msg=form.errors)  # Añadir mensaje para errores

    def test_form_invalid_age(self):
        form_data = {
            'age': -5,  # Edad inválida
            'avg_glucose_level': 150.5,
            'bmi': 25.3,
            'hypertension': True,
            'heart_disease': False,
            'gender': 'Male',
            'ever_married': True,
            'work_type': 'Private',
            'residence_type': 'Urban',
            'smoking_status': 'never smoked'
        }
        form = StrokePredictionForm(data=form_data)
        self.assertFalse(form.is_valid(), msg=form.errors)  # Añadir mensaje para errores


# Test para la vista de predicción con datos válidos
class StrokePredictionViewTest(TestCase):
    def test_predict_stroke_post_valid(self):
        form_data = {
            'age': 45,
            'avg_glucose_level': 150.5,
            'bmi': 25.3,
            'hypertension': True,
            'heart_disease': False,
            'gender': 'Male',
            'ever_married': True,
            'work_type': 'Private',
            'Residence_type': 'Urban',
            'smoking_status': 'never smoked'
        }
        response = self.client.post(reverse('predict_stroke'), data=form_data)

        # Comprobar que se redirige correctamente y devuelve el código 200
        self.assertEqual(response.status_code, 200)

        # Comprobar que la predicción se ha guardado en la base de datos
        self.assertEqual(StrokePrediction.objects.count(), 1)

        # Comprobar que el resultado de riesgo de apoplejía se muestra
        self.assertIn('risk', response.context)
        self.assertIn(response.context['risk'], ['yes', 'no'])


# Tests para comprobar que la base de datos guarda correctamente
class StrokePredictionModelTest(TestCase):
    def test_stroke_prediction_creation(self):
        prediction = StrokePrediction.objects.create(
            age=45,
            avg_glucose_level=150.5,
            bmi=25.3,
            hypertension=True,
            heart_disease=False,
            gender='Male',
            ever_married=True,
            work_type='Private',
            Residence_type='Urban',
            smoking_status='never smoked',
            stroke_risk='no'
        )

        self.assertEqual(prediction.age, 45)
        self.assertEqual(prediction.avg_glucose_level, 150.5)
        self.assertEqual(prediction.bmi, 25.3)
        self.assertEqual(prediction.hypertension, True)
        self.assertEqual(prediction.heart_disease, False)
        self.assertEqual(prediction.gender, 'Male')
        self.assertEqual(prediction.ever_married, True)
        self.assertEqual(prediction.work_type, 'Private')
        self.assertEqual(prediction.Residence_type, 'Urban')
        self.assertEqual(prediction.smoking_status, 'never smoked')
        self.assertEqual(prediction.stroke_risk, 'no')


# Tests para la vista GET del formulario
class StrokePredictionViewGETTest(TestCase):
    def test_predict_stroke_get(self):
        response = self.client.get(reverse('predict_stroke'))

        # Comprobar que se devuelve el código 200 (OK)
        self.assertEqual(response.status_code, 200)

        # Comprobar que el formulario está presente en el contexto
        self.assertIn('form', response.context)

        # Comprobar que el template correcto se utiliza
        self.assertTemplateUsed(response, 'stroke_app/prediction_form.html')


# Test para predicción con datos inválidos
class StrokePredictionInvalidTest(TestCase):
    def test_predict_stroke_invalid_data(self):
        form_data = {
            'age': '',  # Falta la edad, lo cual es inválido
            'avg_glucose_level': 150.5,
            'bmi': 25.3,
            'hypertension': True,
            'heart_disease': False,
            'gender': 'Male',
            'ever_married': True,
            'work_type': 'Private',
            'Residence_type': 'Urban',
            'smoking_status': 'never smoked'
        }
        response = self.client.post(reverse('predict_stroke'), data=form_data)

        # Comprobar que la respuesta es un 200 (sin redireccionamiento, porque los datos son inválidos)
        self.assertEqual(response.status_code, 200)

        # Comprobar que el formulario no es válido
        self.assertFalse(response.context['form'].is_valid())

        # Comprobar que no se ha creado ningún registro en la base de datos
        self.assertEqual(StrokePrediction.objects.count(), 0)


# Test para la predicción (mock del modelo)
from unittest.mock import patch

class StrokePredictionMockTest(TestCase):
    @patch('stroke_app.models.StrokePrediction.predict')  # Cambiado para apuntar al método correcto
    def test_predict_stroke_mock(self, mock_predict):
        # Simulamos que el modelo devuelve un resultado de 'no'
        mock_predict.return_value = [0]  # Cambiar el valor si tu modelo devuelve algo diferente

        form_data = {
            'age': 45,
            'avg_glucose_level': 150.5,
            'bmi': 25.3,
            'hypertension': True,
            'heart_disease': False,
            'gender': 'Male',
            'ever_married': True,
            'work_type': 'Private',
            'Residence_type': 'Urban',
            'smoking_status': 'never smoked'
        }
        response = self.client.post(reverse('predict_stroke'), data=form_data)

        # Comprobamos que el mock ha sido llamado
        mock_predict.assert_called_once()

        # Comprobar que la predicción es 'no'
        self.assertIn('risk', response.context)
        self.assertEqual(response.context['risk'], 'no')
