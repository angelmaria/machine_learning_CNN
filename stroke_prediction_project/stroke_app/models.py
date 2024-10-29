from django.db import models
class StrokePrediction(models.Model):
    GENDER_CHOICES = [
        ('Male', 'Male'),
        ('Female', 'Female'),
    ]

    SMOKING_STATUS_CHOICES = [
        ('never smoked', 'Never Smoked'),
        ('formerly smoked', 'Formerly Smoked'),
        ('smokes', 'Smokes'),
    ]

    WORK_TYPE_CHOICES = [
        ('Private', 'Private'),
        ('Self-employed', 'Self-employed'),
        ('Govt_job', 'Govt Job'),
        ('children', 'Children'),
        ('Never_worked', 'Never worked'),
    ]

    RESIDENCE_TYPE_CHOICES = [
        ('Urban', 'Urban'),
        ('Rural', 'Rural'),
    ]

    age = models.IntegerField()
    avg_glucose_level = models.FloatField()
    bmi = models.FloatField()
    hypertension = models.BooleanField()
    heart_disease = models.BooleanField()
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES)
    ever_married = models.BooleanField()
    work_type = models.CharField(max_length=50, choices=WORK_TYPE_CHOICES)
    Residence_type = models.CharField(max_length=50, choices=RESIDENCE_TYPE_CHOICES)
    smoking_status = models.CharField(max_length=20, choices=SMOKING_STATUS_CHOICES)
    stroke_risk = models.CharField(max_length=255, blank=True, null=True) 
    date_submitted = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction {self.id} - Age: {self.age}, Glucose: {self.avg_glucose_level}"

