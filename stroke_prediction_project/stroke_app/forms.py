from django import forms

class RangeInput(forms.widgets.Input):
    input_type = 'range'

class StrokePredictionForm(forms.Form):
    age = forms.IntegerField(
        min_value=0, 
        max_value=100, 
        widget=RangeInput(attrs={'class': 'form-range', 'value': 50, 'oninput': 'this.nextElementSibling.value = this.value'}),
        label="Age"
    )
    avg_glucose_level = forms.FloatField(
        min_value=50, 
        max_value=300, 
        widget=RangeInput(attrs={'class': 'form-range', 'value': 150, 'oninput': 'this.nextElementSibling.value = this.value', 'min': 50, 'max': 300}),
        label="Avg Glucose Level"
    )
    bmi = forms.FloatField(
        min_value=10, 
        max_value=60, 
        widget=RangeInput(attrs={'class': 'form-range', 'value': 25, 'oninput': 'this.nextElementSibling.value = this.value', 'min': 10, 'max': 60}),
        label="BMI"
    )
    hypertension = forms.ChoiceField(
        choices=[('Yes', 'Yes'), ('No', 'No')], 
        widget=forms.Select(attrs={'class': 'form-select'}),
        label="Hypertension"
    )
    heart_disease = forms.ChoiceField(
        choices=[('Yes', 'Yes'), ('No', 'No')], 
        widget=forms.Select(attrs={'class': 'form-select'}),
        label="Heart Disease"
    )
    gender = forms.ChoiceField(
        choices=[('Male', 'Male'), ('Female', 'Female')], 
        widget=forms.Select(attrs={'class': 'form-select'}),
        label="Gender"
    )
    ever_married = forms.ChoiceField(
        choices=[('Yes', 'Yes'), ('No', 'No')], 
        widget=forms.Select(attrs={'class': 'form-select'}),
        label="Ever Married"
    )
    work_type = forms.ChoiceField(
        choices=[
            ('Private', 'Private'), 
            ('Self-employed', 'Self-employed'), 
            ('Govt_job', 'Government Job'),
            ('children', 'Children'),
            ('Never_worked', 'Never Worked')
        ], 
        widget=forms.Select(attrs={'class': 'form-select'}),
        label="Work Type"
    )
    Residence_type = forms.ChoiceField(
        choices=[('Urban', 'Urban'), ('Rural', 'Rural')], 
        widget=forms.Select(attrs={'class': 'form-select'}),
        label="Residence Type"
    )
    smoking_status = forms.ChoiceField(
        choices=[
            ('formerly smoked', 'Formerly Smoked'), 
            ('never smoked', 'Never Smoked'), 
            ('smokes', 'Smokes'), 
            ('Unknown', 'Unknown')
        ], 
        widget=forms.Select(attrs={'class': 'form-select'}),
        label="Smoking Status"
    )