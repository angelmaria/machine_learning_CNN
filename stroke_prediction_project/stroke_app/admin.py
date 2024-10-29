# Register your models here.
# admin.py
from django.contrib import admin
from .models import StrokePrediction

@admin.register(StrokePrediction)
class StrokePredictionAdmin(admin.ModelAdmin):
    list_display = ['age', 'gender', 'stroke_risk', 'date_submitted']
    list_filter = ['stroke_risk', 'gender']