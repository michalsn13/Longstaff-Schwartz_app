from django import forms
from django.core.validators import MaxValueValidator, MinValueValidator

class OptionForm(forms.Form):
    options = (('call','Vanilla Call'),('put','Vanilla Put'))
    option_type = forms.ChoiceField(label = 'Option type', widget = forms.RadioSelect, choices = options)
    S0 = forms.FloatField(label = 'Price', validators=[MinValueValidator(0.0)])
    K = forms.FloatField(label = 'Strike', validators=[MinValueValidator(0.0)])
    T = forms.FloatField(label = 'Years to expire', validators=[MinValueValidator(0.0)])
    sigma = forms.FloatField(label = 'Volatility (%)', validators=[MinValueValidator(0.0)])
    r = forms.FloatField(label = 'Interest rate (%)', validators=[MinValueValidator(0.0)])
    values_per_year = forms.IntegerField(required = False, label = 'Exercises per year', validators=[MinValueValidator(5.0),])
    