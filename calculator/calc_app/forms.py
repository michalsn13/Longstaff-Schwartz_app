from django import forms
from django.core.validators import MaxValueValidator, MinValueValidator
import numpy as np
class OptionForm(forms.Form):
    barriers = (('vanilla','Vanilla'),('up_in','Up-&-In'),('up_out','Up-&-Out'),('down_in','Down-&-In'),('down_out','Down-&-Out'))
    dividends = (('1','Absolute annualy'), ('4','Absolute quarterly'), ('12','Absolute monthly'), ('infty','Continous yield'))
    barrier_type = forms.CharField(label='Barrier type', widget=forms.Select(choices=barriers))
    barrier = forms.FloatField(required = False, label = 'Barrier', validators=[MinValueValidator(0.0)])
    S0 = forms.FloatField(label = 'Price', validators=[MinValueValidator(0.0)])
    K = forms.FloatField(label = 'Strike', validators=[MinValueValidator(0.0)])
    T = forms.FloatField(label = 'Years to expire', validators=[MinValueValidator(0.0)])
    sigma = forms.FloatField(label = 'Volatility (%)', validators=[MinValueValidator(0.0)])
    r = forms.FloatField(label = 'Interest rate (%)', validators=[MinValueValidator(0.0)])
    div_freq = forms.CharField(label='Dividend type', widget=forms.Select(choices=dividends))
    div = forms.FloatField(required = False, label = 'Dividend value (% if continous)', validators=[MinValueValidator(0.0)])
    next_div_moment = forms.FloatField(required = False, label = 'Time till next absolute dividend (years)', validators=[MinValueValidator(0.0)])
    values_per_year = forms.IntegerField(required = False, label = 'Exercises per year', validators=[MinValueValidator(5.0),])
    rounding = forms.IntegerField(label = "Rounding", validators = [MinValueValidator(0),])
    