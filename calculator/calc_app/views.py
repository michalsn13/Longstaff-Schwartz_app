from django.shortcuts import render
from django.views import View
from .forms import OptionForm

import sys
sys.path.append('..')
from underlying import GBM
from option import Option
from stochastic_mesh_func import *

class Index(View):
    def get(self, request):
        form = OptionForm()
        return render(request, 'calc_app/index.html', {'form': form})
    def post(self, request):
        form = OptionForm(request.POST)
        if form.is_valid():
            form_dict = form.cleaned_data
            option_type = form_dict['option_type']
            S0 = form_dict['S0']
            K = form_dict['K']
            T = form_dict['T']
            sigma = form_dict['sigma']/100
            r = form_dict['r']/100
            values_per_year = form_dict['values_per_year']
            call_bool = (option_type == 'call')
            mu = 0
            if values_per_year:
                underlying = GBM(S0, mu, sigma, r, values_per_year)
            else:
                underlying = GBM(S0, mu, sigma, r)
            payoff_func = lambda X, t: np.maximum((-1)**call_bool *(K - X), 0)
            option = Option(underlying, payoff_func, T)
            V_sm, _, _, _ = stochastic_mesh(option, 1000)
            context = {'output':{
                                'Stochastic Mesh':{'price':round(V_sm, 4)}
                                },
                       'input':form_dict
                      }
            return render(request, 'calc_app/result.html', context)