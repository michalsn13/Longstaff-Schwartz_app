from django.shortcuts import render
from django.views import View
from .forms import OptionForm

import sys
sys.path.append('..')
from underlying import GBM
from option import Option
from stochastic_mesh_func import *
from Longstaff_Schwartz import LS

class Index(View):
    def get(self, request):
        initial = {
            'barrier_type': request.session.get('barrier_type'),
            'barrier': request.session.get('barrier'),
            'S0': request.session.get('S0'),
            'K': request.session.get('K'),
            'T': request.session.get('T'),
            'sigma': request.session.get('sigma'),
            'r': request.session.get('r'),
            'div_freq': request.session.get('div_freq'),
            'div': request.session.get('div') if request.session.get('div') else 0,
            'next_div_moment': request.session.get('next_div_moment') if request.session.get('next_div_moment') else 0,
            'values_per_year': request.session.get('values_per_year') if request.session.get('values_per_year') else 120,
            'rounding': request.session.get('rounding') if request.session.get('rounding') else 4
        }
        form = OptionForm(initial = initial)
        return render(request, 'calc_app/index.html', {'form': form})
    def post(self, request):
        form = OptionForm(request.POST)
        if form.is_valid():
            form_dict = form.cleaned_data
            S0 = form_dict['S0']
            K = form_dict['K']
            T = form_dict['T']
            sigma = form_dict['sigma']/100
            r = form_dict['r']/100
            div_freq = np.infty if form_dict['div_freq'] == 'infty' else float(form_dict['div_freq'])
            div = form_dict['div']/100 if div_freq == np.infty else form_dict['div']
            next_div_moment = form_dict['next_div_moment']
            values_per_year = form_dict['values_per_year']
            if form_dict['barrier_type']!='vanilla':
                direction, outcome = form_dict['barrier_type'].split('_')
                sign = (-1)**((direction == 'up')==(outcome == 'in'))
                barrier_func = lambda X, t : sign*X < sign*form_dict['barrier']
                barrier_out = outcome == 'out'
                #def barrier_func_ls(X,t):
                #    X_temp = np.hstack((X,np.ones(X.shape[0],1)))
                #    barrier = form_dict['barrier']
                #    if outcome == 'out':
                #        return t < t[np.ix_(np.arange(X.shape[0]),np.argmax(X_temp>=barrier,axis=1))] 
                #    else:
                #        return t >= t[np.ix_(np.arange(X.shape[0]),np.argmax(X_temp>=barrier,axis=1))] 
                #barrier_func_ls = lambda X, t: t < t[:,np.argmax(X>=barrier)]
            else:
                barrier_func = lambda X, t : True
                barrier_out = False
            rounding = int(form_dict['rounding'])
            mu = 0
            if values_per_year:
                underlying = GBM(S0, mu, sigma, r, div = div, div_freq = div_freq, next_div_moment = next_div_moment, values_per_year = values_per_year)
            else:
                underlying = GBM(S0, mu, sigma, r, div = div, div_freq = div_freq, next_div_moment = next_div_moment)
            payoff_func_call = lambda X, t: np.maximum(X-K, 0)
            payoff_func_put = lambda X, t: np.maximum(K-X, 0)
            call = Option(underlying, payoff_func_call, T, barrier_func, barrier_out)
            put = Option(underlying, payoff_func_put, T, barrier_func, barrier_out)
            V_sm_call, _, _, _ = stochastic_mesh(call, 1000)
            V_sm_put, _, _, _ = stochastic_mesh(put, 1000)
            LS_call, _, _, _ = LS(call,int(1e5))
            LS_put, _, _, _ = LS(put,int(1e5))
            context = {'output':{
                                'Stochastic Mesh':{'href':'sm', 'price_call':round(V_sm_call, rounding), 'price_put':round(V_sm_put, rounding)},
                                'Longstaff-Schwartz':{'href':'ls', 'price_call':round(LS_call, rounding), 'price_put':round(LS_put, rounding)}
                                },
                       'form':form
                      }
            return render(request, 'calc_app/index.html', context)
        return render(request, 'calc_app/index.html', {'form':form})