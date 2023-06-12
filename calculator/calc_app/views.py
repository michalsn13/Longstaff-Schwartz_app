from django.shortcuts import render
from django.views import View
from .forms import OptionForm
from .visual_funcs import SM_graphs
import sys
sys.path.append('..')
from underlying import GBM
from option import Option
from stochastic_mesh_func import *
from Longstaff_Schwartz import LS
from finite_difference import FD
from State_Space import *

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

                barrier = form_dict['barrier']
                def barrier_func_ls(X,t):
                    X = np.hstack((S0*np.ones((X.shape[0],1)),X))
                    tn = np.hstack((np.zeros((X.shape[0],1)),t))
                    sign1 = (-1)**(direction == 'up')
                    sign2 = (-1)**(outcome == 'in')
                    t_out_of_bar = tn[0,np.argmax(sign1*X < sign1*barrier,axis=1)]
                    t_out_of_bar[t_out_of_bar==0] = 1e9
                    return sign2*t < sign2*t_out_of_bar.reshape((-1,1))
            else:
                barrier_func_ls = lambda X, t : True
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
            
            V_sm_call, bools_call, mesh_call, Q_call = stochastic_mesh(Option(underlying, payoff_func_call, T, barrier_func, barrier_out), 1000)
            V_sm_put, bools_put, mesh_put, Q_put = stochastic_mesh(Option(underlying, payoff_func_put, T, barrier_func, barrier_out), 1000)
            LS_call, _, _, _ = LS(Option(underlying, payoff_func_call, T, barrier_func_ls, barrier_out),int(2e4))
            LS_put, _, _, _ = LS(Option(underlying, payoff_func_put, T, barrier_func_ls, barrier_out),int(2e4))
            FD_call, _, _, _ = FD(Option(underlying, payoff_func_call, T, barrier_func, barrier_out),400)
            FD_put, _, _, _ = FD(Option(underlying, payoff_func_put, T, barrier_func, barrier_out),400)
            Probs, Sims, Hsims = prob(Option(underlying, payoff_func_put, T, barrier_func, barrier_out), nbin = 500,b=5*10**4)#if you write it like 1e5 it breaks because of float and I dont have the patience to fix it again
            SS_call = SS(Option(underlying, payoff_func_call, T, barrier_func, barrier_out),Sims,  Probs, Hsims)
            SS_put = SS(Option(underlying, payoff_func_put, T, barrier_func, barrier_out),Sims, Probs, Hsims)
            context = {'output':{
                                'stochastic_mesh':{'name':'Stochastic Mesh', 'href':'sm', 
                                                    'call':{'price':round(V_sm_call, rounding)},#, 'plots': SM_graphs(V_sm_call, bools_call, mesh_call, Q_call, T)}, 
                                                    'put':{'price':round(V_sm_put, rounding)}#, 'plots': SM_graphs(V_sm_put, bools_put, mesh_put, Q_put, T)}
                                                  },
                                'longstaff-schwartz':{'name':'Longstaff-Schwartz', 'href':'ls', 
                                                      'call':{'price':round(LS_call, rounding)}, 
                                                      'put':{'price':round(LS_put, rounding)}
                                                     },
                                'state-space partitioning':{'name':'State-Space Partitioning', 'href':'ss', 
                                                      'call':{'price':round(SS_call, rounding)}, 
                                                      'put':{'price':round(SS_put, rounding)}
                                                     },
                                'finite-difference':{'name':'Finite Difference', 'href':'fd',
                                                     'call':{'price':round(FD_call,rounding)},
                                                     'put':{'price':round(FD_put,rounding)}
                                                    }
                                },
                        'form':form
                      }
            return render(request, 'calc_app/index.html', context)
        return render(request, 'calc_app/index.html', {'form':form})