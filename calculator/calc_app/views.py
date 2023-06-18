import time
import os
import pandas as pd
from multiprocess import Pool
import sys
sys.path.append('..')

from django.shortcuts import render
from django.views import View
from .forms import OptionForm, DataOptionForm, ConvOptionForm
from .utils import *

from underlying import GBM, DataUnderlying
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
        context = {'form':form}
        if form.is_valid():
            context = {'output':{},
            'form':form
            }
            form_dict = form.cleaned_data
            methods = form_dict['method_types']
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
                if outcome == 'in':
                    methods = list(set(methods).intersection(set(['ls'])))
            else:
                barrier_func_ls = lambda X, t : True
                barrier_func = lambda X, t : True
                barrier_out = False
            rounding = int(form_dict['rounding'])
            if values_per_year:
                underlying = GBM(S0, sigma, r, div = div, div_freq = div_freq, next_div_moment = next_div_moment, values_per_year = values_per_year)
            else:
                underlying = GBM(S0, sigma, r, div = div, div_freq = div_freq, next_div_moment = next_div_moment)
            payoff_func_call = lambda X, t: np.maximum(X-K, 0)
            payoff_func_put = lambda X, t: np.maximum(K-X, 0)
            if 'sm' in methods:
                t0 = time.time()
                V_sm_call, bools_call, mesh_call, Q_call = stochastic_mesh(Option(underlying, payoff_func_call, T, barrier_func, barrier_out), 1000)
                V_sm_put, bools_put, mesh_put, Q_put = stochastic_mesh(Option(underlying, payoff_func_put, T, barrier_func, barrier_out), 1000)
                context['output']['stochastic_mesh'] = {'name':'Stochastic Mesh', 'href':'sm', 'time': f'{time.time() - t0:.4f}s',
                                        'call':{'price':round(V_sm_call, rounding)}, 
                                        'put':{'price':round(V_sm_put, rounding)}
                                      }
                #plots
                request.session['sm_call'] = SM_graphs(V_sm_call, bools_call, mesh_call, Q_call, T)
                request.session['sm_put'] = SM_graphs(V_sm_put, bools_put, mesh_put, Q_put, T)
                #endplots
            if 'ls' in methods:
                t0 = time.time()
                LS_call, _, _, _ = LS(Option(underlying, payoff_func_call, T, barrier_func_ls, barrier_out),int(2e4))
                LS_put, _, _, _ = LS(Option(underlying, payoff_func_put, T, barrier_func_ls, barrier_out),int(2e4))
                context['output']['longstaff-schwartz'] = {'name':'Longstaff-Schwartz', 'href':'ls', 'time':f'{time.time() - t0:.4f}s',
                                                      'call':{'price':round(LS_call, rounding)}, 
                                                      'put':{'price':round(LS_put, rounding)}
                                                     }
            if 'ss' in methods:
                t0 = time.time()
                Probs, Sims, Hsims = prob(Option(underlying, payoff_func_put, T, barrier_func, barrier_out), nbin = 500,b=10**5)#if you write it like 1e5 it breaks because of float and I dont have the patience to fix it again
                SS_call, Grid_call, Hs_call = SS(Option(underlying, payoff_func_call, T, barrier_func, barrier_out),Sims,  Probs, Hsims)
                SS_put, Grid_put, Hs_put = SS(Option(underlying, payoff_func_put, T, barrier_func, barrier_out),Sims, Probs, Hsims)
                context['output']['state-space-partitioning'] = {'name':'State-Space Partitioning', 'href':'ss', 'time': f'{time.time() - t0:.4f}s',
                                                      'call':{'price':round(SS_call, rounding)}, 
                                                      'put':{'price':round(SS_put, rounding)}
                                                     }
                #plots
                request.session['ss_call'] = SS_graphs(Grid_call, Hs_call, Hsims, T)
                request.session['ss_put'] = SS_graphs(Grid_put, Hs_put, Hsims, T)
                #endplots
            if 'fd' in methods:
                t0 = time.time()
                FD_call, _, _, _ = FD(Option(underlying, payoff_func_call, T, barrier_func, barrier_out),400)
                FD_put, _, _, _ = FD(Option(underlying, payoff_func_put, T, barrier_func, barrier_out),400)
                context['output']['finite-difference'] = {'name':'Finite Difference', 'href':'','time':f'{time.time() - t0:.4f}s',
                                                     'call':{'price':round(FD_call,rounding)},
                                                     'put':{'price':round(FD_put,rounding)}
                                                    }

        return render(request, 'calc_app/index.html', context)


class DataIndex(View):
    def get(self, request):
        initial = {
            'barrier_type': request.session.get('barrier_type'),
            'barrier': request.session.get('barrier'),
            'S0': request.session.get('S0'),
            'K': request.session.get('K'),
            'T': request.session.get('T'),
            'r': request.session.get('r'),
            'div_freq': request.session.get('div_freq'),
            'div': request.session.get('div') if request.session.get('div') else 0,
            'next_div_moment': request.session.get('next_div_moment') if request.session.get('next_div_moment') else 0,
            'rounding': request.session.get('rounding') if request.session.get('rounding') else 4
        }
        form = DataOptionForm(initial = initial)
        return render(request, 'calc_app/data.html', {'form': form})
    def post(self, request):
        form = DataOptionForm(request.POST, request.FILES)
        context = {'form':form}
        if form.is_valid():
            context = {'output':{},
            'form':form
            }
            form_dict = form.cleaned_data
            csv_file = form_dict['csv_file']
            with open(f"files/{csv_file.name}", "wb+") as destination:
                for chunk in csv_file.chunks():
                    destination.write(chunk)
            methods = form_dict['method_types']
            S0 = form_dict['S0']
            K = form_dict['K']
            T = form_dict['T']
            r = form_dict['r']/100
            div_freq = np.infty if form_dict['div_freq'] == 'infty' else float(form_dict['div_freq'])
            div = form_dict['div']/100 if div_freq == np.infty else form_dict['div']
            next_div_moment = form_dict['next_div_moment']
            
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
                if outcome == 'in':
                    methods = list(set(methods).intersection(set(['ls'])))
            else:
                barrier_func_ls = lambda X, t : True
                barrier_func = lambda X, t : True
                barrier_out = False
            rounding = int(form_dict['rounding'])
            underlying = DataUnderlying(f"files/{csv_file.name}", S0, r, div = div, div_freq = div_freq, next_div_moment = next_div_moment)
            payoff_func_call = lambda X, t: np.maximum(X-K, 0)
            payoff_func_put = lambda X, t: np.maximum(K-X, 0)
            if 'ls' in methods:
                t0 = time.time()
                LS_call, _, _, _ = LS(Option(underlying, payoff_func_call, T, barrier_func_ls, barrier_out),int(2e4))
                LS_put, _, _, _ = LS(Option(underlying, payoff_func_put, T, barrier_func_ls, barrier_out),int(2e4))
                context['output']['longstaff-schwartz'] = {'name':'Longstaff-Schwartz', 'href':'ls', 'time':f'{time.time() - t0:.4f}s',
                                                      'call':{'price':round(LS_call, rounding)}, 
                                                      'put':{'price':round(LS_put, rounding)}
                                                     }
            if 'ss' in methods:
                t0 = time.time()
                Probs, Sims, Hsims = prob(Option(underlying, payoff_func_put, T, barrier_func, barrier_out), nbin = 500,b=5*10**4)#if you write it like 1e5 it breaks because of float and I dont have the patience to fix it again
                SS_call, Grid_call, Hs_call = SS(Option(underlying, payoff_func_call, T, barrier_func, barrier_out),Sims,  Probs, Hsims)
                SS_put, Grid_put, Hs_put = SS(Option(underlying, payoff_func_put, T, barrier_func, barrier_out),Sims, Probs, Hsims)
                context['output']['state-space-partitioning'] = {'name':'State-Space Partitioning', 'href':'ss', 'time': f'{time.time() - t0:.4f}s',
                                                      'call':{'price':round(SS_call, rounding)}, 
                                                      'put':{'price':round(SS_put, rounding)}
                                                     }
                #plots
                request.session['ss_call'] = SS_graphs(Grid_call, Hs_call, Hsims, T)
                request.session['ss_put'] = SS_graphs(Grid_put, Hs_put, Hsims, T)
                #endplots
                
            check_bool, check_p = underlying.check
            context['check_message'] = f"No reasons to reject null hypothesis about your sims representing a martingale (p-value = {check_p:.4} > alpha)" if check_bool else f"Null hypothesis about your sims representing a martingale got rejected (p-value = {check_p:.4} <= alpha)"
            context['check_color'] = "#008000" if check_bool else "#FF0000"
            os.remove(f"files/{csv_file.name}")
        else:
            print(form.is_valid())
        return render(request, 'calc_app/data.html', context)

class ConvIndex(View):
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
            'rounding': request.session.get('rounding') if request.session.get('rounding') else 4,
            'processors': request.session.get('processors') if request.session.get('processors') else 5
        }
        form = ConvOptionForm(initial = initial)
        return render(request, 'calc_app/conv.html', {'form': form})
    def post(self, request):
        form = ConvOptionForm(request.POST)
        context = {'form':form}
        if form.is_valid():
            context = {'output':{'exer':{}},
            'form':form
            }
            form_dict = form.cleaned_data
            methods = form_dict['method_types']
            S0 = form_dict['S0']
            K = form_dict['K']
            T = form_dict['T']
            sigma = form_dict['sigma']/100
            r = form_dict['r']/100
            div_freq = np.infty if form_dict['div_freq'] == 'infty' else float(form_dict['div_freq'])
            div = form_dict['div']/100 if div_freq == np.infty else form_dict['div']
            next_div_moment = form_dict['next_div_moment']
            
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
                if outcome == 'in':
                    methods = list(set(methods).intersection(set(['ls'])))
            else:
                barrier_func_ls = lambda X, t : True
                barrier_func = lambda X, t : True
                barrier_out = False
            rounding = int(form_dict['rounding'])
            processors = int(form_dict['processors'])

            payoff_func_call = lambda X, t: np.maximum(X-K, 0)
            payoff_func_put = lambda X, t: np.maximum(K-X, 0)
            
            exercises = range(0, 6)

            def pool_f(e):
                results_call = pd.DataFrame(index = [e], columns = methods)
                results_put = pd.DataFrame(index = [e], columns = methods)                
                underlying = GBM(S0, sigma, r, div = div, div_freq = div_freq, next_div_moment = next_div_moment, values_per_year = 12 * 2 ** e)
                if 'sm' in methods:
                    price_call, _, _, _ = stochastic_mesh(Option(underlying, payoff_func_call, T, barrier_func, barrier_out), 1000)
                    price_put, _, _, _ = stochastic_mesh(Option(underlying, payoff_func_put, T, barrier_func, barrier_out), 1000)
                    results_call.loc[e, 'sm'] = price_call
                    results_put.loc[e, 'sm'] = price_put
                    
                if 'ls' in methods:
                    price_call, _, _, _ = LS(Option(underlying, payoff_func_call, T, barrier_func_ls, barrier_out),int(2e4))
                    price_put, _, _, _ = LS(Option(underlying, payoff_func_put, T, barrier_func_ls, barrier_out),int(2e4))
                    results_call.loc[e, 'ls'] = price_call
                    results_put.loc[e, 'ls'] = price_put
                if 'ss' in methods:
                    Probs, Sims, Hsims = prob(Option(underlying, payoff_func_put, T, barrier_func, barrier_out), nbin = 500,b=5*10**4)
                    price_call, Grid_call, Hs_call = SS(Option(underlying, payoff_func_call, T, barrier_func, barrier_out),Sims,  Probs, Hsims)
                    price_put, Grid_put, Hs_put = SS(Option(underlying, payoff_func_put, T, barrier_func, barrier_out),Sims, Probs, Hsims)
                    results_call.loc[e, 'ss'] = price_call
                    results_put.loc[e, 'ss'] = price_put
                return results_call, results_put
            with Pool(processors) as pool:
                pool_results = pool.map(pool_f, exercises)
                
            Vs_call = pd.concat([i[0] for i in pool_results])
            Vs_put = pd.concat([i[1] for i in pool_results])
            
            underlying = GBM(S0, sigma, r, div = div, div_freq = div_freq, next_div_moment = next_div_moment, values_per_year = 10)
            ref_call, _, _, _ = FD(Option(underlying, payoff_func_call, T, barrier_func, barrier_out),400)
            ref_put, _, _, _ = FD(Option(underlying, payoff_func_put, T, barrier_func, barrier_out),400)
            
            names = {'ls':'Longstaff-Schwartz', 'sm': 'Stochastic Mesh', 'ss': 'State Space Partitioning'}
            Vs_call.columns = [names[idx] for idx in Vs_call.columns]
            Vs_put.columns = [names[idx] for idx in Vs_put.columns]
            
            plots = conv_graphs(Vs_call, ref_call, Vs_put, ref_put)
            context['output']['exer']['call'] = plots[0]
            context['output']['exer']['put'] = plots[1]
        return render(request, 'calc_app/conv.html', context)
    
def sm(request):
    call = request.session.get('sm_call')
    put = request.session.get('sm_put')
    return render(request, "calc_app/sm.html", {"call": call, 'put': put})

def ls(request):
    call = request.session.get('ls_call')
    put = request.session.get('ls_put')
    return render(request, "calc_app/ls.html", {"call": call, 'put': put})

def ss(request):
    call = request.session.get('ss_call')
    put = request.session.get('ss_put')
    return render(request, "calc_app/ss.html", {"call": call, 'put': put})

def fd(request):
    call = request.session.get('fd_call')
    put = request.session.get('fd_put')
    return render(request, "calc_app/fd.html", {"call": call, 'put': put})