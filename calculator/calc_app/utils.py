import matplotlib
matplotlib.use('Agg')
import seaborn as sb
from io import BytesIO
import base64

import numpy as np
import pandas as pd

def SM_graphs(V, bools, mesh, Q, T):
    bools = np.array(pd.DataFrame(bools).apply(lambda row: row.map({1:'exercise',0:'wait',-1:'option out'}),axis = 1))
    b, values_per_life = mesh.shape
    time_matrix = np.tile(np.arange(1, values_per_life + 1) / values_per_life * T , (b,1))
    fig, axs = matplotlib.pyplot.subplots()
    sb.set_style("ticks",{'axes.grid' : True})
    g = sb.scatterplot(x = time_matrix.flatten(), y = mesh.flatten(), hue = Q.flatten(), linewidth=0, ax = axs, palette = sb.color_palette("rocket", as_cmap=True))
    g.set_title('Option price based on underlying value in time')
    g.set_xlabel('Time (years)')
    g.set_xlim([0,T*1.1])
    g.set_ylabel('Underlying price')
    g.set_ylim([mesh.min()*0.9, mesh.max()*1.1])
    axs.legend([],[], frameon=False)
    norm = matplotlib.pyplot.Normalize(np.floor(Q.min()), np.ceil(Q.max()))
    cmap = sb.color_palette("rocket", as_cmap=True)
    sm = matplotlib.pyplot.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([axs.get_position().x1+0.001, axs.get_position().y0, 0.03, axs.get_position().height])
    cbar = axs.figure.colorbar(sm, cax=cax)
    imgdata = BytesIO()
    fig.savefig(imgdata, format='png')
    img_b64_1 = base64.b64encode(imgdata.getvalue()).decode()   
    fig, axs = matplotlib.pyplot.subplots()
    sb.set_style("ticks",{'axes.grid' : True})
    g = sb.scatterplot(x = time_matrix.flatten(), y = mesh.flatten(), hue = bools.flatten(), linewidth=0, palette = {'exercise':'#84b701','wait':'#448ee4','option out':'#cf524e'}, ax = axs)
    g.set_title('Moments of early exercise based on underlying value in time')
    g.set_xlabel('Time (years)')
    g.set_xlim([0,T*1.1])
    g.set_ylabel('Underlying price')
    imgdata = BytesIO()
    fig.savefig(imgdata, format='png')
    img_b64_2 = base64.b64encode(imgdata.getvalue()).decode()
    matplotlib.pyplot.close()
    return [img_b64_1, img_b64_2]

def conv_graphs(Vs_call, ref_call, Vs_put, ref_put):
    exercises = Vs_call.index
    
    fig, axs = matplotlib.pyplot.subplots()
    g = Vs_call.plot(style = '-o', ax = axs)
    g.legend(loc = 'best')
    g.set_title(f'Price convergence depending on number of exercises per year\nFinite Diff ref price: {ref_call:.4f}')
    g.set_xlabel('log2(Number of exercises per year / 12)')
    g.set_ylabel('Price')
    g.hlines(xmin = min(exercises), xmax = max(exercises), y = ref_call, linestyles = 'dashed', color = 'black')
    imgdata = BytesIO()
    fig.savefig(imgdata, format='png')
    img_b64_1 = base64.b64encode(imgdata.getvalue()).decode()
    
    fig, axs = matplotlib.pyplot.subplots()
    g = Vs_put.plot(style = '-o', ax = axs)
    g.legend(loc = 'best')
    g.set_title(f'Price convergence depending on number of exercises per year\nFinite Diff ref price: {ref_put:.4f}')
    g.set_xlabel('log2(Number of exercises per year / 12)')
    g.set_ylabel('Price')
    g.hlines(xmin = min(exercises), xmax = max(exercises), y = ref_put, linestyles = 'dashed', color = 'black')
    imgdata = BytesIO()
    fig.savefig(imgdata, format='png')
    img_b64_2 = base64.b64encode(imgdata.getvalue()).decode()
    matplotlib.pyplot.close()
    return [img_b64_1, img_b64_2]

#SS

def SS_graphs(grid, hs, hsims, T):
    nbins = np.shape(grid)[0]
    b = np.shape(hsims)[0]
    values_per_life = np.shape(grid)[1]
    c = b//nbins
    inds = np.arange(0, b, c)
    binvalues = hsims[inds,:]
    optimals = np.zeros(np.shape(grid))
    optimals[grid>hs] = 1    
    time_matrix = np.tile(np.arange(1, values_per_life + 1) / values_per_life * T , (nbins,1))
    fig, axs = matplotlib.pyplot.subplots()
    sb.set_style("ticks",{'axes.grid' : True})
    g = sb.scatterplot(x = time_matrix.flatten(), y = binvalues.flatten(),hue = grid.flatten(), linewidth=0, ax = axs, palette = sb.color_palette("rocket", as_cmap=True))
    g.set_title('Option price based on underlying value in time')
    g.set_xlabel('Time (years)')
    g.set_xlim([0,T*1.1])
    g.set_ylabel('Underlying price')
    g.set_ylim([hsims.min()*0.9, hsims.max()*1.1])
    axs.legend([],[], frameon=False)
    norm = matplotlib.pyplot.Normalize(np.floor(grid.min()), np.ceil(grid.max()))
    cmap = sb.color_palette("rocket", as_cmap=True)
    ss = matplotlib.pyplot.cm.ScalarMappable(cmap=cmap, norm=norm)
    ss.set_array([])
    cax = fig.add_axes([axs.get_position().x1+0.001, axs.get_position().y0, 0.03, axs.get_position().height])
    cbar = axs.figure.colorbar(ss, cax=cax)
    imgdata = BytesIO()
    fig.savefig(imgdata, format='png')
    img_b64_3 = base64.b64encode(imgdata.getvalue()).decode()   
    fig, axs = matplotlib.pyplot.subplots()
    sb.set_style("ticks",{'axes.grid' : True})
    g = sb.scatterplot(x = time_matrix.flatten(), y = binvalues.flatten(), hue = optimals.flatten(), linewidth=0, palette = {1:'#84b701',0:'#448ee4'}, ax = axs)
    g.set_title('Moments of early exercise based on underlying value in time')
    g.set_xlabel('Time (years)')
    g.set_xlim([0,T*1.1])
    g.set_ylabel('Underlying price')
    imgdata = BytesIO()
    fig.savefig(imgdata, format='png')
    img_b64_4 = base64.b64encode(imgdata.getvalue()).decode()
    matplotlib.pyplot.close()
    #data = imgdata.getvalue()
    return [img_b64_3, img_b64_4]