import matplotlib
matplotlib.use('Agg')
import seaborn as sb
from io import BytesIO
import base64
import matplotlib.pyplot as plt

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
    g = sb.scatterplot(x = time_matrix.flatten(), y = mesh.flatten(), hue = bools.flatten(), linewidth=0, palette = {'exercise':'red','wait':'lightblue','option out':'black'}, ax = axs)
    g.set_title('Moments of early exercise based on underlying value in time')
    g.set_xlabel('Time (years)')
    g.set_xlim([0,T*1.1])
    g.set_ylabel('Underlying price')
    g.set_ylim([mesh.min()*0.9, mesh.max()*1.1])
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
    inds = np.arange(c//2, b, c)
    binvalues = hsims[inds,:]
    optimals = np.zeros(np.shape(grid))
    optimals[(hs>=grid)*(hs>0)] = 1
    optimals[hs < 0] = -1
    optimals = np.array(pd.DataFrame(optimals).apply(lambda row: row.map({1:'exercise',0:'wait',-1:'option out'}),axis = 1))
    time_matrix = np.tile(np.arange(1, values_per_life + 1) / values_per_life * T , (nbins,1))
    fig, axs = matplotlib.pyplot.subplots()
    sb.set_style("ticks",{'axes.grid' : True})
    g = sb.scatterplot(x = time_matrix.flatten(), y = binvalues.flatten(),hue = grid.flatten(), linewidth=0, ax = axs, palette = sb.color_palette("rocket", as_cmap=True))
    g.set_title('Option price based on underlying value in time')
    g.set_xlabel('Time (years)')
    g.set_xlim([0,T*1.1])
    g.set_ylabel('Underlying price')
    #g.set_ylim([hsims.min()*0.9, hsims.max()*1.1])
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
    g = sb.scatterplot(x = time_matrix.flatten(), y = binvalues.flatten(), hue = optimals.flatten(), linewidth=0, palette = {'exercise':'red','wait':'lightblue','option out':'black'}, ax = axs)
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


## FD

def FD_graphs(S0, bools0, mesh0, Q0, T, k=150):
    mesh0 = mesh0.reshape((-1,mesh0.shape[1]))
    b, values_per_life = mesh0.shape
    time0 = np.arange(0, values_per_life) / (values_per_life-1) * T

    div_wh = np.arange(values_per_life-1)
    div_wh = div_wh[mesh0[1,:-1]!=mesh0[1,1:]]
    id_time = np.linspace(0,values_per_life-1,k,dtype=int)
    mesh = mesh0[:,id_time]
    bools = bools0[:,id_time]
    Q = Q0[:,id_time]
    time = np.hstack((time0[id_time],time0[:-1][div_wh]))
    time_matrix = np.tile(time,(b,1))
    mesh = np.hstack((mesh,mesh0[:,:-1][:,div_wh]))
    bools = np.hstack((bools,bools0[:,:-1][:,div_wh]))
    Q = np.hstack((Q,Q0[:,:-1][:,div_wh]))

    size = bools*50+20
    size[size<0] = 20
    bools = np.array(pd.DataFrame(bools).apply(lambda row: row.map({1:'exercise',0:'wait',-1:'option out'}),axis = 1))
    inba = (mesh>0.7*S0) & (mesh<1.3*S0)
    time_matrix = time_matrix[inba]
    mesh = mesh[inba]
    Q = Q[inba]
    bools = bools[inba] 
    size = size[inba]
    fig, axs = plt.subplots()
    sb.set_style("ticks",{'axes.grid' : True})
    g = sb.scatterplot(x = time_matrix, y = mesh, hue = Q, linewidth=0, ax = axs, palette = sb.color_palette("rocket", as_cmap=True))
    g.set_title('Option price based on underlying value in time')
    g.set_xlabel('Time (years)')
    g.set_xlim([0,T])
    g.set_ylabel('Underlying price')
    g.set_ylim([mesh.min(), mesh.max()])
    axs.legend([],[], frameon=False)
    norm = plt.Normalize(Q.min(), Q.max())
    cmap = sb.color_palette("rocket", as_cmap=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([axs.get_position().x1+0.05, axs.get_position().y0, 0.06, axs.get_position().height])
    axs.figure.colorbar(sm, cax=cax)

    imgdata = BytesIO()
    fig.savefig(imgdata, format='png')
    img_b64_1 = base64.b64encode(imgdata.getvalue()).decode()   
    fig, axs = matplotlib.pyplot.subplots()

    fig, axs = plt.subplots()
    sb.set_style("ticks",{'axes.grid' : True})
    g = sb.scatterplot(s=size,x = time_matrix, y = mesh, hue = bools, linewidth=0, palette = {'exercise':'red','wait':'lightblue','option out':'black'}, ax = axs)
    g.set_title('Moments of early exercise based on underlying value in time')
    g.set_xlabel('Time (years)')
    g.set_xlim([0,T])
    g.set_ylim([mesh.min(), mesh.max()])
    g.set_ylabel('Underlying price')

    imgdata = BytesIO()
    fig.savefig(imgdata, format='png')
    img_b64_2 = base64.b64encode(imgdata.getvalue()).decode()
    matplotlib.pyplot.close()
    return [img_b64_1, img_b64_2]
