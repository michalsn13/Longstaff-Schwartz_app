import matplotlib.pyplot as plt
import seaborn as sb
from io import StringIO

import numpy as np
import pandas as pd

def SM_graphs(V, bools, mesh, Q, T):
    bools = np.array(pd.DataFrame(bools).apply(lambda row: row.map({1:'exercise',0:'wait',-1:'option out'}),axis = 1))
    b, values_per_life = mesh.shape
    time_matrix = np.tile(np.arange(1, values_per_life + 1) / values_per_life * T , (b,1))
    fig, axs = plt.subplots()
    sb.set_style("ticks",{'axes.grid' : True})
    g = sb.scatterplot(x = time_matrix.flatten(), y = mesh.flatten(), hue = Q.flatten(), linewidth=0, ax = axs, palette = sb.color_palette("rocket", as_cmap=True))
    g.set_title('Option price based on underlying value in time')
    g.set_xlabel('Time (years)')
    g.set_xlim([0,T*1.1])
    g.set_ylabel('Underlying price')
    g.set_ylim([mesh.min()*0.9, mesh.max()*1.1])
    axs.legend([],[], frameon=False)
    norm = plt.Normalize(Q.min(), Q.max())
    cmap = sb.color_palette("rocket", as_cmap=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([axs.get_position().x1+0.05, axs.get_position().y0, 0.06, axs.get_position().height])
    axs.figure.colorbar(sm, cax=cax)
    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    
    fig, axs = plt.subplots()
    sb.set_style("ticks",{'axes.grid' : True})
    g = sb.scatterplot(x = time_matrix.flatten(), y = mesh.flatten(), hue = bools.flatten(), linewidth=0, palette = {'exercise':'#84b701','wait':'#448ee4','option out':'#cf524e'}, ax = axs)
    g.set_title('Moments of early exercise based on underlying value in time')
    g.set_title('Option price based on underlying value in time')
    g.set_xlabel('Time (years)')
    g.set_xlim([0,T*1.1])
    g.set_ylabel('Underlying price')
    fig.savefig(imgdata, format='svg')
    data = imgdata.getvalue()
    return data