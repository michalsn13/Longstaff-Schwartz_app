import numpy as np

def weights(x_curr_mesh, x_next_mesh, underlying, t):
    b = x_curr_mesh.shape[0]
    dt = 1 / underlying.values_per_year
    f_x_mesh = []
    for x_curr in x_curr_mesh:
        f_x_mesh.append(underlying.transition_dens(x_curr, x_next_mesh))
    g_x_mesh = 1/b * sum(f_x_mesh)
    f_x_mesh = np.array(f_x_mesh)
    return f_x_mesh / g_x_mesh

def stochastic_mesh(option, b):
    dt = 1 / option.underlying.values_per_year
    time = np.tile(np.arange(dt, option.T + dt, dt), (b,1))
    _, mesh = option.underlying.simulate_Q(b, option.T)
    condition = option.barrier_ind_func(mesh, time)
    Q = option.payoff_func(mesh, time) * condition
    exercise_bool = np.ones(Q.shape) * (Q > 0)
    for i in range(1,Q.shape[1]):
        t = option.T - i * dt
        curr_mesh, next_mesh = mesh[:,-(i+1)], mesh[:,-i]
        condition = option.barrier_ind_func(curr_mesh, t) if option.barrier_out else True
        w = weights(curr_mesh, next_mesh, option.underlying, t)
        Q_new = (w * Q[:,-i]).mean(axis = 1) * condition
        exercise_bool[:,-(i+1)] = condition * (Q[:,-(i+1)] > (Q_new * np.exp(-option.underlying.r * dt))) + (-1) * (1 - condition)
        Q[:,-(i+1)] = np.maximum(Q[:,-(i+1)] , Q_new * np.exp(-option.underlying.r * dt))
    V = np.maximum(option.payoff_func(option.underlying.S0, 0), Q[:,0].mean() * np.exp(-option.underlying.r * dt))
    return (V, exercise_bool, mesh, Q)