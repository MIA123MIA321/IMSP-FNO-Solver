from scipy.optimize import minimize
from Solver import *
from q_method import f_gen

def data_gen(Q_gen, N_gen, N, k_list, m, noise_level):
    times = N_gen // N
    k_len = len(k_list)
    f_data = np.zeros((k_len, m, (N_gen + 1), (N_gen + 1)), dtype = np.complex128)
    for j in range(k_len):
        f_data[j] = f_gen(N_gen, k_list[j], m)
    f_data_output = f_data[...,::times,::times]
    partial_data = np.zeros((k_len, m, 4 * N - 4), dtype = np.complex128)
    Matrix_analysis(N_gen)
    for j in range(k_len):
        Matrix_factorize(N_gen, k_list[j], Q_gen)
        for i in range(m):
            tmp_u = Matrix_solve(Q_gen * f_data[j, i].reshape(-1,),False)[::times,::times]
            partial_data[j, i] = Round(data_projection(tmp_u,True), noise_level)
    return f_data_output, partial_data


def J_MUMPS(Q, N, N_comp, k, f_data, partial_data, device, return_grad = True):
    J_value = 0.
    m = f_data.shape[0]
    times = N // N_comp
    Q_comp = Q.reshape((N+1,N+1))[::times,::times].reshape(-1,)
    f_data_comp = f_data.reshape((m,N+1,N+1))[:,::times,::times].reshape((m,-1))
    Matrix_factorize(N_comp, k, Q_comp)
    if return_grad:
        J_grad = np.zeros_like(Q)
        res_tmp = np.ones(((N + 1), (N + 1)))
        res_tmp[0, 0] = res_tmp[-1, 0] = res_tmp[0, -1] = res_tmp[-1, -1] = 0.5
        res_tmp[1:-1, 1:-1] = 2
        res_tmp = res_tmp.reshape(-1,)
    for j in range(m):
        phi = Matrix_solve(Q_comp * f_data_comp[j],False)
        phi = INTERPOLATE(phi,N_comp,N,device)
        J_inner = data_projection(phi,True) - partial_data[j]
        J_value += np.linalg.norm(J_inner, ord = 2) ** 2
        if return_grad:
            fun1 = (f_data[j] - k * k * phi).reshape(-1,)
            fun2 = data_projection(J_inner,False)
            fun2 = fun2.reshape((N+1,N+1))[::times,::times].reshape(-1,)
            tmp_fun = Matrix_solve(fun2.real,True)
            tmp_fun = INTERPOLATE(tmp_fun,N_comp,N,device).reshape(-1,)
            tmpr = fun1.real * tmp_fun.real - fun1.imag * tmp_fun.imag
            tmp_fun = Matrix_solve(fun2.imag,True)
            tmp_fun = INTERPOLATE(tmp_fun,N_comp,N,device).reshape(-1,)
            tmpi = fun1.imag * tmp_fun.real + fun1.real * tmp_fun.imag
            J_grad += (tmpr + tmpi) * res_tmp
    J_value = 0.5 * k ** 4 * J_value / m
    if return_grad:
        J_grad = k ** 4 * J_grad / m
        return J_value, J_grad
    else:
        return J_value


def J_NET(Q, N, N_comp, k, f_data, partial_data, NET, device, NS_length, return_grad = True):
    J_value = 0.
    m = partial_data.shape[0]
    if return_grad:
        J_grad = np.zeros_like(Q)
        res_tmp = np.ones(((N + 1), ( N + 1)))
        res_tmp[0, 0] = res_tmp[-1, 0] = res_tmp[0, -1] = res_tmp[-1, -1] = 0.5
        res_tmp[1:-1, 1:-1] = 2
        res_tmp = res_tmp.reshape(-1,)
    q_torch = torch.from_numpy(Q.reshape(1, 1, N + 1, N + 1)).to(
                torch.float32).to(device).repeat(m, 1, 1, 1)
    q_torch1 = q_torch.repeat(1, 2, 1, 1)
    phi = NET_method(N, N_comp, q_torch, k, q_torch1 * f_data, NET, device, NS_length)
    J_inner = data_projection(phi,True) - partial_data
    J_value = torch.norm(J_inner).item() ** 2
    J_value = 0.5 * k ** 4 * J_value / m
    if return_grad:
        fun1 = f_data - k * k * phi
        fun2 = data_projection(J_inner,False).to(device)
        tmp_fun = NET_method(N, N_comp, q_torch, k,
                torch.stack([fun2[:, 0], torch.zeros_like(fun2[:, 0])], 1), NET, device, NS_length)
        tmpr = fun1[:,0] * tmp_fun[:,0] - fun1[:,1] * tmp_fun[:,1]
        tmp_fun = NET_method(N, N_comp, q_torch, k,
                torch.stack([fun2[:, 1], torch.zeros_like(fun2[:, 1])], 1), NET, device, NS_length)
        tmpi = fun1[:,1] * tmp_fun[:,0] + fun1[:,0] * tmp_fun[:,1]
        J_grad_tmp = (tmpr + tmpi).reshape(-1, (N + 1) ** 2).cpu().detach().numpy()
        for j in range(m):
            J_grad += J_grad_tmp[j] * res_tmp
        J_grad = k ** 4 * J_grad / m
        return J_value, J_grad
    else:
        return J_value


def J_single_frequency(Q,*argsargs):
    N, N_comp, k, f_data, partial_data, maxq, NET, device, NS_length, return_grad = argsargs
    if NET is None:
        return J_MUMPS(Q, N, N_comp, k, f_data, partial_data, device, return_grad)
    else:
        return J_NET(Q, N, N_comp, k, f_data, partial_data, NET, device, NS_length, return_grad)


def SOLVE(fun,Q0,args,jac,method='L-BFGS-B',
        options={'disp': True, 'gtol': 1e-5, 'maxiter': 100}):
    """
    args = N, q, k, arg_list
    """
    if method == 'L-BFGS-B' or method == 'CG':
        res = minimize(fun, x0 = Q0, args = args, method = method, jac = jac,
                        options = options, callback = callbackF)
        return res.success, res.x