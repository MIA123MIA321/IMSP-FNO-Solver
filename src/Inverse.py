from scipy.optimize import minimize
from Solver import *
from q_method import f_gen

def data_gen(Q_gen, N_gen, N_src, N_rec_list, N_buffer_gen, bd_num, bd_type, k_list, noise_level,scheme):
    k_len = len(k_list)
    f_data_list = []
    partial_data_list = []
    for j in range(k_len):
        f_data_list.append(f_gen(N_gen, k_list[j], N_src))
    Matrix_analysis(N_gen,scheme=scheme)
    for j in range(k_len):
        times = N_gen // N_rec_list[j]
        N_buffer_rec = N_buffer_gen // times
        Matrix_factorize(N_gen, k_list[j], Q_gen, scheme=scheme)
        partial_data = np.zeros((N_src, bd_num * (N_rec_list[j] -2*N_buffer_rec - 1)), dtype = np.complex128)
        for i in range(N_src):
            tmp_u = Matrix_solve(- k_list[j] ** 2 * Q_gen * f_data_list[j][i].reshape(-1,),
                                 False,scheme=scheme)
            partial_data[i] = Round(data_projection(tmp_u, N_gen, N_rec_list[j], True, bd_num, bd_type), noise_level, 1)
        partial_data_list.append(partial_data)
    return f_data_list, partial_data_list

def J_MUMPS(Q, N_args, bd_args, data_args, comp_args, return_grad):
    N_gen, N_src, N_rec, N_comp, N_buffer = N_args
    bd_num, bd_type =  bd_args
    k, f_data, partial_data = data_args
    NET, device, NS_length, scheme = comp_args
    J_value = 0.
    f_data = INTERPOLATE(f_data, N_gen, N_comp)
    if return_grad:
        J_grad = np.zeros_like(Q)
    Matrix_factorize(N_comp, k, Q, scheme=scheme)
    for j in range(N_src):
        phi = Matrix_solve(- k * k * Q * f_data[j].reshape(-1,),False,scheme=scheme)
        J_inner = data_projection(phi, N_comp, N_rec, True, bd_num, bd_type) - partial_data[j]
        J_value += np.linalg.norm(J_inner, ord = 2) ** 2
        if return_grad:
            fun1 = (f_data[j] + phi).reshape(-1,)
            fun2 = - k * k * data_projection(J_inner, N_comp, N_rec, False, bd_num, bd_type).reshape(-1,)
            tmp_fun = Matrix_solve(fun2.real, True, scheme=scheme)
            tmpr = fun1.real * tmp_fun.real - fun1.imag * tmp_fun.imag
            tmp_fun = Matrix_solve(fun2.imag,True,scheme=scheme)
            tmpi = fun1.imag * tmp_fun.real + fun1.real * tmp_fun.imag
            J_grad += (tmpr + tmpi) 
    J_value = 0.5 * J_value / N_src
    if return_grad:
        J_grad = J_grad / N_src
        return J_value, J_grad
    else:
        return J_value


def J_NET(Q, N_args, bd_args, data_args, comp_args, return_grad):
    N_gen, N_src, N_rec, N_comp, N_buffer = N_args
    bd_num, bd_type =  bd_args
    k, f_data, partial_data = data_args
    NET, device, NS_length, scheme = comp_args
    J_value = 0.
    f_data = INTERPOLATE(f_data, N_gen, N_comp)
    if return_grad:
        J_grad = np.zeros_like(Q)
    q_torch = torch.from_numpy(Q.reshape(1, 1, N_comp + 1, N_comp + 1)).to(
                torch.float32).to(device).repeat(N_src, 1, 1, 1)
    q_torch1 = q_torch.repeat(1, 2, 1, 1)
    phi = NET_method(N_comp, N_comp, q_torch, k, -k * k * q_torch1 * f_data, NET, device, NS_length)
    J_inner = data_projection(phi, N_comp, N_rec, True, bd_num, bd_type) - partial_data
    J_value = torch.norm(J_inner).item() ** 2
    J_value = 0.5 * J_value / N_src
    if return_grad:
        fun1 = f_data + phi
        fun2 = - k * k * data_projection(J_inner, N_comp, N_rec, False, bd_num, bd_type).to(device)
        tmp_fun = NET_method(N_comp, N_comp, q_torch, k,
                torch.stack([fun2[:, 0], torch.zeros_like(fun2[:, 0])], 1), NET, device, NS_length)
        tmpr = fun1[:,0] * tmp_fun[:,0] - fun1[:,1] * tmp_fun[:,1]
        tmp_fun = NET_method(N_comp, N_comp, q_torch, k,
                torch.stack([fun2[:, 1], torch.zeros_like(fun2[:, 1])], 1), NET, device, NS_length)
        tmpi = fun1[:,1] * tmp_fun[:,0] + fun1[:,0] * tmp_fun[:,1]
        J_grad_tmp = (tmpr + tmpi).reshape(-1, (N_comp + 1) ** 2).cpu().detach().numpy()
        for j in range(N_src):
            J_grad += J_grad_tmp[j] 
        J_grad = J_grad / N_src
        return J_value, J_grad
    else:
        return J_value


def J_single_frequency(Q,*argsargs):
    N_args, bd_args, data_args, comp_args, return_grad = argsargs
    if comp_args[0] is None:
        return J_MUMPS(Q, N_args, bd_args, data_args, comp_args, return_grad)
    else:
        return J_NET(Q, N_args, bd_args, data_args, comp_args, return_grad)
    

def SOLVE(fun,Q0,args,jac,method='L-BFGS-B',
        options={'disp': True, 'gtol': 1e-5, 'maxiter': 100}):
    if method == 'L-BFGS-B' or method == 'CG':
        res = minimize(fun, x0 = Q0, args = args, method = method, jac = jac,
                        options = options, callback = callbackF)
        return res.success, res.x