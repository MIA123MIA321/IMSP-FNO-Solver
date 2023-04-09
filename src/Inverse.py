from scipy.optimize import minimize
from Solver import *


def pdata_gen(N, Qt, k, f_multi_angle, matrix_A, noise_level = 0.0):
    Matrix_factorize(N, k, Qt)
    m = f_multi_angle.shape[0]
    res = np.zeros((m, 4 * N - 4), dtype = np.complex128)
    for i in range(m):
        res[i] = matrix_A @ Matrix_solve(Qt * f_multi_angle[i])
        res[i] = Round(res[i], noise_level)
    return res


def J_MUMPS(Q, N, partial_data, k, f_data, matrix_A, return_grad = True):
    J_value = 0.
    m = f_data.shape[0]
    Matrix_factorize(N, k, Q)
    if return_grad:
        J_grad = np.zeros_like(Q)
        res_tmp = np.ones(((N + 1), (N + 1)))
        res_tmp[0, 0] = res_tmp[-1, 0] = res_tmp[0, -1] = res_tmp[-1, -1] = 0.5
        res_tmp[1:-1, 1:-1] = 2
        res_tmp = res_tmp.reshape(-1,)
    for j in range(m):
        phi = Matrix_solve(Q * f_data[j])
        J_inner = matrix_A @ phi - partial_data[j]
        J_value += np.linalg.norm(J_inner, ord = 2) ** 2
        if return_grad:
            fun1 = f_data[j] - k * k * phi
            fun2 = matrix_A.T @ J_inner
            tmp_fun = Matrix_solve(fun2.real)
            tmpr = fun1.real * tmp_fun.real - fun1.imag * tmp_fun.imag
            tmp_fun = Matrix_solve(fun2.imag)
            tmpi = fun1.imag * tmp_fun.real + fun1.real * tmp_fun.imag
            J_grad += (tmpr + tmpi) * res_tmp
    J_value = 0.5 * k ** 4 * J_value / m
    if return_grad:
        J_grad = k ** 4 * J_grad / m
        return J_value, J_grad
    else:
        return J_value


def J_NET(Q, N, partial_data, k, f_data, matrix_A, NET, device, NS_length, return_grad = True):
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
    phi = NET_method(N, q_torch, k, q_torch1 * f_data, NET, NS_length)
    J_inner = torch.einsum('abcd,abde->abce', matrix_A, phi.reshape(m, 2,
                                    (N + 1)**2, 1)) - partial_data.unsqueeze(-1)
    J_value = torch.norm(J_inner).item() ** 2
    J_value = 0.5 * k ** 4 * J_value / m
    if return_grad:
        fun1 = f_data - k * k * phi
        fun2 = torch.einsum('abcd,abde->abce', matrix_A.permute(0, 1, 3, 2),
                            J_inner).reshape(m, 2, N + 1, N + 1)
        tmp_fun = NET_method(N, q_torch, k,
                torch.stack([fun2[:, 0], torch.zeros_like(fun2[:, 0])], 1), NET, NS_length)
        tmpr = fun1[:,0] * tmp_fun[:,0] - fun1[:,1] * tmp_fun[:,1]
        tmp_fun = NET_method(N, q_torch, k,
                torch.stack([fun2[:, 1], torch.zeros_like(fun2[:, 1])], 1), NET, NS_length)
        tmpi = fun1[:,1] * tmp_fun[:,0] + fun1[:,0] * tmp_fun[:,1]
        J_grad_tmp = (tmpr + tmpi).reshape(-1, (N + 1) ** 2).cpu().detach().numpy()
        for j in range(m):
            J_grad += J_grad_tmp[j] * res_tmp
        J_grad = k ** 4 * J_grad / m
        return J_value, J_grad
    else:
        return J_value


def J_MULTI(Q,*argsargs):
    N, partial_data, k_list, f_data, matrix_A, maxq, NET, device, NS_length, return_grad = argsargs
    J_value = 0.
    if return_grad:
        J_grad = np.zeros_like(Q)
    if NET is None:
        for i in range(len(k_list)):
            tmp_data = J_MUMPS(Q, N, partial_data[0][i], k_list[i], f_data[0][i], matrix_A[0], return_grad)
            if return_grad:
                J_value += tmp_data[0]
                J_grad += tmp_data[1]
            else:
                J_value += tmp_data
    else:
        for i in range(len(k_list)):
            tmp_data = J_NET(Q, N,partial_data[1][i], k_list[i], f_data[1][i],
                         matrix_A[1], NET, device, NS_length, return_grad)
            if return_grad:
                J_value += tmp_data[0]
                J_grad += tmp_data[1]
            else:
                J_value += tmp_data
    if return_grad:
        return J_value, J_grad
    else:
        return J_value


def SOLVE(fun,Q0,args,jac,method='L-BFGS-B',
        options={'disp': True, 'gtol': 1e-5, 'maxiter': 100}):
    """
    args = N, q, k, arg_list
    """
    if method == 'L-BFGS-B' or method == 'CG':
        res = minimize(fun, x0 = Q0, args = args, method = method, jac = jac,
                        options = options, callback = callbackF)
        return res.success, res.x