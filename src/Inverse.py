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


def J(Q, N, partial_data, k, f_data, matrix_A):
    Q_res = np.zeros_like(partial_data)
    Matrix_factorize(N, k, Q)
    m = Q_res.shape[0]
    for i in range(m):
        Q_res[i] = matrix_A @ Matrix_solve(Q * f_data[i])
    res = 0
    for i in range(m):
        res += np.linalg.norm(Q_res[i] - partial_data[i],ord = 2) ** 2
    print('value')
    return 0.5 * k ** 4 * res / m


def J_prime(Q, N, partial_data, k, f_data, matrix_A):
    res = np.zeros_like(Q)
    res_tmp = np.ones(((N + 1), (N + 1)))
    res_tmp[0, 0] = res_tmp[-1, 0] = res_tmp[0, -1] = res_tmp[-1, -1] = 0.5
    res_tmp[1:-1, 1:-1] = 2
    res_tmp = res_tmp.reshape(-1,)
    Matrix_factorize(N, k, Q)
    m = f_data.shape[0]
    for j in range(m):
        phi = Matrix_solve(Q * f_data[j])
        grad = f_data[j] - k * k * phi
        item = matrix_A.T @(matrix_A @ phi - partial_data[j])
        tmpp = Matrix_solve(item.real)
        tmpr = grad.real * tmpp.real - grad.imag * tmpp.imag
        tmpp = Matrix_solve(item.imag)
        tmpi = grad.imag * tmpp.real + grad.real * tmpp.imag
        res+= (tmpr + tmpi) * res_tmp
    print('prime')
    return k ** 4 * res / m


def J_NET(Q, N, partial_data, k, f_data, matrix_A, NET, device, NS_length):
    m = partial_data.shape[0]
    q_torch = torch.from_numpy(Q.reshape(1, 1, N+1, N+1)).to(
                        torch.float32).to(device).repeat(m, 1, 1, 1)
    q_torch1 = q_torch.repeat(1, 2, 1, 1)
    data = NET_method(N, q_torch, k, q_torch1 * f_data, NET, NS_length
                        ).reshape(m, 2, (N + 1) ** 2, 1)
    res = torch.norm(torch.einsum('abcd,abde->abce', matrix_A, data
                                 )[...,0] - partial_data).item() ** 2
    print('value')
    return 0.5 * k ** 4 * res / m


def J_prime_NET(Q, N, partial_data, k, f_data, matrix_A, NET, device, NS_length):
    res = np.zeros_like(Q)
    res_tmp = np.ones(((N + 1), ( N + 1)))
    res_tmp[0, 0] = res_tmp[-1, 0] = res_tmp[0, -1] = res_tmp[-1, -1] = 0.5
    res_tmp[1:-1, 1:-1] = 2
    res_tmp = res_tmp.reshape(-1,)
    m = partial_data.shape[0]
    q_torch = torch.from_numpy(Q.reshape(1, 1, N + 1, N + 1)).to(
                torch.float32).to(device).repeat(m, 1, 1, 1)
    q_torch1 = q_torch.repeat(1, 2, 1, 1)
    phi = NET_method(N, q_torch, k, q_torch1 * f_data, NET, NS_length)
    grad = f_data - k * k * phi
    item = torch.einsum('abcd,abde->abce', matrix_A, phi.reshape(m, 2,
                                    (N + 1)**2, 1)) - partial_data.unsqueeze(-1)
    item = torch.einsum('abcd,abde->abce', matrix_A.permute(0, 1, 3, 2),
                        item).reshape(m,2,N+1,N+1)
    tmpp = NET_method(N, q_torch, k,
            torch.stack([item[:, 0], torch.zeros_like(item[:, 0])], 1), NET, NS_length)
    tmpr = grad[:,0] * tmpp[:,0] - grad[:,1] * tmpp[:,1]
    tmpp = NET_method(N, q_torch, k,
            torch.stack([item[:, 1], torch.zeros_like(item[:, 1])], 1), NET, NS_length)
    tmpi = grad[:,1] * tmpp[:,0] + grad[:,0] * tmpp[:,1]
    plus = (tmpr + tmpi).reshape(-1, (N + 1) ** 2).cpu().detach().numpy()
    for j in range(m):
        res += plus[j] * res_tmp
    print('prime')
    return k ** 4 * res / m


def J_MULTI(Q,*argsargs):
    N, partial_data, k_list, f_data, matrix_A, maxq, NET, device, NS_length = argsargs
    ans = 0
    if NET is None:
        for i in range(len(k_list)):
            ans += J(Q, N, partial_data[0][i], k_list[i], f_data[0][i], matrix_A[0])
    else:
        for i in range(len(k_list)):
            ans += J_NET(Q, N,partial_data[1][i], k_list[i], f_data[1][i],
                         matrix_A[1], NET, device, NS_length)
    return ans


def J_MULTIPRIME(Q,*argsargs):
    N, partial_data, k_list, f_data, matrix_A, maxq, NET, device, NS_length = argsargs
    ans = np.zeros_like(Q)
    if NET is None:
        for i in range(len(k_list)):
            ans += J_prime(Q, N, partial_data[0][i], k_list[i], f_data[0][i], matrix_A[0])
    else:
        for i in range(len(k_list)):
            ans += J_prime_NET(Q, N, partial_data[1][i], k_list[i], f_data[1][i],
                               matrix_A[1], NET, device, NS_length)
    return ans


def SOLVE(fun,Q0,args,jac,method='L-BFGS-B',
        options={'disp': True, 'gtol': 1e-5, 'maxiter': 100}):
    """
    args = N, q, k, arg_list
    """
    if method == 'L-BFGS-B' or method == 'CG':
        res = minimize(fun, x0 = Q0, args = args, method = method, jac = jac,
                        options = options, callback = callbackF)
        return res.success, res.x