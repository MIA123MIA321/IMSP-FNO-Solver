from mumps import DMumpsContext
from utils import *

ctx = DMumpsContext()
ctx.set_silent()
    

def CG_method(N, Q, k, F) -> np.ndarray:
    # (Q,F) = (((N+1)**2,),((N+1)**2,)) --> ((N+1)**2,)
    res = cg(Matrix_Gen(N, Q, k), F)
    if res[1] == 0:
        return res[0]
    print('can not converge')


def Matrix_analysis(N, k = 2):
    global ctx
    Q = np.zeros((N+1) ** 2,)
    _Matrix_ = Matrix_Gen(N, Q, k)
    ctx.set_shape(_Matrix_.shape[0])
    if ctx.myid == 0:
        ctx.set_centralized_assembled_rows_cols(
            _Matrix_.row + 1, _Matrix_.col + 1)
    ctx.run(job = 1)


def Matrix_factorize(N, k, Q = None):
    # Q:((N+1)**2,)
    global ctx
    if Q is None:
        Q = np.zeros((N+1) ** 2,)
    _Matrix_ = Matrix_Gen(N, Q, k)
    ctx.set_centralized_assembled_values(_Matrix_.data)
    ctx.run(job = 2)
    return


def Matrix_solve(F: np.ndarray):
    # F:((N+1)**2,) --> ((N+1)**2,)
    global ctx
    M = F.shape[0]
    F = np.append(F.real, F.imag)
    _Right_ = F
    x = _Right_.copy()
    ctx.set_rhs(x)
    ctx.run(job = 3)
    tmp = x.reshape(-1, )
    return tmp[:M] + 1j * tmp[M:]


def mathscr_F0(N, Q, k, F, solver = 'MUMPS'):
    # (Q,F):(((N+1)**2,),((N+1)**2,)) --> ((N+1)**2,)
    global ctx
    if solver == 'CG':
        return CG_method(N, Q, k, F)
    elif solver == 'MUMPS':
        Matrix_factorize(N, Q, k)
        return Matrix_solve(F)
    
    
def NET_method(N, q, k, f, NET, length = 3):
    if torch.norm(f) < 1e-8:
        return torch.zeros_like(f).to(device)
    U_NET = NET(torch.cat([torch.ones_like(q), -f/(k*k)],1))
    SUM_NET = U_NET.clone().detach()
    if torch.norm(q) > 1e-8:
        for i in range(length):
            U_NET_TMP = U_NET.clone().detach()
            del U_NET
            U_NET = NET(torch.cat([q,U_NET_TMP],1))
            del U_NET_TMP
            SUM_NET_OLD = SUM_NET.clone().detach()
            del SUM_NET
            SUM_NET = SUM_NET_OLD + U_NET
            del SUM_NET_OLD
    del U_NET
    return SUM_NET 