from mumps import DMumpsContext
from utils import *

ctx = DMumpsContext()
ctx.set_silent()


def Matrix_Gen_5(N, Q, k, expand_times,Transpose=False):
    '''
    data1 : middle
    data2 : middle +- 1
    data3 : middle +- M
    data4 : middle +- M^2
    '''
    M = N*expand_times + 1
    Q = expand_grids(Q.reshape(N+1,N+1),expand_times).reshape(-1,)
    Matrix1 = (k * k * (1 + Q) - 4 * N * N).reshape(M,M)
    data1 = np.tile(Matrix1.reshape(-1,),2)
    
    Matrix2_plus = np.ones((M,M))
    Matrix2_plus[:,0] = 0
    Matrix2_plus[:,1] = 2
    Matrix2_plus *= N * N
    data2_plus = np.tile(Matrix2_plus.reshape(-1,),2)
    Matrix2_minus = Matrix2_plus[:,::-1]
    data2_minus = np.tile(Matrix2_minus.reshape(-1,),2)
    
    Matrix3_plus = Matrix2_plus.T
    data3_plus = np.tile(Matrix3_plus.reshape(-1,),2)
    Matrix3_minus = Matrix3_plus[::-1,:]
    data3_minus = np.tile(Matrix3_minus.reshape(-1,),2)

    Matrix4 = np.ones((M, M))
    Matrix4[0, 0] = Matrix4[-1, 0] = Matrix4[-1, -1] = Matrix4[0, -1] = 2
    Matrix4[1:-1, 1:-1] = 0
    Matrix4 *= 2 * k * N
    data4_minus = np.tile(Matrix4.reshape(-1), 2)
    data4_plus = -data4_minus
    if Transpose:
        data = (np.c_[data1,data2_minus,data2_plus,
                data3_minus,data3_plus,data4_plus,
                data4_minus]).transpose()
    else:
        data = (np.c_[data1,data2_minus,data2_plus,
            data3_minus,data3_plus,data4_minus,
            data4_plus]).transpose()
    offsets = np.array([0, -1, 1, -M, M, -M * M, M * M])
    dia = dia_matrix((data, offsets), shape=(2 * M * M, 2 * M * M))
    mat = dia.tocoo()
    if Transpose:
        mat = mat.T
    return mat


def Matrix_Gen_9(N, Q, k):
    '''
    data1 : middle
    data2 : middle +- 1
    data3 : middle +- M
    data4 : middle + M +- 1
    data5 : middle - M +- 1
    data6 : middle +- M^2
    data7 : middle - M^2 +- 1
    data8 : middle - M^2 +- M
    data9 : middle + M^2 +- 1
    data10 : middle + M^2 +- M
    '''
    M = N + 1
    Matrix1 = (k * k * (1 + Q) - 10 * N * N / 3).reshape(M,M)
    Matrix1[0,0] -= 2 * k * k / 3
    Matrix1[0,-1] -= 2 * k * k / 3
    Matrix1[-1,0] -= 2 * k * k / 3
    Matrix1[-1,-1] -= 2 * k * k / 3
    data1 = np.tile(Matrix1.reshape(-1,),2)
    
    Matrix2_plus = np.ones((M,M)) * N * N * 2 / 3
    Matrix2_plus[:,0] = 0
    Matrix2_plus[:,1] *= 2
    data2_plus = np.tile(Matrix2_plus.reshape(-1,),2)
    Matrix2_minus = Matrix2_plus[:,::-1]
    data2_minus = np.tile(Matrix2_minus.reshape(-1,),2)
    
    Matrix3_plus = Matrix2_plus.T
    data3_plus = np.tile(Matrix3_plus.reshape(-1,),2)
    Matrix3_minus = Matrix3_plus[::-1,:]
    data3_minus = np.tile(Matrix3_minus.reshape(-1,),2)
    
    Matrix4_plus = np.ones((M, M)) * N * N / 6
    Matrix4_plus[0,:] = 0
    Matrix4_plus[:,0] = 0
    Matrix4_plus[1,:] *= 2
    Matrix4_plus[:,1] *= 2
    Matrix4_plus[1,1] *= 4
    data4_plus = np.tile(Matrix4_plus.reshape(-1,), 2)
    Matrix4_minus = Matrix4_plus[:,::-1]
    data4_minus = np.tile(Matrix4_minus.reshape(-1), 2)
    
    Matrix5_plus = Matrix4_plus[::-1,:]
    Matrix5_minus = Matrix4_minus[::-1,:]
    data5_plus = np.tile(Matrix5_plus.reshape(-1), 2)
    data5_minus = np.tile(Matrix5_minus.reshape(-1), 2)
    
    
    Matrix6 = np.zeros((M, M))
    Matrix6[0,1:-1] = 2 * k * N * 2 / 3
    Matrix6[-1,1:-1] = 2 * k * N * 2 / 3
    Matrix6[1:-1,0] = 2 * k * N * 2 / 3
    Matrix6[1:-1,-1] = 2 * k * N * 2 / 3
    data6_minus = np.tile(Matrix6.reshape(-1), 2)
    data6_plus = -data6_minus
    
    Matrix7_plus = np.zeros((M,M))
    Matrix7_plus[0,1] = 4 * k * N / 6
    Matrix7_plus[0,2:] = 2 * k * N / 6
    Matrix7_plus[-1] = Matrix7_plus[0]
    data7_plus = np.tile(Matrix7_plus.reshape(-1,),2)
    Matrix7_minus = Matrix7_plus[:,::-1]
    data7_minus = np.tile(Matrix7_minus.reshape(-1,),2)
    
    Matrix8_plus = Matrix7_plus.T
    data8_plus = np.tile(Matrix8_plus.reshape(-1,),2)
    Matrix8_minus = Matrix8_plus[::-1,:]
    data8_minus = np.tile(Matrix8_minus.reshape(-1,),2)
    
    data9_plus = - data7_plus
    data9_minus = - data7_minus
    data10_plus = - data8_plus
    data10_minus = -data8_minus

    data = (np.c_[data1,
                  data2_plus,data2_minus,
                  data3_plus,data3_minus,
                  data4_plus,data4_minus,
                  data5_plus,data5_minus,
                  data6_plus,data6_minus,
                  data7_plus,data7_minus,
                  data8_plus,data8_minus,
                  data9_plus,data9_minus,
                  data10_plus,data10_minus]).transpose()
    offsets = np.array([0,
                        1, -1,
                        M, -M,
                        M + 1,M - 1,
                        -M + 1, -M - 1,
                        M * M, -M * M,
                       -M * M + 1, -M*M -1,
                       -M * M + M, -M*M -M,
                       M * M + 1, M*M -1,
                       M * M + M, M*M -M])
    dia = dia_matrix((data, offsets), shape=(2 * M * M, 2 * M * M))
    mat = dia.tocoo()
    return mat


def Matrix_PML(N,Q,k,thickness=0.05, gamma = 2.8,Transpose=False):   
    h = 1/N
    constant = k
    N_PML = int(N*thickness)
    N_total = N + 2*N_PML
    M = N_total-1
    q_total = np.pad(Q.reshape(N+1,N+1),N_PML)
    A = np.ones((N_total,N_total+1),dtype = np.complex128)
    for j in range(N_PML):
        A[:,j] -= 1j*constant*(N_PML-j)**2/(N_PML**2)
        A[:,-j-1] -= 1j*constant*(N_PML-j)**2/(N_PML**2)
    for i in range(N_PML):
        A[i] /= (1-1j*constant*(N_PML-i-0.5)**2/(N_PML**2))
        A[-i-1] /= (1-1j*constant*(N_PML-i-0.5)**2/(N_PML**2))
    B = A.T
    C = np.ones((N_total+1,N_total+1),dtype = np.complex128)
    for i in range(N_PML):
        C[i] -= 1j*constant*(N_PML-i)**2/(N_PML**2)
        C[-i-1] -= 1j*constant*(N_PML-i)**2/(N_PML**2)
    for j in range(N_PML):
        C[:,j] *= (1-1j*constant*(N_PML-j)**2/(N_PML**2))
        C[:,-j-1] *= (1-1j*constant*(N_PML-j)**2/(N_PML**2))
    C *= k*k*(1+q_total)
    M0 = (2/3+gamma/36)*h*h*C[1:-1,1:-1] - 5/6*(A[:-1,1:-1]+A[1:,1:-1]+B[1:-1,1:]+B[1:-1,:-1])
    m0 = np.tile(M0.reshape(-1,),2)[0:0+M*M]
    M1 = 1/12*(1-gamma/6)*h*h*C[:-2,1:-1] + 5/6*A[:-1,1:-1] - 1/12*(B[:-2,:-1]+B[:-2,1:])
    M1[0] *= 0
    m1 = np.tile(M1.reshape(-1,),2)[M:M+M*M]
    M2 = 1/12*(1-gamma/6)*h*h*C[2:,1:-1] + 5/6*A[1:,1:-1] - 1/12*(B[2:,:-1]+B[2:,1:])
    M2[-1] *= 0
    m2 = np.tile(M2.reshape(-1,),2)[M*M-M:M*M-M+M*M]
    M3 = 1/12*(1-gamma/6)*h*h*C[1:-1,:-2] + 5/6*B[1:-1,:-1] - 1/12*(A[:-1,:-2]+A[1:,:-2])
    M3[:,0] *= 0
    m3 = np.tile(M3.reshape(-1,),2)[1:1+M*M]
    M4 = 1/12*(1-gamma/6)*h*h*C[1:-1,2:] + 5/6*B[1:-1,1:] - 1/12*(A[:-1,2:]+A[1:,2:])
    M4[:,-1] *= 0
    m4 = np.tile(M4.reshape(-1,),2)[M*M-1:M*M-1+M*M]
    M5 = 1/12*(A[:-1,:-2]+B[:-2,:-1]) + gamma*h*h/144*C[:-2,:-2]
    M5[0] *= 0
    M5[:,0] *= 0
    m5 = np.tile(M5.reshape(-1,),2)[M+1:M+1+M*M]
    M6 = 1/12*(A[1:,:-2]+B[2:,:-1])+ gamma*h*h/144*C[2:,:-2]
    M6[-1] *= 0
    M6[:,0] *= 0
    m6 = np.tile(M6.reshape(-1,),2)[M*M-M+1:M*M-M+1+M*M]
    M7 = 1/12*(A[:-1,2:]+B[:-2,1:])+ gamma*h*h/144*C[:-2,2:]
    M7[0] *= 0
    M7[:,-1] *= 0
    m7 = np.tile(M7.reshape(-1,),2)[M-1:M-1+M*M]
    M8 = 1/12*(A[1:,2:]+B[2:,1:])+ gamma*h*h/144*C[2:,2:]
    M8[-1] *= 0
    M8[:,-1] *= 0
    m8 = np.tile(M8.reshape(-1,),2)[M*M-M-1:M*M-M-1+M*M]
    if Transpose:
        data = (np.c_[np.tile(m0,2).real,np.tile(m1,2).real,np.tile(m2,2).real,
                      np.tile(m3,2).real,np.tile(m4,2).real,np.tile(m5,2).real,
                      np.tile(m6,2).real,np.tile(m7,2).real,np.tile(m8,2).real,
                      np.tile(m0,2).imag,np.tile(m1,2).imag,np.tile(m2,2).imag,
                      np.tile(m3,2).imag,np.tile(m4,2).imag,np.tile(m5,2).imag,
                      np.tile(m6,2).imag,np.tile(m7,2).imag,np.tile(m8,2).imag,
                      -np.tile(m0,2).imag,-np.tile(m1,2).imag,-np.tile(m2,2).imag,
                      -np.tile(m3,2).imag,-np.tile(m4,2).imag,-np.tile(m5,2).imag,
                      -np.tile(m6,2).imag,-np.tile(m7,2).imag,-np.tile(m8,2).imag]).transpose()
    else:
        data = (np.c_[np.tile(m0,2).real,np.tile(m1,2).real,np.tile(m2,2).real,
              np.tile(m3,2).real,np.tile(m4,2).real,np.tile(m5,2).real,
              np.tile(m6,2).real,np.tile(m7,2).real,np.tile(m8,2).real,
              -np.tile(m0,2).imag,-np.tile(m1,2).imag,-np.tile(m2,2).imag,
              -np.tile(m3,2).imag,-np.tile(m4,2).imag,-np.tile(m5,2).imag,
              -np.tile(m6,2).imag,-np.tile(m7,2).imag,-np.tile(m8,2).imag,
              np.tile(m0,2).imag,np.tile(m1,2).imag,np.tile(m2,2).imag,
              np.tile(m3,2).imag,np.tile(m4,2).imag,np.tile(m5,2).imag,
              np.tile(m6,2).imag,np.tile(m7,2).imag,np.tile(m8,2).imag]).transpose()
    offsets = np.array([0,
                        -M, M,
                        -1, 1,
                        -M - 1, M - 1,
                        -M + 1, M + 1,
                        M*M,
                        M*M-M, M*M+M,
                        M*M-1, M*M+1,
                        M*M-M - 1, M*M+M - 1,
                        M*M-M + 1, M*M+M + 1,
                        -M*M,
                        -M*M-M, -M*M+M,
                        -M*M-1, -M*M+1,
                        -M*M-M - 1, -M*M+M - 1,
                        -M*M-M + 1, -M*M+M + 1])
    dia = dia_matrix((data, offsets), shape=(2 * M * M, 2 * M * M))
    mat = dia.tocoo()
    if Transpose:
        mat = mat.T
    return mat




def Matrix_Gen(N, Q, k,scheme = 'ABC',expand_times=2,thickness = 0.05,ToTensor=False,Transpose=False):
    assert scheme=='ABC' or scheme=='ABC9' or scheme=='PML'
    if scheme == 'ABC':
        mat = Matrix_Gen_5(N,Q,k,expand_times,Transpose)
    elif scheme == 'PML':
        mat = Matrix_PML(N,Q,k,thickness,Transpose)
    else:
        mat = Matrix_Gen_9(N,Q,k)
    if not ToTensor:
        return mat
    else:
        mat = dia.tocoo()
        values = torch.tensor(mat.data)
        indices = torch.tensor(np.array([mat.row, mat.col]), dtype=torch.long)
        shape = torch.Size(mat.shape)
        torch_sparse_mat = torch.sparse_coo_tensor(indices, values, shape)
        return torch_sparse_mat
    
    
def F_laplacian(F):
    N = int(np.sqrt(F.shape[0])) - 1
    f = F.reshape((N+1,N+1))
    f1 = np.zeros_like(f)
    f1[1:-1,1:-1] = f[2:,1:-1] + f[:-2,1:-1] + f[1:-1,:-2] + f[1:-1,2:] - 4*f[1:-1,1:-1]
    f1[0,0] = 4*f[0,0]-5*f[1,0]+4*f[2,0]-f[3,0]-5*f[0,1]+4*f[0,2]-f[0,3]
    f1[0,-1] = 4*f[0,-1]-5*f[1,-1]+4*f[2,-1]-f[3,-1]-5*f[0,-2]+4*f[0,-3]-f[0,-4]
    f1[-1,0] = 4*f[-1,0]-5*f[-2,0]+4*f[-3,0]-f[-4,0]-5*f[-1,1]+4*f[-1,2]-f[-1,3]
    f1[-1,-1] = 4*f[-1,-1]-5*f[-2,-1]+4*f[-3,-1]-f[-4,-1]-5*f[-1,-2]+4*f[-1,-3]-f[-1,-4]
    f1[0,1:-1] = f[0,:-2]+f[0,2:]-5*f[1,1:-1]+4*f[2,1:-1]-f[3,1:-1]
    f1[-1,1:-1] = f[-1,:-2]+f[-1,2:]-5*f[-2,1:-1]+4*f[-3,1:-1]-f[-4,1:-1]
    f1[1:-1,0] = f[:-2,0]+f[2:,0]-5*f[1:-1,1]+4*f[1:-1,2]-f[1:-1,3]
    f1[1:-1,-1] = f[:-2,-1]+f[2:,-1]-5*f[1:-1,-2]+4*f[1:-1,-3]-f[1:-1,-4]
    return f1.reshape(-1)

    
def Matrix_analysis(N, k = 2, scheme = 'ABC', expand_times = 2,thickness = 0.05,Transpose=False):
    global ctx
    assert scheme=='ABC' or scheme=='ABC9' or scheme=='PML'
    Q = np.zeros((N+1) ** 2,)
    if scheme!='PML':
        _Matrix_ = Matrix_Gen(N, Q, k, scheme, expand_times=expand_times, Transpose=Transpose)
    else:      
        _Matrix_ = Matrix_Gen(N, Q, k, scheme, thickness=thickness, Transpose=Transpose)
    ctx.set_shape(_Matrix_.shape[0])
    if ctx.myid == 0:
        ctx.set_centralized_assembled_rows_cols(
            _Matrix_.row + 1, _Matrix_.col + 1)
    ctx.run(job = 1)


def Matrix_factorize(N, k, Q = None,scheme = 'ABC',expand_times = 2,thickness=0.05,Transpose=False):
    # Q:((N+1)**2,)
    global ctx
    assert scheme=='ABC' or scheme=='ABC9' or scheme=='PML'
    if Q is None:
        Q = np.zeros((N +1) ** 2,)
    if scheme!='PML':
        _Matrix_ = Matrix_Gen(N, Q, k, scheme, expand_times, Transpose=Transpose)
    else:
        _Matrix_ = Matrix_Gen(N, Q, k, scheme, thickness=thickness, Transpose=Transpose)
    ctx.set_centralized_assembled_values(_Matrix_.data)
    ctx.run(job = 2)
    return


def Matrix_solve(F: np.ndarray, one_dim = True,scheme='ABC',expand_times=2,thickness=0.05,Transpose=False):
    # F:((N+1)**2,) --> ((N+1)**2,)
    global ctx
    assert scheme=='ABC' or scheme=='ABC9' or scheme=='PML'
    M = int(np.sqrt(F.shape[0]))
    N = M-1
    h = 1/N
    N_PML = int(N*thickness)
    N_total = N + 2*N_PML
    if scheme!='PML':
        F = expand_grids(F.reshape(M,M),expand_times).reshape(-1,)
    else:
        F = np.pad(F.reshape(M,M),N_PML)
        if not Transpose:
            F = h*h/3*(F[2:,1:-1]+F[:-2,1:-1]+F[1:-1,2:]+F[1:-1,:-2]-F[1:-1,1:-1]).reshape(-1,)
        else:
            F = F[1:-1,1:-1]
    if scheme == 'ABC9':
        F = F + F_laplacian(F)/12
    F = np.append(F.real, F.imag)
    _Right_ = F
    x = _Right_.copy()
    ctx.set_rhs(x)
    ctx.run(job = 3)
    tmp = x.reshape(-1, )
    if scheme!='PML':
        output = tmp[:((M-1)*expand_times+1)**2] + 1j * tmp[((M-1)*expand_times+1)**2:]
        output = squeeze_grids(output,expand_times,one_dim)
    else:
        output = (tmp[:(N_total-1)**2] + 1j * tmp[(N_total-1)**2:]).reshape((N_total-1),(N_total-1))
        output = np.pad(output,1)
        if Transpose:
            output[1:-1,1:-1] = (h*h)/3*(output[2:,1:-1]+output[:-2,1:-1]+output[1:-1,2:]+output[1:-1,:-2]-output[1:-1,1:-1])
        output = output[N_PML:-N_PML,N_PML:-N_PML]
        if one_dim:
            output = output.reshape(-1,)
    return output
    
    
def NET_method(N, N_comp, q, k, f, NET, device, length = 3):
    assert N_comp == 64 or N_comp == 128
    q = INTERPOLATE(q,N,N_comp)
    f = INTERPOLATE(f,N,N_comp)
    if torch.norm(f) < 1e-8:
        ANS = torch.zeros_like(f).to(device)
        return INTERPOLATE(ANS,N_comp,N) 
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
    return INTERPOLATE(SUM_NET,N_comp,N) 