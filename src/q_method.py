import random
import numpy as np


def q_T(N, q_value_1 = 1, q_value_2 = -1, x1 = 0.2, x2 = 0.4,
        x3 = 0.7, y1 = 0.2, y2 = 0.3, y3 = 0.6, y4 = 0.7):
    q = np.zeros((N + 1, N + 1))
    q[int(x1 * N):int(x2 * N), int(y1 * N):int(y4 * N)] = q_value_1
    q[int(x2 * N):int(x3 * N), int(y2 * N):int(y3 * N)] = q_value_2
    return q


def q_Gaussian(N, b1 = 0.3, b2 = 0.6, a1 = 150, a2 = 70, gamma = 1):
    """
    q(x,y) = \gamma * \\exp (  -a1(x-b)^2   -a2(y-b2)^2    )
    """
    q = np.zeros((N+1, N+1))
    tmp = np.linspace(1, N-1, N-1)/ N
    Y, X = np.meshgrid(tmp, tmp)
    q[1:-1, 1:-1] = gamma * np.exp(-a1*(X-b1)**2)*np.exp(-a2*(Y-b2)**2)
    return q


def q_Continuous(N):
    q = np.zeros((N+1, N+1))
    tmp = np.linspace(1, N-1, N-1)/ N
    Y, X = np.meshgrid(tmp, tmp)
    X, Y = 6 * X - 3, 6 * Y - 3
    q[1:-1, 1:-1] = 0.3 * (1 - X) ** 2 * np.exp(-X ** 2 - (Y + 1) ** 2) - \
                    (0.2 * X - X ** 3 - Y ** 5) * np.exp(-X ** 2 - Y ** 2) - \
                    0.03 * np.exp(-(X + 1) ** 2 - Y ** 2)
    return q


def q_gen(N, method = 'T', gamma = 1):
    """
    Returns.shape = (N+1,N+1)
    """
    q = np.zeros((N+1,N+1))
    if method == 'T':
        q = q_T(N)
    elif method == 'G':
        q = q_Gaussian(N)
    elif method == 'C':
        q = q_Continuous(N)
    elif method == 'MG':
        q += q_Gaussian(N, 0.3, 0.6, 150, 70, 1)
        q -= q_Gaussian(N, 0.5, 0.3, 120, 80, 0.8)
        q += q_Gaussian(N, 0.8, 0.5, 40, 90, 0.3)
    Max_Value = np.max(np.abs(q))
    return gamma * q / Max_Value


def gen_A(N):
    tmp0 = np.zeros(((N-1)*4,(N+1)**2))
    tmp = np.zeros((N+1,N+1))
    tmp[0] = tmp[-1] = tmp[:,0] = tmp[:,-1] = 1
    tmp[0,0] = tmp[0,-1] = tmp[-1,0] = tmp[-1,-1] = 0
    b = np.where(tmp > 0)
    for i in range(4*(N-1)):
        tmp0[i,b[0][i]*(N+1)+b[1][i]] = 1
    return tmp0


def gen_A1(N):
    tmp = gen_A(N)
    tmp0 = np.zeros_like(tmp)
    return np.array(np.bmat('tmp tmp0;tmp0 tmp'))


def f_gen(N,k,m):
    res = np.zeros((m,(N+1)**2),dtype = np.complex128)
    tmp = np.linspace(0,1,N+1)
    Y,X = np.meshgrid(tmp, tmp)    
    for j in range(m):    
        res[j] = np.exp(1j*k*(X*np.cos(2*np.pi*j/m)+Y*np.sin(2*np.pi*j/m))).reshape(-1)
    return res


def generate_t_shape(N,value1=0.1,value2=-0.1,direct=0,left=0.2,right=0.8):
    x_axis = [int(random.uniform(left, right) * N) for i in range(3)]
    y_axis = [int(random.uniform(left, right) * N) for i in range(4)]
    x_axis.sort()
    y_axis.sort()
    q = np.zeros((N + 1, N + 1))
    q[x_axis[0]:x_axis[1], y_axis[0]:y_axis[3]] = value1
    q[x_axis[1]:x_axis[2], y_axis[1]:y_axis[2]] = value2
    if direct==1:
        return q[::-1,:]
    elif direct==2:
        return q.T
    elif direct==3:
        return q[::-1,:].T
    else:
        return q

def generate_phi_incident(comp_grid, k, order, angle_total):
    phi_incident = np.zeros((comp_grid + 1, comp_grid + 1))
    l = np.linspace(0, 1, comp_grid + 1)
    y,x = np.meshgrid(l, l)
    phii = np.exp(1j * k * (x * np.cos(2 * np.pi * order / angle_total) + 
                            y * np.sin(2 * np.pi * order / angle_total)))
    return phii

def generate_phi_incident_repeat(nsample, comp_grid, k, order, angle_total):
    phi_incident = np.zeros((2, comp_grid + 1, comp_grid + 1))
    phii = generate_phi_incident(comp_grid, k, order, angle_total)                        
    phi_incident[0], phi_incident[1] = phii.real, phii.imag
    res = phi_incident[np.newaxis,...]
    res = np.tile(res,(nsample, 1, 1, 1))                        
    return res

def generate_gauss_shape(N,R,num=6,left=0.1,right=0.9):
    l = np.linspace(0, N, N + 1)
    k1, k2 = np.meshgrid(l, l)
    NNN = random.randint(1, num)
    Q_total = np.zeros((N + 1, N + 1))
    for iter in range(NNN):
        a,c = random.uniform(R/2,R),random.uniform(R/2,R)
        b,d = random.uniform(left,right),random.uniform(left,right)
        lamb = random.uniform(-1,1)
        Q_total += lamb*np.exp(-a*(k1/N-b)**2-c*(k2/N-d)**2)
    return Q_total
    