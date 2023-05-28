import random
import numpy as np
from scipy.ndimage import gaussian_filter

def q_T(N, q_value_1 = 1, q_value_2 = -1, x1 = 0.2, x2 = 0.4,
        x3 = 0.7, y1 = 0.2, y2 = 0.3, y3 = 0.6, y4 = 0.7, sigma = 0):
    q = np.zeros((N + 1, N + 1))
    q[int(x1 * N):int(x2 * N), int(y1 * N):int(y4 * N)] = q_value_1
    q[int(x2 * N):int(x3 * N), int(y2 * N):int(y3 * N)] = q_value_2
    q = gaussian_filter(q, sigma/128*N)
    return q

def q_Square(N, q_value = 1, left = 0.2, right = 0.8, bottom = 0.2, top = 0.8, sigma = 0):
    left1,right1 = min(left,right),max(left,right)
    bottom1,top1 = min(bottom,top),max(bottom,top)
    q = np.zeros((N + 1, N + 1))
    q[int(left1 * N):int(right1 * N), int(bottom1 * N):int(top1 * N)] = q_value
    q = gaussian_filter(q, sigma/128*N)
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


def q_circle(N, center_x = 0.5, center_y = 0.5, radius = 0.3, sigma = 1):
    q = np.zeros((N+1, N+1))
    x, y = np.meshgrid(np.linspace(0, 1, N+1), np.linspace(0, 1, N+1))
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    q[dist <= radius] = 1
    q = gaussian_filter(q, sigma/128*N)
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


def q_test(N):
    # q1 = np.zeros((N+1, N+1))
    # q2 = np.zeros((N+1, N+1))
    # random.seed(0)
    # for i in range(5):
    #     x_axis = random.uniform(0.2,0.8)
    #     y_axis = random.uniform(0.2,0.8)
    #     R1 = random.uniform(150,250)
    #     R2 = random.uniform(150,250)
    #     gamma = random.uniform(-1,1)
    #     q1 += q_Gaussian(N, x_axis, y_axis, R1, R2, gamma)
    # for i in range(5):
    #     x_axis = random.uniform(0.2,0.8)
    #     y_axis = random.uniform(0.2,0.8)
    #     R1 = random.uniform(150,250)
    #     R2 = random.uniform(150,250)
    #     gamma = random.uniform(-1,1)
    #     q2 += q_Gaussian(N, x_axis, y_axis, R1, R2, gamma)
    # return 2*q_T(N,sigma = 1)+0.8*q_circle(N, radius = 0.3) - 0.5*q_circle(N, radius = 0.2) + q2
    # return 2*q_T(N,sigma = 2) - q_T(N,2,-3,0.3,0.6,0.7,0.45,0.55,0.7,0.82,3)+0.8*q_circle(N, radius = 0.3) - 0.5*q_circle(N, radius = 0.2) + q2
    # return 2*q_T(N,sigma = 1) - q_T(N,2,-3,0.3,0.6,0.7,0.45,0.55,0.7,0.82,2)
    # q = np.zeros((N+1, N+1))
    # q += q_Square(N,1,0.2,0.35,0.65,0.8,1)
    # q += q_Square(N,-2,0.55,0.8,0.6,0.85,2)
    # q += q_Square(N,3,0.65,0.8,0.2,0.35,3)
    # q += q_Square(N,-4,0.25,0.3,0.1,0.45,4)
    # q += - 1.5*q_circle(N, radius = 0.04)
    # q +=  5*q_circle(N, radius = 0.1)
    # return q
    q = np.zeros((N+1, N+1))
    q += q_T(N,2,3,0.2,0.3,0.45,0.55,0.65,0.7,0.85,3)
    q -= q_T(N,4,5,0.6,0.75,0.8,0.15,0.2,0.3,0.35,3)[::-1,:]
    q += q_Square(N,3,0.65,0.8,0.2,0.35,3)
    q -= q_Square(N,4,0.25,0.3,0.1,0.45,4)[::-1,::-1]
    return q
        
    # return q_Gaussian(N)
    # return q_T(N, sigma = 5)
    # return q_circle(N, radius = 0.3)
    # q = np.zeros((N+1,N+1))
    # q += q_Gaussian(N, 0.3, 0.6, 150, 70, 1)
    # q -= q_Gaussian(N, 0.5, 0.5, 20, 30, 0.8)
    # q += q_Gaussian(N, 0.7, 0.5, 40, 90, 0.6)
    # q += q_circle(N, radius = 0.15, sigma = 5)
    # return q
    # return q_Continuous(N) + q_Gaussian(N, 0.3, 0.6, 150, 70, 1)
    
    
    
def q_gen(N, method = 'T', gamma = 1):
    """
    Returns.shape = (N+1,N+1)
    """
    q = np.zeros((N+1,N+1))
    if method == 'T':
        q = q_T(N, sigma = 0)
    elif method == 'T0':
        q = q_T(N, 1, 1, sigma = 2)
    elif method == 'G':
        q = q_Gaussian(N)
    elif method == 'C':
        q = q_Continuous(N)
    elif method == 'CIRCLE':
        q = q_circle(N)
    elif method == 'MG':
        q += q_Gaussian(N, 0.3, 0.6, 150, 70, 1)
        q -= q_Gaussian(N, 0.5, 0.3, 120, 80, 0.8)
        q += q_Gaussian(N, 0.8, 0.5, 40, 90, 0.3)
    elif method == 'TEST':
        q = q_test(N)
    Max_Value = np.max(np.abs(q))
    q =  gamma * q / Max_Value
    
    return q


def f_gen(N,k,m):
    res = np.zeros((m,(N+1),(N+1)),dtype = np.complex128)
    tmp = np.linspace(0,1,N+1)
    Y,X = np.meshgrid(tmp, tmp)    
    for j in range(m):    
        res[j] = np.exp(1j*k*(X*np.cos(2*np.pi*j/m)+Y*np.sin(2*np.pi*j/m)))
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