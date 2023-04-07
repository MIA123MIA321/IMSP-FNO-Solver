from Inverse import *
from q_method import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--PROJECT_DIR', type = str, default = '/data/liuziyang/Programs/pde_solver/')
parser.add_argument('--N', type = int, default = 64)
parser.add_argument('--k', type = str, default = '2')
parser.add_argument('--m', type=int, default = 16)
parser.add_argument('--maxq', type=float, default = 0.1)
parser.add_argument('--q_method', type=str, default = 'T')
parser.add_argument('--noise_level', type=float, default = 0.0)
parser.add_argument('--gtol', type = float, default = 1e-10)
parser.add_argument('--maxiter', type = int, default = 30)
parser.add_argument('--forward_solver', type=str, default = 'NET')
parser.add_argument('--title', type = str, default = 'tmp')
parser.add_argument('--output_filename', type = str, required=True)
parser.add_argument('--Net_name', type = str,default = '')
parser.add_argument('--NS_length', type = int, default = 3)
args = parser.parse_args()

print('Data loading             %s' % str(datetime.now())[:-7])
PROJECT_DIR = args.PROJECT_DIR
sys.path.append(PROJECT_DIR)
N = args.N
k = args.k
m = args.m
maxq = args.maxq
q_method = args.q_method
noise_level = args.noise_level
gtol = args.gtol
maxiter = args.maxiter
forward_solver = args.forward_solver
title = args.title
output_filename = args.output_filename
Net_name = args.Net_name
NS_length = args.NS_length

jpgdir = PROJECT_DIR + 'pic/process_jpg/'
gifdir = PROJECT_DIR + 'pic/process_gif/'
Netdir = PROJECT_DIR + 'Network/'

if isinstance(k,str):
    tmp_k = k.split(',')
    k = [float(eval(item)) for item in tmp_k]
k_len = len(k)

if forward_solver == 'NET':
    NET = torch.load( Netdir + Net_name + '.pth', map_location = device)
    pic_list = [0, 1, 2, 5, -2, -1]
elif forward_solver == 'MUMPS':
    NET = None
    pic_list = [0, 1, 2, 5, 10, -2, -1]
    
q = q_gen(N, q_method, maxq)
Q = q.reshape(-1, )
Q0 = Q*0
matrix_A = gen_A(N)
Matrix_analysis(N)
f_data = np.zeros((k_len,m,(N+1)**2),dtype = np.complex128)
partial_data = np.zeros((k_len,m,4*N-4),dtype = np.complex128)
for j in range(k_len):
    f_data[j] = f_gen(N, k[j], m)
    partial_data[j] = pdata_gen(N, Q, k[j], f_data[j], matrix_A,noise_level)

if forward_solver == 'NET':
    f_data_torch = torch.stack([torch.from_numpy(f_data.reshape(k_len,m,(N+1),(N+1)).real).to(torch.float32).to(device),
                               torch.from_numpy(f_data.reshape(k_len,m,(N+1),(N+1)).imag).to(torch.float32).to(device)],2)
    partial_data_torch = torch.stack([torch.from_numpy(partial_data.reshape(k_len,m,4*N-4).real).to(torch.float32).to(device),
                               torch.from_numpy(partial_data.reshape(k_len,m,4*N-4).imag).to(torch.float32).to(device)],2)
    matrix_A_torch = torch.from_numpy(matrix_A).to(torch.float32).to(device)
    matrix_A_torch = matrix_A_torch.unsqueeze(0).unsqueeze(1).repeat(m,2,1,1)
else:
    f_data_torch = None
    partial_data_torch = None
    matrix_A_torch = None
f_data = (f_data, f_data_torch)
partial_data = (partial_data,partial_data_torch)
matrix_A = (matrix_A,matrix_A_torch)
args1 = (N, partial_data, k, f_data, matrix_A, maxq, NET, device, NS_length)
print('Data loading completed   %s' % str(datetime.now())[:-7])  

if forward_solver == 'NET':
    ftol = 1e-5 * J_MULTI(Q0,*args1)
else:
    ftol = 1e-10 * J_MULTI(Q0,*args1)
X_list.append(Q0)
t0 = time.time()
RES2 = SOLVE(J_MULTI,Q0=Q0,args=args1,jac=J_MULTIPRIME,
            options={'disp': True,'gtol': gtol,
                     'maxiter': maxiter,'ftol':ftol},
            method='L-BFGS-B')
time_avg = (time.time() - t0) / len(X_list)
ll = len(X_list)
plot_list, label_list, Error_list = [], [], []
for j in range(ll):
    Error_list.append(Error(X_list[j], Q))
    plot_list.append(X_list[j].reshape((N + 1, N + 1)))
    label_list.append('Iter = ' + str(j))
plot_list.append(Q.reshape((N + 1, N + 1)))
label_list.append('Qt')


fp = open(output_filename, 'a+')
print('****************************************************************', file=fp)
print('****************************************************************', file=fp)
print('%s' % str(datetime.now())[:-7], file = fp)
print('Solver={}'.format(forward_solver), file = fp)
print('N={},m={},k={}'.format(N, m, k), file = fp)
print('gtol={},maxiter={}'.format(gtol, maxiter), file = fp)
print('q_method={},maxq={}'.format(q_method, maxq), file = fp)
print('noise_level={}'.format(noise_level), file = fp)
print('total_iter={},t_avg={:.2f}'.format(len(X_list[1:]), time_avg), file = fp)
print('relative_model_error:', file = fp)
print(Error_list, file = fp)
print('J(qt)={}'.format(J_MULTI(Q,*args1)), file = fp)
percent_list = [str(round(Error_list[i]*100,2))+'%' for i in range(len(Error_list))]
percent_list[0] = ''
label_list[0] = 'Init'
percent_list.append('')
fp.close()
plot_heatmap(plot_list, title, jpgdir, gifdir, label_list, percent_list, pic_list)


ctx.destroy()
