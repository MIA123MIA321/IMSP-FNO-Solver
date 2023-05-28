from Inverse import *
from q_method import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--PROJECT_DIR', type = str, default = '/data/liuziyang/Programs/pde_solver/')
parser.add_argument('--N', type = int, default = 1024)
parser.add_argument('--N_comp', type = str)
parser.add_argument('--k', type = str, default = '2')
parser.add_argument('--m', type=int, default = 16)
parser.add_argument('--maxq', type=float, default = 0.1)
parser.add_argument('--q_method', type=str, default = 'T')
parser.add_argument('--noise_level', type=float, default = 0.0)
parser.add_argument('--gtol', type = float, default = 1e-10)
parser.add_argument('--maxiter', type = str)
parser.add_argument('--forward_solver', type=str, default = 'NET')
parser.add_argument('--title', type = str, default = 'tmp')
parser.add_argument('--output_filename', type = str, required=True)
parser.add_argument('--NS_length', type = int, default = 3)
parser.add_argument('--load_boundary', type = str, default = 'F')
parser.add_argument('--bd_num', type = int, default = 4)
args = parser.parse_args()
print('Boundary data preparing'+'  '+str(datetime.now())[:-7])
PROJECT_DIR = args.PROJECT_DIR
sys.path.append(PROJECT_DIR)
N = args.N
N_comp = args.N_comp
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
NS_length = args.NS_length
load_boundary = args.load_boundary
scheme = 5
expand_times = 2
bd_num = args.bd_num

jpgdir = PROJECT_DIR + 'pic/process_jpg/'
gifdir = PROJECT_DIR + 'pic/process_gif/'
Netdir = PROJECT_DIR + 'Network/'
DATApath = PROJECT_DIR + 'Dataset/tmp_boundary.npz'
suffix = '_P_4,64,uniform_G_0.1_NST_R200_12,32,4_1.pth'
pic_list = [0, 1, 2, 5, -2, -1]
if isinstance(k+'',str):
    tmp_k = k.split(',')
    k_list = [int(eval(item)) for item in tmp_k]
k_len = len(k_list)
N_comp_list = [int(eval(item)) for item in N_comp.split(',')]
forward_solver_list = (forward_solver+'').split(',')
maxiter_list = [int(eval(item)) for item in maxiter.split(',')]
assert len(forward_solver_list) == k_len
assert len(N_comp_list) == k_len

q = q_gen(N , q_method, maxq) # q:(N+1,N+1)
Q = q.reshape(-1, )
assert load_boundary == 'T' or load_boundary == 'F'
if load_boundary == 'T':
    tmp_data = np.load(DATApath)
    f_data_np, partial_data_np = tmp_data['f'], tmp_data['p']
else:    
    N = 512
    qq = q_gen(N , q_method, maxq)
    Q_gen = qq.reshape(-1,)
    f_data_np, partial_data_np = data_gen(Q_gen, N, k_list, m, noise_level, scheme, bd_num, expand_times)
    np.savez(DATApath, f = f_data_np, p = partial_data_np)
# f_data_np:(k_len+1,m,N+1,N+1)
# partial_data_np:(k_len+1,m,4N-4)
f_data_list, partial_data_list = [], []
NET_list = []
for i in range(k_len):
    if forward_solver_list[i] == 'NET':
        NET_list.append(torch.load(Netdir+'k{}'.format(k_list[i])+suffix, map_location = device))
        f_data_list.append(torch.stack([torch.from_numpy(f_data_np[i].real).to(torch.float32).to(device),
                               torch.from_numpy(f_data_np[i].imag).to(torch.float32).to(device)],1))
        partial_data_list.append(torch.stack([torch.from_numpy(partial_data_np[i].real).to(torch.float32).to(device),
                        torch.from_numpy(partial_data_np[i].imag).to(torch.float32).to(device)],1))
    elif forward_solver_list[i] == 'MUMPS':
        NET_list.append(None)
        f_data_list.append(f_data_np[i])
        partial_data_list.append(partial_data_np[i])
    else:
        raise ValueError("Solver Error")
print('Boundary data prepared'+'   '+str(datetime.now())[:-7])
print('********************************************')

Q0 = None
total_time = 0.
J_rel_str = ''
time_total_str = ''
J00_str = ''
Jtt_str = ''
for i in range(k_len):
    if Q0 is None:
        Q0 = Q * 0
    X_list.append(Q0)
    Matrix_analysis(N_comp_list[i],scheme=scheme,expand_times=expand_times)
    args1 = (N, N_comp_list[i], k_list[i], f_data_list[i], partial_data_list[i], maxq, NET_list[i], device, NS_length, scheme, bd_num, expand_times, True)
    args2 = (N, N_comp_list[i], k_list[i], f_data_list[i], partial_data_list[i], maxq, NET_list[i], device, NS_length, scheme, bd_num, expand_times, False)
    J00 = J_single_frequency(Q0, *args2)
    Jtt = J_single_frequency(Q, *args2)
    J00_str += str(J00)[:5]+','
    Jtt_str += str(Jtt)[:5]+','
    print('Process {}'.format(i)+'                '+str(datetime.now())[:-7])
    J_rel_str += str(Jtt/J00)[:5]+','
    ftol = 1e-5 * J00
    t0 = time.time()
    _,Q0 = SOLVE(J_single_frequency,Q0=Q0,args=args1,jac=True,
            options={'disp': True,'gtol': gtol,
                     'maxiter': maxiter_list[i],'ftol':ftol},
            method='L-BFGS-B')
    print('********************************************')
    time_total = time.time() - t0
    time_total_str += str(time_total)[:5]+','

ll = len(X_list)
plot_list, label_list, Error_list = [], [], []
for j in range(ll):
    Error_list.append(Error(X_list[j], Q))
    plot_list.append(X_list[j].reshape(N + 1, N + 1))
    label_list.append('Iter = ' + str(j))
plot_list.append(Q.reshape((N + 1, N + 1)))
label_list.append('Qt')


fp = open(output_filename, 'a+')
print('****************************************************************', file=fp)
print('****************************************************************', file=fp)
print('%s' % str(datetime.now())[:-7], file = fp)
print('Solver={}'.format(forward_solver), file = fp)
print('N={}  N_comp={}  m={}  k={}'.format(N, N_comp, m, k), file = fp)
print('q_method={}  maxq={}  max_iter={}'.format(q_method, maxq, maxiter), file = fp)
print('gtol={}  noise_level={}'.format(gtol, noise_level), file = fp)
print('-----------------------------', file = fp)
print('relative_model_error:', file = fp)
print(Error_list, file = fp)
print('t_total={}'.format(time_total_str[:-1]), file = fp)
print('-----------------------------', file = fp)
print('J(qt)={}'.format(Jtt_str[:-1]), file = fp)
print('J(q0)={}'.format(J00_str[:-1]), file = fp)
print('J(qt)/J(q0)={}'.format(J_rel_str[:-1]), file = fp)
percent_list = [str(round(Error_list[i]*100,2))+'%' for i in range(len(Error_list))]
percent_list[0] = ''
label_list[0] = 'Init'
percent_list.append('')
fp.close()
plot_heatmap(plot_list, title, jpgdir, gifdir, label_list, percent_list, pic_list)


ctx.destroy()
