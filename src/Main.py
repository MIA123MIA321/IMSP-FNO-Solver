from Inverse import *
from q_method import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--PROJECT_DIR', type = str, default = '/data/liuziyang/Programs/pde_solver/')
parser.add_argument('--N_src', type=int, default = 16)
parser.add_argument('--N_comp', type = str)
parser.add_argument('--k', type = str, default = '2')
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
parser.add_argument('--scheme', type = str, default = 'PML')
parser.add_argument('--bd_type', type = str, default = 'N')
args = parser.parse_args()
print('Boundary data preparing'+'  '+str(datetime.now())[:-7])
PROJECT_DIR = args.PROJECT_DIR
sys.path.append(PROJECT_DIR)


N_gen = 512
N_buffer_gen = int(N_gen*14/128)
N_int_gen = N_gen - 2 * N_buffer_gen
N_src = args.N_src # number of incident waves
N_comp = args.N_comp # number of computer grids
N_comp_list = [int(eval(item)) for item in N_comp.split(',')]
N_bm = 512 # number of benchmark grids on which to be compared
N_buffer_bm = int(N_bm * 14 / 128)
N_int_bm = N_bm - 2 * N_buffer_bm
N_buffer_list = [int(item * 14 / 128) for item in N_comp_list]
N_int_list = [item - 2 * int(item * 14 / 128) for item in N_comp_list]

k = args.k
if isinstance(k+'',str):
    tmp_k = k.split(',')
    k_list = [int(eval(item)) for item in tmp_k]
k_len = len(k_list)
assert len(N_comp_list) == k_len

maxq = args.maxq
q_method = args.q_method
scheme = args.scheme
bd_num = args.bd_num
bd_type = args.bd_type
noise_level = args.noise_level

gtol = args.gtol
maxiter = args.maxiter
maxiter_list = [int(eval(item)) for item in maxiter.split(',')]
forward_solver = args.forward_solver
forward_solver_list = (forward_solver+'').split(',')
assert len(forward_solver_list) == k_len
Rec_dict = dict()
Rec_dict[20] = 64
Rec_dict[40] = 128
Rec_dict[80] = 256
N_rec_list = [Rec_dict[k] for k in k_list] # number of receivers on each edge
N_rec_str = ','.join(str(_) for _ in N_rec_list)

title = args.title
output_filename = args.output_filename
NS_length = args.NS_length
load_boundary = args.load_boundary

jpgdir = PROJECT_DIR + 'pic/process_jpg/'
gifdir = PROJECT_DIR + 'pic/process_gif/'
Netdir = PROJECT_DIR + 'Network/'
DATApath = PROJECT_DIR + 'Dataset/tmp_boundary.npz'
suffix = dict()
suffix[20] = '_P_4,64,uniform_G_0.1_NST_R200_12,32,4_1.pth'
suffix[40] = '_P_4,64,uniform_G_0.1_NST_R200_12,32,4_1.pth'
suffix[80] = '_P_8,64,uniform_G_0.1_NST_R200_30,64,4_0.pth'
pic_list = [0, 1, 2, 5, -2, -1]


Q_bm = np.pad(q_gen(N_int_bm , q_method, maxq),N_buffer_bm).reshape(-1,) # the benchmark Q ((N_bm+1)**2,)
Q_gen = np.pad(q_gen(N_int_gen, q_method, maxq),N_buffer_gen).reshape(-1,)
assert load_boundary in ['T', 'F']
if load_boundary == 'T':
    tmp_data = np.load(DATApath)
    f_data_np, partial_data_np = tmp_data['f'], tmp_data['p']
else:
    f_data_np, partial_data_np = data_gen(Q_gen, N_gen, N_src, N_rec_list, N_buffer_gen, bd_num, bd_type, k_list, noise_level, scheme)
    np.savez(DATApath, f = f_data_np, p = partial_data_np)
# f_data_np:(k_len+1,m,N_rec+1,N_rec+1)
# partial_data_np:(k_len+1,m,4N_rec-4)

f_data_list, partial_data_list = [], []
NET_list = []
for i in range(k_len):
    if forward_solver_list[i] == 'NET':
        NET_list.append(torch.load(Netdir+'k{}_{}'.format(k_list[i],scheme)+suffix[k_list[i]], map_location = device))
        f_data_list.append(torch.stack([torch.from_numpy(f_data_np[i].real).to(torch.float32).to(device),
                               torch.from_numpy(f_data_np[i].imag).to(torch.float32).to(device)],1))
        partial_data_tmp = torch.stack([torch.from_numpy(partial_data_np[i].real).to(torch.float32).to(device),
                torch.from_numpy(partial_data_np[i].imag).to(torch.float32).to(device)],1)
        pd_shape = partial_data_tmp.shape
        N_rec_list[i] = N_comp_list[i]
        times = ((pd_shape[-1] + bd_num) // bd_num) // (N_rec_list[i]*50//64)
        partial_data_tmp = partial_data_tmp.reshape((pd_shape[0],pd_shape[1],bd_num,-1))
        index = [times*i + times-1 for i in range((N_rec_list[i]*50//64)-1)]
        partial_data_tmp = partial_data_tmp[...,index].reshape((pd_shape[0],pd_shape[1],-1))
        partial_data_list.append(partial_data_tmp)
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
        Q = INTERPOLATE(Q_gen, N_gen, N_comp_list[i]).reshape(-1,)
    else:
        Q = INTERPOLATE(Q0, N_comp_list[i-1], N_comp_list[i]).reshape(-1,)
    Q0 = Q * 0
    X_list.append(Q0)
    Matrix_analysis(N_comp_list[i],scheme=scheme)
    N_args = (N_gen, N_src, N_rec_list[i], N_comp_list[i], N_buffer_list[i])
    bd_args = (bd_num, bd_type)
    data_args = (k_list[i], f_data_list[i], partial_data_list[i])
    comp_args = (NET_list[i],device,NS_length, scheme)
    args1 = (N_args, bd_args, data_args, comp_args, True)
    args2 = (N_args, bd_args, data_args, comp_args, False)
    J00 = J_single_frequency(Q0, *args2)
    Jtt = J_single_frequency(Q, *args2)
    J00_str += str(J00)+','
    Jtt_str += str(Jtt)+','
    print('Process {}'.format(i)+'                '+str(datetime.now())[:-7])
    J_rel_str += str(Jtt/J00)+','
    ftol = 1e-5 * J00
    t0 = time.time()
    _,Q0 = SOLVE(J_single_frequency,Q0=Q0,args=args1,jac=True,
            options={'disp': True,'gtol': gtol,
                     'maxiter': maxiter_list[i],'ftol':ftol},
            method='L-BFGS-B')
    print('********************************************')
    time_total = time.time() - t0
    time_total_str += str(time_total)[:5]+','

def trans_to_bm(Q):
    N_tmp = int(np.sqrt(Q.shape[0])) - 1
    Q_RES = INTERPOLATE(Q.reshape(N_tmp + 1, N_tmp + 1), N_tmp, N_bm)
    return Q_RES[0][N_buffer_bm:-N_buffer_bm,N_buffer_bm:-N_buffer_bm]

ll = len(X_list)
plot_list, label_list, Error_list = [], [], []
for j in range(ll):
    Error_list.append(Error(trans_to_bm(X_list[j]), trans_to_bm(Q_bm)))
    plot_list.append(trans_to_bm(X_list[j]))
    label_list.append('Iter = ' + str(j))
plot_list.append(trans_to_bm(Q_bm))
label_list.append('Qt')


fp = open(output_filename, 'a+')
print('****************************************************************', file=fp)
print('****************************************************************', file=fp)
print('%s' % str(datetime.now())[:-7], file = fp)
print('Solver={}'.format(forward_solver), file = fp)
print('N_gen={} N_bm={} N_src={} k={}'.format(N_gen, N_bm, N_src, k), file = fp)
print('N_comp={} N_rec={}'.format(N_comp, N_rec_str), file = fp)
print('q_method={}  maxq={}'.format(q_method, maxq), file = fp)
print('bd_num={}  bd_type={} noise_level={}'.format(bd_num, bd_type, noise_level), file = fp)
print('scheme={} maxiter={} gtol={}'.format(scheme, maxiter, gtol), file = fp)
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
