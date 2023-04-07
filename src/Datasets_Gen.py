from Solver import *
from q_method import *
from utils import default_timer


class Datasets:
    def __init__(self, nsample, comp_grid = 128, k = 2, qmethod = 'G', times = 2, 
                 maxq = 0.1, label = '', R=200,
                 angle_TYPE = 'P', angle_total = 64,
                 angle_for_test = 1, angle_mode = 'first',
                 NS_return = 'T', NS_length = 5):
        '''
        angle_TYPE = 'P' : Plane wave with frequency k
        angle_TYPE = 'O' : One
        '''
        self.nsample = nsample
        self.comp_grid = comp_grid
        self.k = k
        self.times = times
        self.qmethod = qmethod
        self.maxq = maxq
        self.label = label
        self.R = R
        self.T = dict(value1 = 0.1, value2 = -0.1,
                      direct = 0, left = 0.2, right = 0.8)
        self.GAUSS = dict(num = 6, left = 0.1, right = 0.9, R = R)
        self.ANGLE = dict(TYPE = angle_TYPE, total = angle_total, mode = angle_mode)
        if self.ANGLE['TYPE'] == 'P':
            self.ANGLE['ntest'] = angle_for_test
        elif self.ANGLE['TYPE'] == 'O':
            self.ANGLE['ntest'] = 1
        else:
            self.ANGLE['ntest'] = 0
        self.NS = dict(_return = NS_return, length = NS_length)
        
    def q_data_gen(self):
        ans = np.zeros((self.nsample, 1, self.comp_grid + 1, self.comp_grid + 1))
        if self.qmethod == 'T':
            for i in range(self.nsample):
                Q_tmp = generate_t_shape(self.comp_grid, self.T['value1'], self.T['value2'],
                                         self.T['direct'], self.T['left'], self.T['right'])
                ans[i,0] = self.maxq * Q_tmp / (abs(Q_tmp).max())
        elif self.qmethod == 'G':
            for i in range(self.nsample):
                Q_total = generate_gauss_shape(self.comp_grid, self.GAUSS['R'], self.GAUSS['num'],
                                               self.GAUSS['left'], self.GAUSS['right'])
                ans[i,0] = self.maxq * Q_total / (abs(Q_total).max())
        return ans
    
    def wave_gen(self, return_mode = 'comp'):
        if self.ANGLE['TYPE'] == 'P':
            if self.ANGLE['mode']=='first':
                order_list = list(range(self.ANGLE['ntest']))
            elif self.ANGLE['mode'] == 'uniform':
                angle_interval =  self.ANGLE['total'] // self.ANGLE['ntest']
                order_list = [i * angle_interval for i in range(self.ANGLE['ntest'])]
        elif self.ANGLE['TYPE'] == 'O':
            order_list = [0]  # len(order_list) = self.ANGLE['ntest']
        if return_mode == 'comp':
            if self.ANGLE['TYPE'] == 'P':
                return [generate_phi_incident(self.comp_grid, self.k, order,
                    self.ANGLE['total']).reshape(-1,) for order in order_list]
                
            elif self.ANGLE['TYPE'] == 'O':
                return [1]
        elif return_mode == 'save':
            if self.ANGLE['TYPE'] == 'P':
                wave = np.zeros((self.ANGLE['ntest'], self.nsample, 2,
                                 self.comp_grid + 1, self.comp_grid + 1))
                for order_id in range(self.ANGLE['ntest']):
                    wave[order_id] = generate_phi_incident_repeat(self.nsample,self.comp_grid,
                                            self.k, order_list[order_id], self.ANGLE['total'])
            else:
                wave = np.ones((self.ANGLE['ntest'], self.nsample, 1,
                self.comp_grid + 1, self.comp_grid + 1))
            return wave
        else:
            pritn('return_mode Error')

    def u_data_gen(self, q_data, sol_type):
        ctx = DMumpsContext()
        Matrix_analysis(self.comp_grid)
        if sol_type == 'init':
            Matrix_factorize(self.comp_grid, self.k)
        sol = np.zeros((self.ANGLE['ntest'], self.nsample, 2, self.comp_grid + 1, self.comp_grid + 1))
        WAVE_list = self.wave_gen('comp')
        for i in range(self.nsample):
            Q = q_data[i].reshape(-1,)
            if sol_type == 'total':
                Matrix_factorize(self.comp_grid, self.k, Q)
            for order in range(self.ANGLE['ntest']):
                print('{}     angle{}     n{}'.format(sol_type, order, i))
                tmp = Matrix_solve(-self.k * self.k * WAVE_list[order] * Q).reshape(self.comp_grid + 1, self.comp_grid + 1)
                sol[order,i] = np.stack((tmp.real, tmp.imag), 0)  
        ctx.destroy()
        return sol
    

    def NS_data_gen(self, q, u_i):
        if self.NS['_return'] == 'T':
            ctx = DMumpsContext()
            ctx.set_silent()
            Q = q.reshape(self.nsample,-1)
            Matrix_analysis(self.comp_grid)
            Matrix_factorize(self.comp_grid, self.k)
            sol = np.zeros((self.ANGLE['ntest'], self.nsample, self.NS['length'],
                            2, self.comp_grid + 1, self.comp_grid + 1))
            for order_id in range(self.ANGLE['ntest']):
                u_now = u_i[order_id].copy() # (nsample, 2, 65, 65)
                for j in range(self.NS['length']):
                    for i in range(self.nsample):
                        f_in = (u_now[i,0]+1j*u_now[i,1]).reshape(-1,)
                        tmp = Matrix_solve(-self.k * self.k * Q[i] * f_in)
                        tmp = tmp.reshape(self.comp_grid + 1,self.comp_grid + 1)
                        u_now[i] = np.stack((tmp.real,tmp.imag),0)
                        print('NS     angle{}     length{}     n{}'.format(order_id, j, i))
                    sol[order_id, :, j] = u_now
            ctx.destroy()
            return sol
        else:
            return None
                
    def dataset_gen(self):
        q = self.q_data_gen()
        WAVE = self.wave_gen('save')
        u_i = self.u_data_gen(q,'init')
        t = default_timer()
        u_t = self.u_data_gen(q,'total')
        t1 = default_timer()
        u_NS = self.NS_data_gen(q, u_i)
        q = q[...,::self.times,::self.times]
        WAVE = WAVE[...,::self.times,::self.times]
        u_i = u_i[...,::self.times,::self.times]
        u_t = u_t[...,::self.times,::self.times]
        if u_NS is not None:
            u_NS = u_NS[...,::self.times,::self.times]
        print(t1-t)
        return q, WAVE, u_i, u_t, u_NS
    
    def save_data(self):
        q, WAVE, u_i, u_t, u_NS = self.dataset_gen()
        Dataset_dir = '/data/liuziyang/Programs/pde_solver/Dataset/'
        filename = Dataset_dir + 'k{}_{}_{}_{},{},{}_{}_{}_NS{}_{}.npz'.format(self.k,self.nsample,
                        self.ANGLE['TYPE'], self.ANGLE['ntest'], self.ANGLE['total'], self.ANGLE['mode'],
                        self.qmethod, self.maxq, self.NS['_return'], self.label)
        np.savez(filename, q = q, WAVE = WAVE, u_i = u_i, u_t = u_t, u_NS = u_NS,
                 nsample = self.nsample, comp_grid = self.comp_grid, k = self.k, times = self.times,
                 qmethod = self.qmethod ,maxq = self.maxq, label = self.label,
                 T = self.T, GAUSS = self.GAUSS, ANGLE = self.ANGLE, NS = self.NS)
        try:
            ctx.destroy()
        except:
            pass
        print('completed')
        
if __name__ == '__main__':                           
    nsample = 32
    k = 2
    qmethod = 'G'
    R = 200
    label = 'R200'
    angle_TYPE = 'P'
    angle_for_test = 4
    angle_mode = 'uniform'
    maxq = 0.1
    comp_grid = 128
    times = 2
    Data = Datasets(nsample, k = k, qmethod = qmethod, label = label, R = R,
                    maxq = maxq, comp_grid = comp_grid, times = times, 
                    angle_TYPE = angle_TYPE, angle_for_test = angle_for_test,
                    angle_mode = angle_mode)
    Data.save_data()
    