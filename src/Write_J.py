import argparse
import re

parser = argparse.ArgumentParser()

parser.add_argument('--output_filename', type=str, required=True)
parser.add_argument('--PROJECT_DIR', type=str, required = True)
args = parser.parse_args()

output_filename, PROJECT_DIR = args.output_filename, args.PROJECT_DIR

J_list = []
Find_Tnf = False
Tnf_list = []
Tit_list = []
tmp_filename = PROJECT_DIR + '.tmp.log'
for line in open(tmp_filename, "r", encoding='UTF-8'):
    search_ = re.findall('f=  .*? ', line)
    if Find_Tnf:
        T_tmp = re.findall(r'\d+', re.sub(r'\s+', ' ', line))
        if len(T_tmp) > 0:
            if int(T_tmp[0])>int(T_tmp[1]):
                Tit_list.append(int(T_tmp[1]))
                Tnf_list.append(int(T_tmp[2]))
            else:
                Tit_list.append(int(T_tmp[0]))
                Tnf_list.append(int(T_tmp[1]))
            Find_Tnf = False
    search_Tnf = re.findall('Tnf  Tnint', line)
    if len(search_Tnf) > 0:
        Find_Tnf = True
    if len(search_) > 0:
        a = search_[0]
        J_list.append(eval(a[4:-5] + 'e' + a[-4:-1]))
start_id = 0
end_id = 0
J_output = []
for i in range(len(Tit_list)):
    end_id =  start_id + Tit_list[i] + 1
    J_tmp = J_list[start_id:end_id]
    J_tmp = [i / J_tmp[0] for i in J_tmp]
    start_id = end_id
    J_output += J_tmp
    
    
for line in open(output_filename, "r", encoding='UTF-8'):
    search_t_total = re.findall('t_total=.*?\n', line)
    if len(search_t_total) == 1:
        t_total_tmp = eval(search_t_total[0][8:-1])
        try:
            t_total_list = list(t_total_tmp)
        except:
            t_total_list = [t_total_tmp]
Tnf_str = ''
Tit_str = ''
t_avg_str = ''
for i in range(len(Tit_list)):
    Tnf_str += str(Tnf_list[i])[:5] + ','
    Tit_str += str(Tit_list[i])[:5] + ','
    t_avg_str += str(t_total_list[i]/Tnf_list[i])[:5] + ','
    
fp = open(output_filename, 'a+')
print('Tit={}  Tnf={}'.format(Tit_str[:-1], Tnf_str[:-1]), file = fp)
print('t_avg_f={}'.format(t_avg_str[:-1]), file = fp)
print('-----------------------------', file = fp)
print('relative_data_error(J):', file=fp)
print(J_output, file=fp)
fp.close()