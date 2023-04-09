import argparse
import re

parser = argparse.ArgumentParser()

parser.add_argument('--output_filename', type=str, required=True)
parser.add_argument('--save_tmp_filename', type=str, default='.tmp.log')
args = parser.parse_args()

output_filename, save_tmp_filename = args.output_filename, args.save_tmp_filename

J_list = []
Find_Tnf = False
Tnf = 0
for line in open(save_tmp_filename, "r", encoding='UTF-8'):
    search_ = re.findall('f=  .*? ', line)
    
    if Find_Tnf:
        Tnf_list = re.findall(r'\d+', re.sub(r'\s+', ' ', line))
        Tnf = int(Tnf_list[2])
    search_Tnf = re.findall('Tnf  Tnint', line)
    if len(search_Tnf) > 0:
        Find_Tnf = True
    else:
        Find_Tnf = False
    if len(search_) > 0:
        a = search_[0]
        J_list.append(eval(a[4:-5] + 'e' + a[-4:-1]))

t_total = 0.
for line in open(output_filename, "r", encoding='UTF-8'):
    search_t_total = re.findall('t_total=.*?\n', line)
    if len(search_t_total) == 1:
        t_total = eval(search_t_total[0][8:-1])
t_avg_f = t_total / Tnf
fp = open(output_filename, 'a+')
J00 = J_list[0]
print('Tnf={},t_avg_f={}'.format(Tnf, t_avg_f), file = fp)
print('-----------------------------', file = fp)
print('relative_data_error(J):', file=fp)
print([i / J00 for i in J_list], file=fp)
fp.close()
