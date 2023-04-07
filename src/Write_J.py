import argparse
import re

parser = argparse.ArgumentParser()

parser.add_argument('--output_filename', type=str, required=True)
parser.add_argument('--save_tmp_filename', type=str, default='.tmp.log')
args = parser.parse_args()

output_filename, save_tmp_filename = args.output_filename, args.save_tmp_filename

J_list = []
for line in open(save_tmp_filename, "r", encoding='UTF-8'):
    search_ = re.findall('f=  .*? ', line)
    if len(search_) > 0:
        a = search_[0]
        J_list.append(eval(a[4:-5] + 'e' + a[-4:-1]))


fp = open(output_filename, 'a+')
J00 = J_list[0]
print('J(q0)={}'.format(J00), file=fp)
print('relative_data_error(J):', file=fp)
print([i / J00 for i in J_list], file=fp)
fp.close()
