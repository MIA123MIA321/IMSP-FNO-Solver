import re
import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--logname', type=str, default='')
parser.add_argument('--argname', type=str, default='')
parser.add_argument('--savepath', type=str, default='pic/res/')
parser.add_argument('--q_method', type=str, default='1')
args = parser.parse_args()

logname, argname, savepath, q_method = args.logname, args.argname, args.savepath, args.q_method

search_float_0,search_float_1, search_float_2 = argname + '=.*?]\n',argname + '=.*?,', argname + '=.*?\n'
label_list = []
model_error_list = []
data_error_list = []
Find_model, Find_data = False, False
for line in open(logname, "r", encoding='UTF-8'):
    search_0, search_1, search_2 = re.findall(
        search_float_0, line),re.findall(
        search_float_1, line), re.findall(
        search_float_2, line)
    if argname == 'k':
        if len(search_0) == 1:
            label_list.append(search_0[0][:-1])
        elif len(search_1) == 1:
            label_list.append(search_1[0][:-1])
        elif len(search_2) == 1:
            label_list.append(search_2[0][:-1])
    else:
        if len(search_1) == 1:
            label_list.append(search_1[0][:-1])
        elif len(search_2) == 1:
            label_list.append(search_2[0][:-1])
    if Find_model:
        tmp_list = line[1:-2].split(',')
        tmp_list_1 = [eval(item) for item in tmp_list]
        model_error_list.append(tmp_list_1)
    if Find_data:
        tmp_list = line[1:-2].split(',')
        tmp_list_1 = [eval(item) for item in tmp_list]
        data_error_list.append(tmp_list_1)
    Find_data = (re.findall('data_error', line) != [])
    Find_model = (re.findall('model_error', line) != [])

Solver_List, Linestyle_List = [], []
search_float_Solver = 'Solver=.*?\n'
for line in open(logname, "r", encoding='UTF-8'):
    search_Solver = re.findall(search_float_Solver, line)
    if len(search_Solver) == 1:
        Solver_List.append(search_Solver[0][7:-1])
for i in range(len(label_list)):
    label_list[i] += Solver_List[i]
    if Solver_List[i] == 'MUMPS':
        Linestyle_List.append('-')
    else:
        Linestyle_List.append('--')
    
color_list = ['r', 'r', 'g', 'g', 'b', 'b', 'c', 'c', 'm', 'm',
              'y', 'y', 'k', 'k', 'pink', 'pink', 'black', 'black']

plt.close()
plt.xlabel('Iter')
plt.title('Model-Rel-Err')

for i in range(len(label_list)):
    plt.plot(list(range(len(model_error_list[i]))),
             model_error_list[i],
             color=color_list[i],
             label=label_list[i],
             linestyle=Linestyle_List[i])
plt.legend()
plt.savefig(savepath + argname + '_'+q_method+'_1.jpg')
plt.close()
plt.xlabel('Iter')
plt.ylabel('10^-')
plt.title('Data-Rel-Err')
for i in range(len(label_list)):
    plt.plot(list(range(len(data_error_list[i]))),
             np.log10(data_error_list[i]),
             color=color_list[i],
             label=label_list[i],
             linestyle=Linestyle_List[i])
plt.legend()
plt.savefig(savepath + argname + '_'+q_method+'_2.jpg')
plt.close()
