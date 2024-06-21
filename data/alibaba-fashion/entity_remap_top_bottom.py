import os
import pickle
import pdb
import numpy as np

#kg_final_top_bottom.txt
root_path = '/home/zhantiantian/work/AAAI2022/KGIN/data/alibaba-fashion'
kg_top_bottom_path = os.path.join(root_path,'kg_final_top_bottom.txt')
can_kg_np = np.loadtxt(kg_top_bottom_path, dtype=np.int32)
count_temp_line = 0
head_list_top_bottom = []
tail_list_top_bottom = []
relation_list_top_bottom = []

for temp_kg in can_kg_np:
	count_temp_line += 1
	#print(temp_kg)
	print('processing {}/{}\n'.format(count_temp_line,can_kg_np.shape[0]))
	temp_head = temp_kg[0]
	temp_tail = temp_kg[2]
	tail_relation = temp_kg[1]
	head_list_top_bottom.append(temp_head)
	tail_list_top_bottom.append(temp_tail)
	relation_list_top_bottom.append(tail_relation)
head_tail_list_top_bottom = head_list_top_bottom + tail_list_top_bottom
head_tail_array = np.array(head_tail_list_top_bottom)
head_tail_array_u = np.unique(head_tail_array,axis=0)
n_entity = len(head_tail_array_u)
max_entity = max(head_tail_array_u)
entity_mapping_array = np.zeros((max_entity+1),dtype=np.int)
head_tail_array_u_list = list(head_tail_array_u)
for indx, temp_entity in enumerate(head_tail_array_u_list):
	entity_mapping_array[temp_entity] = indx

zero_pad_list = []
for i in range(1, max_entity+1):
	if entity_mapping_array[i]==0:
		zero_pad_list.append(i)

#relation
relation_array = np.array(relation_list_top_bottom)
relation_array_u = np.unique(relation_array,axis=0)
n_relation = len(relation_array_u)
max_relation = max(relation_array_u)
relation_mapping_array = np.zeros((max_relation+1),dtype=np.int)
relation_array_u_list = list(relation_array_u)
for indx, temp_relation in enumerate(relation_array_u_list):
	relation_mapping_array[temp_relation] = indx

#user transaction: train.txt, test.txt
'''
train_path = os.path.join(root_path,'train.txt')
fid_train = open(train_path,'r')
lines_train = fid_train.readlines()
train_remap_path = os.path.join(root_path,'train_remap.txt')
fid_train_remap = open(train_remap_path,'w')
for line_id, line in enumerate(lines_train):
	print('train: processing {}/{}\n'.format(line_id,len(lines_train)))
	user = line.split(' ')[0]
	trans_outfit_list = line.split(' ')[1:]
	for idx, outfit in enumerate(trans_outfit_list):
		trans_outfit_list[idx] = entity_mapping_array[int(outfit)]
	if line_id!=0:
		trans_outfit_list = list(filter(lambda num: num != 0, trans_outfit_list))
	trans_outfit_list.insert(0,int(user))
	write_line = " ".join(str(item) for item in trans_outfit_list)
	fid_train_remap.write(write_line+ '\n')
fid_train.close()
fid_train_remap.close()
'''
#test
#user transaction: test.txt
test_path = os.path.join(root_path,'test.txt')
fid_test = open(test_path,'r')
lines_test = fid_test.readlines()

test_remap_path = os.path.join(root_path,'test_remap.txt')
fid_test_remap = open(test_remap_path,'w')
for line_id, line in enumerate(lines_test):
	print('test: processing {}/{}\n'.format(line_id,len(lines_test)))
	user = line.split(' ')[0]
	test_outfit_list = line.split(' ')[1:]
	for idx, outfit in enumerate(test_outfit_list):
		test_outfit_list[idx] = entity_mapping_array[int(outfit)]
	test_outfit_list = list(filter(lambda num: num != 0, test_outfit_list))
	test_outfit_list.insert(0,int(user))
	write_line = " ".join(str(item) for item in test_outfit_list)
	fid_test_remap.write(write_line+ '\n')
fid_test.close()
fid_test_remap.close()
pdb.set_trace()

kg_top_bottom_remap_path = os.path.join(root_path,'kg_final_top_bottom_remap.txt')
fid_top_bottom_remap = open(kg_top_bottom_remap_path,'w')
for temp_kg in can_kg_np:
	count_temp_line += 1
	#print(temp_kg)
	print('processing {}/{}\n'.format(count_temp_line,can_kg_np.shape[0]))
	temp_head = temp_kg[0]
	temp_tail = temp_kg[2]
	temp_relation = temp_kg[1]
	temp_head_remap = entity_mapping_array[temp_head]
	temp_tail_remap = entity_mapping_array[temp_tail]
	temp_relation_remap = relation_mapping_array[temp_relation]
	write_line = ' '.join([str(temp_head_remap), str(temp_relation_remap), str(temp_tail_remap)])
	fid_top_bottom_remap.write(write_line+'\n')
fid_top_bottom_remap.close()
