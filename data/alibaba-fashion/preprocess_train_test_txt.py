import os
#import pdb
import numpy as np 

#check train_remap_user
train_remap_reorg_path = 'train_remap_reorg.txt'
test_remap_reorg_path = 'test_remap_reorg.txt'
fid_train_remap = open(train_remap_reorg_path,'r')
lines_train_remap = fid_train_remap.readlines()

fid_test_remap = open(test_remap_reorg_path,'r')
lines_test_remap = fid_test_remap.readlines()

for line_id, line_c in enumerate(lines_test_remap):
	if len(line_c.strip('\n').split(' '))==1:
		#pdb.set_trace()
	temp_line_id = line_id + 1
	user_id = line_c.split(' ')[0]
	if line_id == int(user_id):
		continue
	else:
		print(line_id,line_c)
		break
#pdb.set_trace()

###################################
train_remap_path = 'train_remap.txt'
test_remap_path = 'test_remap.txt'

fid_train_remap = open(train_remap_path,'r')
lines_train_remap = fid_train_remap.readlines()

fid_test_remap = open(test_remap_path,'r')
lines_test_remap = fid_test_remap.readlines()

train_line_list, test_line_list = [], []
user_list = []
user_list_with_null = []

for line_id, line_c in enumerate(lines_train_remap):
	train_line = line_c
	test_line = lines_test_remap[line_id]
	num_of_outfits_train = train_line.strip('\n').split(' ')
	num_of_outfits_test = test_line.strip('\n').split(' ')
	
	if len(num_of_outfits_train)==1:
		user_list_with_null.append(num_of_outfits_train[0])
		print(train_line)
		continue
	elif len(num_of_outfits_test)==1:
		#print('before processing: train:{} test:{}'.format(train_line,test_line))
		if len(num_of_outfits_train) == 2:
			test_line = test_line.strip('\n') + ' ' + num_of_outfits_train[1]
			test_line = test_line + '\n'
		else:
			num_test =  int(np.floor(len(num_of_outfits_train)/5))+1
			test_outfits = num_of_outfits_train[-num_test:]
			train_line = num_of_outfits_train[:-num_test]
			train_line = ' '.join(train_line)
			train_line = train_line + '\n'
			test_outfits.insert(0,test_line.strip('\n'))
			test_line = ' '.join(test_outfits)
			test_line = test_line + '\n'
		#print('after processing: train:{} test:{}'.format(train_line,test_line))
		#pdb.set_trace()
	else:
		pass
	user_list.append(int(num_of_outfits_train[0]))
	if len(train_line.split(' '))==1:
		#pdb.set_trace()
	if len(test_line.split(' '))==1:
		#pdb.set_trace()
	train_line_list.append(train_line)
	test_line_list.append(test_line)
fid_train_remap.close()
fid_train_remap.close()
print(user_list_with_null)
user_list_u=np.unique(np.array(user_list),axis=0)
n_user = len(user_list_u)
max_user = max(user_list_u)
user_mapping_array = np.zeros((max_user+1),dtype=np.int)
user_list_u_L = list(user_list_u)
for indx, temp_user in enumerate(user_list_u_L):
	user_mapping_array[temp_user] = indx
#pdb.set_trace()


###########write################
fid_train_reorg = open('train_remap_reorg.txt','w')
fid_test_reorg = open('test_remap_reorg.txt','w')
for line in train_line_list:
	#write_line = " ".join(str(item) for item in line)
	user_org = line.split(' ')[0]
	outfit = line.split(' ')[1:]
	user_remap = user_mapping_array[int(user_org)]
	outfit.insert(0,str(user_remap))
	write_line = " ".join(str(item) for item in outfit)
	fid_train_reorg.write(write_line)
fid_train_reorg.close()
for line in test_line_list:
	user_org = line.split(' ')[0]
	outfit = line.split(' ')[1:]
	user_remap = user_mapping_array[int(user_org)]
	outfit.insert(0,str(user_remap))
	write_line = " ".join(str(item) for item in outfit)	
	fid_test_reorg.write(write_line)
fid_test_reorg.close()

