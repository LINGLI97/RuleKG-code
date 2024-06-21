import os
import pickle	
import numpy as np
import pdb

'''
root_path = '/home/zhantiantian/work/AAAI2022/KGIN/data/alibaba-fashion'
kg_path = os.path.join(root_path,'kg_final.txt')
can_kg_np = np.loadtxt(kg_path, dtype=np.int32)
relation_not_equal0_list = []
relation_list = []
fid_kg_bottom_top = open(os.path.join(root_path,'kg_final_top_bottom.txt'),'w')
for temp_kg in can_kg_np:
	temp_head = temp_kg[0]
	temp_tail = temp_kg[2]
	temp_relation = temp_kg[1]
	if temp_head==250 or temp_tail==250:
		print(temp_kg)
pdb.set_trace()
'''
#extract item_list
'''
POG_PATH = '/home/zhantiantian/work/AAAI2022/POG'
with open(os.path.join(POG_PATH,'top_bottom_category_id_list.pkl'),'rb') as fid_pkl:
	top_bottom_category_dict = pickle.load(fid_pkl)
valid_top_ids = top_bottom_category_dict['top']
valid_bottom_ids = top_bottom_category_dict['bottom']


'''

#relation_root_path = '/home/zhantiantian/work/AAAI2022/KGIN/data/alibaba-fashion'
relation_root_path = '/home/ling014/RuleKG/data/alibaba-fashion'

relation_list_file = os.path.join(relation_root_path,'relation_list.txt')
with open(os.path.join(relation_root_path,'relation_list.txt'),'r') as fid_relation:
	lines_relations = fid_relation.readlines()


category_list = []
for line in lines_relations[1:]:
	temp_category = line.split(' ')[0]
	if len(temp_category.split('-'))==3:
		category_list.append(temp_category.split('-')[2])
'''





'''
valid_top_index = []
for valid_top_id in valid_top_ids:
	valid_top_index.append(category_list.index(valid_top_id))
valid_bottom_index = []
for valid_bottom_id in valid_bottom_ids:
	valid_bottom_index.append(category_list.index(valid_bottom_id))
valid_top_index = [gg+1 for gg in valid_top_index]
valid_bottom_index = [gg+1 for gg in valid_bottom_index]
pdb.set_trace()

with open(os.path.join(POG_PATH,'item_data.txt'),'r') as fid_item:
	lines_item = fid_item.readlines()
item_id_list = []
#item_data.txt
#id1,id2;id1 is valid id.id2 is not valid.
for line in lines_item:
	#item_id_list.append(line.split(',')[0])
	item_id_list.append(line.split(',')[0])
	item_id_list.append(line.split(',')[1])
##write to kg_final_top_bottom.txt
with open(os.path.join(POG_PATH,'top_bottom_category_id_list.pkl'),'rb') as fid_pkl:
	top_bottom_category_dict = pickle.load(fid_pkl)

#read entity_list.txt
root_path = '/home/zhantiantian/work/AAAI2022/KGIN/data/alibaba-fashion'
fid_entity = open(os.path.join(root_path,'entity_list.txt'),'r')
lines = fid_entity.readlines()
entity_list = []
for line in lines[1:]:
	sku_id = line.split(' ')[0]
	entity_id = line.split(' ')[1]
	entity_list.append(sku_id)


#write to kg_final_top_bottom.txt
root_path = '/home/zhantiantian/work/AAAI2022/KGIN/data/alibaba-fashion'
kg_path = os.path.join(root_path,'kg_final.txt')
can_kg_np = np.loadtxt(kg_path, dtype=np.int32)
relation_not_equal0_list = []
relation_list = []
fid_kg_bottom_top = open(os.path.join(root_path,'kg_final_top_bottom.txt'),'w')
count_temp_line = 0
for temp_kg in can_kg_np:
	count_temp_line += 1
	print('processing {}/{}\n'.format(count_temp_line,can_kg_np.shape[0]))
	temp_head = temp_kg[0]
	temp_tail = temp_kg[2]
	temp_relation = temp_kg[1]
	if temp_relation==0:
		temp_tail_item_id = entity_list[temp_tail]
		tail_item_id = item_id_list.index(temp_tail_item_id)
		temp_item_line = lines_item[int(tail_item_id/2)]
		tail_category = temp_item_line.split(',')[1]
		if tail_category in top_bottom_category_dict['top'] or tail_category in top_bottom_category_dict['bottom']:
			write_line = str(temp_head) + ' ' + str(temp_relation) + ' ' + str(temp_tail)
			fid_kg_bottom_top.write(write_line+'\n')
	else:
		temp_head_item_id = entity_list[temp_head]
		head_item_id = item_id_list.index(temp_head_item_id)
		temp_item_line = lines_item[int(head_item_id/2)]
		head_category = temp_item_line.split(',')[1]
		if head_category in top_bottom_category_dict['top'] or head_category in top_bottom_category_dict['bottom']:
			write_line = str(temp_head) + ' ' + str(temp_relation) + ' ' + str(temp_tail)
			fid_kg_bottom_top.write(write_line+'\n')
fid_kg_bottom_top.close()
print('finished\n')
pdb.set_trace()
