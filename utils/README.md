# README

- `Function: load_data_org_top_bottom`
  - args.which_rule decides load which rule
- `Fix mapping mistake`
  - change `entity_list.txt` to `OutfitRealid_2_entityID_remap.txt`
  load_rules(directory, '11Aug_outfit_id_with_rules_chinese.txt', 'entity_list.txt', triplets) 
  => load_rules(directory, '11Aug_outfit_id_with_rules_chinese.txt', 'OutfitRealid_2_entityID_remap.txt', triplets)
- `Treat loaded rule as 'including' relation(namely 1)
- There are many load_rules functions in data_loader_org.py but they are just for tests and experiments, so only need to refer the newest rule: rule21
- `Load_rules21`
  - `Input`: \
    directory: data directory \
    '11Aug_outfit_id_with_rules_chinese.txt': load_rule file\
    'OutfitRealid_2_entityID_remap.txt': map outfit_orgid to remap_id in kg_final_top_bottom_remap.txt\
    triplets: original tripets
  - `Output`: \
    triplets: new tripets(have been added into rule triplets)
  - Note: when running load_rules21, must set `--inverse_r False` because load_rules21 already contains inverse function. The running command can be:\
  python main_variant_top_bottom.py --dataset alibaba-fashion --dim 64 --lr 0.0001 --sim_regularity 0.0001 --batch_size 1024 --epoch 500 --node_dropout True --node_dropout_rate 0.5 --mess_dropout True --mess_dropout_rate 0.1 --gpu_id 0 --context_hops 3 --k_step 1 --flag_step 20 --which_rule 21 --inverse_r False --description one\
  The `--description parameter` is added to name the result file. You can add some description here and these words will be appear at the end of result file name.
