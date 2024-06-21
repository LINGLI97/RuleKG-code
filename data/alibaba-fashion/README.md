# README

- `11Aug_outfit_id_with_rules_chinese.txt`
  - org_id (1st column) : ruleID list(defined in 11Augdistribution_chinese.txt)
  - Each line is an outfitID with its activated ruleIDs: (`org_id` and `a list of activated rules`).
- `11Augdistribution_chinese.txt`
  - rule summarization file.
  - Each line is a ruleID and the frequency: (`rule(top+bo` and `frequency`).
  - Note that the differences between Ln and ruleID index
- `OutfitRealid_2_entityID_remap.txt`
  - knowlege graph file.
  - Each line is org_id(1st column), KG entity ID(2nd column, refer to kg_final.txt), KG entity ID (3rd column, refer to kg_final_top_bottom.txt) 

- `cate.txt`
 - first column: category id
 - ['Product description',Number]:Product description is one of products under this category, number is the frequency the category appears
