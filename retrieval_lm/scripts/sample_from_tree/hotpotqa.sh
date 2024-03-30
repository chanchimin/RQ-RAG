cd ./retrieval_lm/output

python sample_from_tree.py \
--run_name \
"your result dir containing final_results.json" \
--task \
hotpotqa \
--original_data \
"your original data containing ground truths" \
--model_name_or_path \
"your model" \
--calc_depth \
1 \
2 \
3 \
--calc_width \
S_Rewritten_Query \
S_Decomposed_Query \
S_Disambiguated_Query \
A_Response \
--calc_retrieval_performance
