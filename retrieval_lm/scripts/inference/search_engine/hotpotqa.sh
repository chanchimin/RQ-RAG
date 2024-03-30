cd retrieval_lm


export PYTHONPATH="$(pwd):$PYTHONPATH"

python ./inference.py \
--model_name_or_path \
"your trained model" \
--input_file \
"your data" \
--max_new_tokens \
100 \
--output_path \
"your output_path" \
--ndocs \
3 \
--use_search_engine \
--use_hf \
--task \
hotpotqa \
--tree_decode \
--oracle \
--max_depth \
2 \
--search_engine_type \
openai_embed \
--expand_on_tokens \
[S_Rewritten_Query] \
[S_Decomposed_Query] \
[S_Disambiguated_Query] \
[A_Response]