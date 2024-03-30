
cd ./data_creation_sag

python main_decomposed_answer_generate.py \
--raw_data_path_decomposed \
"your file" \
--ndocs \
3 \
--output_path \
"your output_path" \
--search_engine_type \
bm25_candidates \
--openai_api_key \
"your key" \
--overwrite_output