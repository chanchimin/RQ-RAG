
cd ./data_creation_sag

python main_multiturn_answer_generate.py \
--raw_data_path_multiturn  \
"your file" \
--ndocs  \
3  \
--output_path  \
"your output_path" \
--search_engine_type  \
duckduckgo  \
--openai_api_key  \
"your key" \
--overwrite_output