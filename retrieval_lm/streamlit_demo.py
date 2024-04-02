from retrieval_lm.streamlit_utils.inference_func import init_model_and_tokenizer_and_tool, generate_and_retrieve
from utils import load_sag_special_tokens
import streamlit as st
from transformers import LogitsProcessor, LogitsProcessorList
import fasttext

MODEL_PATH = "your model path"

st.set_page_config(page_title="RQ-RAG Demo", page_icon="📡")
st.title("📡 RQ-RAG Demo")


class SpecialTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, special_tokens_ids, enable_search):
        self.tokenizer = tokenizer
        self.special_tokens_ids = special_tokens_ids
        self.enable_search = enable_search
        self.first_token = True # we only want to scale the special token on the first token

    def __call__(self, input_ids, scores):

        if self.enable_search and self.first_token:
            for token_id in self.special_tokens_ids:
                logits = scores[:, token_id]
                logits *= 1000
                scores[:, token_id] = logits
            self.first_token = False
        else:
            for token_id in self.special_tokens_ids:
                logits = scores[:, token_id]
                logits *= 0
                scores[:, token_id] = logits

        return scores


def concat_msgs(st_msgs, lang):

    return_str = ""

    if lang == "en":
        return_str += "<|system|>\n You are a helpful assistant."

    for msg in st_msgs:
        if msg["role"] == "user":
            return_str += "<|user|>\n"
            return_str += msg["content"]
            return_str += "</s>\n"

        elif msg["role"] == "assistant":
            return_str += "<|assistant|>\n"
            return_str += msg["content"].replace("</s>", "")
            return_str += "</s>\n"

    # st_msgs will end with user
    return_str += f"<|assistant|>\n"

    return return_str


if "initial" not in st.session_state:
    st.session_state.initial = True
    st.session_state.msgs = []
    langdetect_model = fasttext.load_model('lid.176.bin')
    model, tokenizer, search_engine_api = init_model_and_tokenizer_and_tool(
        model_name_or_path=MODEL_PATH)
    special_tokens_dict = load_sag_special_tokens(tokenizer)
    st.session_state.model = model
    st.session_state.langdetect_model = langdetect_model
    st.session_state.tokenizer = tokenizer
    st.session_state.search_engine_api = search_engine_api
    st.session_state.special_tokens_dict = special_tokens_dict


special_tokens = ["[S_Rewritten_Query]", "[S_Decomposed_Query]", "[S_Disambiguated_Query]", ] # do not made [EOS] to zero, or it won't stop, and do not scale [A_Response]
disable_special_tokens = ["[S_Rewritten_Query]", "[S_Decomposed_Query]", "[S_Disambiguated_Query]", "[A_Response]"]
special_tokens_ids = [st.session_state.tokenizer.convert_tokens_to_ids(special) for special in special_tokens]
disable_special_tokens_ids = [st.session_state.tokenizer.convert_tokens_to_ids(special) for special in disable_special_tokens]

with st.sidebar:
    if st.button("Reset"):
        st.session_state.msgs = []

    option = st.selectbox(
        'Which mode do you want to use',
        ('disable search', 'adaptive search', 'force search'))

    st.write('You selected:', option)

    if option == "disable search":
        threshold_processor = SpecialTokenLogitsProcessor(st.session_state.tokenizer, disable_special_tokens_ids, enable_search=False)
        st.session_state.logits_processor = LogitsProcessorList([threshold_processor])

    elif option == "adaptive search":
        # do not need to search
        st.session_state.logits_processor = LogitsProcessorList([])

    elif option == "force search":
        threshold_processor = SpecialTokenLogitsProcessor(st.session_state.tokenizer, special_tokens_ids, enable_search=True)
        st.session_state.logits_processor = LogitsProcessorList([threshold_processor])

st.chat_message("assistant").write("How can I help you?")

# show cur_session results
for idx, msg in enumerate(st.session_state.msgs):

    with st.chat_message(msg["role"]):
        # Render intermediate steps if any were saved
        if msg["role"] == "assistant":
            for step in msg["search_results"]:
                with st.status(label="done") as cur_status:
                    cur_status.markdown(step)
        st.markdown(msg["display_results"])

if prompt := st.chat_input(placeholder="Hi, how are you today?"):
    st.chat_message("user").write(prompt)

    cur_lang = st.session_state.langdetect_model.predict(prompt.replace("\n", ""))[0][0] # "__label__en"

    display_prompt = prompt
    real_prompt = prompt

    st.session_state.msgs.append({
        "role": "user",
        "content": real_prompt,
        "display_results": display_prompt
    })

    with st.chat_message("assistant"):
        st.container()

        generator = generate_and_retrieve(
            examples=[concat_msgs(st.session_state.msgs, lang="en")],
            model=st.session_state.model,
            tokenizer=st.session_state.tokenizer,
            special_tokens_dict=st.session_state.special_tokens_dict,
            search_engine_api=st.session_state.search_engine_api,
            search_limit=2,
            logits_processor=st.session_state.logits_processor)

        # TODO if we want to change the logic, that determine which status is now (searching or answering),
        #  we may have to change the yield keyword in generator.

        search_results = []
        while True:
            with st.status(label="Answering..." if option == "disable search" else "Searching...") as cur_status:

                try:
                    result_dict = next(generator)
                    if "search_query" in result_dict:
                        cur_status.update(label="done")
                        cur_status.markdown(f"**Search Query:{result_dict['search_query'][0]}**")
                        newline = "\n"
                        cur_status.markdown(f"Search Results:  \n{result_dict['evidence_list'][0].replace(newline, '  '+newline)}")
                        search_results.append(f"**Search Query:{result_dict['search_query'][0]}**  \nSearch Results:  \n{result_dict['evidence_list'][0].replace(newline, '  '+newline)}")

                except StopIteration as e:
                    result_dict = e.value
                    cur_status.update(label="done")
                    cur_status.markdown(result_dict["final_response"][0])
                    break

        st.session_state.msgs.append({"role": "assistant",
                                      "content": result_dict["cur_examples"][0].split("<|assistant|>\n")[-1],
                                      "search_results": search_results,
                                      "display_results": result_dict["final_response"][0],  # previous output from model
                                      })

        # above is written in a st.expanded
        st.markdown(result_dict["final_response"][0])