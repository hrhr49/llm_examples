import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    # AIMessage
)
from langchain_community.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from dotenv import load_dotenv
load_dotenv()

def init_page():
    if st.session_state.get('page') != __file__:
        # ページ遷移時に状態を初期化しておく
        st.session_state.clear()
        st.session_state.page = __file__
    st.set_page_config(
        page_title="Website Summarizer",
        page_icon="🤗"
    )
    st.header("Website Summarizer 🤗")
    st.sidebar.title("Options")


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]
        st.session_state.costs = []


def select_model():
    model_list = ["gpt-4o-mini"]#, "gpt-3.5-turbo", "gpt-4"]
    model = st.sidebar.radio("Choose a model:", model_list)
    temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
    # ChatOpenAI(temperature=temperature, model=model, streaming=True, stream_usage=True)
    map_prompt_template = """¥
以下はとある。Webページのコンテンツである。内容をわかりやすく日本語で要約してください。

------
{text}
------
"""

    map_combine_template="""¥
以下はとある。Webページのコンテンツである。内容をわかりやすく日本語で要約してください。
------
{text}
------
"""

    map_first_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
    map_combine_prompt = PromptTemplate(template=map_combine_template, input_variables=["text"])

    map_chain = load_summarize_chain(
        llm=ChatOpenAI(model=model, temperature=temperature),
        reduce_llm=ChatOpenAI(model=model, temperature=temperature),
        collapse_llm=ChatOpenAI(model=model, temperature=temperature),
        chain_type="map_reduce",
        map_prompt=map_first_prompt,
        combine_prompt=map_combine_prompt,
        collapse_prompt=map_combine_prompt,
        token_max=4000,
        verbose=True)
    return map_chain


def get_url_input():
    url = st.text_input("URL: ", key="input")
    return url


def validate_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def get_content(url):
    try:
        with st.spinner("Fetching Content ..."):
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            # fetch text from main (change the below code to filter page)
            if soup.main:
                return soup.main.get_text()
            elif soup.article:
                return soup.article.get_text()
            else:
                return soup.body.get_text()
    except:
        st.write('something wrong')
        return None


def build_prompt(content, n_chars=300):
    return f"""以下はとある。Webページのコンテンツである。内容を{n_chars}程度でわかりやすく要約してください。

========

{content[:1000]}

========

日本語で書いてね！
"""


def get_answer(map_chain, text):
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1500, chunk_overlap=0 , separator=".", 
    )
    texts = text_splitter.split_text(text)

    docs = [Document(page_content=t) for t in texts]

    with get_openai_callback() as cb:
        result=map_chain.invoke({"input_documents": docs}, return_only_outputs=True)
    return result["output_text"], cb.total_cost


def main():
    init_page()

    map_chain = select_model()
    init_messages()

    container = st.container()
    response_container = st.container()
    content = ''

    with container:
        url = get_url_input()
        is_valid_url = validate_url(url)
        if not is_valid_url:
            st.write('Please input valid url')
            answer = None
        else:
            content = get_content(url)
            if content:
                # prompt = build_prompt(content)


                # st.session_state.messages.append(HumanMessage(content=prompt))
                with st.spinner("ChatGPT is typing ..."):
                    answer, cost = get_answer(map_chain, content)
                st.session_state.costs.append(cost)
            else:
                answer = None

    if answer:
        with response_container:
            st.markdown("## Summary")
            st.write(answer)
            st.markdown("---")
            st.markdown("## Original Text")
            st.write(content)

    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")

if __name__ == '__main__':
    main()
