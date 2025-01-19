import streamlit as st

from langchain_openai import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.callbacks import get_openai_callback

from dotenv import load_dotenv
load_dotenv()


def init_page():
    if st.session_state.get('page') != __file__:
        # ページ遷移時に状態を初期化しておく
        st.session_state.clear()
        st.session_state.page = __file__
    st.set_page_config(
        page_title="チャットサンプル",
        page_icon="🤗",
        layout='wide',
    )
    st.header("チャットサンプル 🤗")
    st.sidebar.title("Options")


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]
        st.session_state.costs = []


def select_model():
    model_list = ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4"]
    model = st.sidebar.radio("Choose a model:", model_list)
    temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=2.0, value=0.0, step=0.01)
    return ChatOpenAI(temperature=temperature, model=model, streaming=True, stream_usage=True)


def main():
    init_page()

    llm = select_model()

    init_messages()

    messages = st.session_state.get('messages', [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message('assistant'):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message('user'):
                st.markdown(message.content)
        else:  # isinstance(message, SystemMessage):
            st.write(f"System message: {message.content}")

    # ユーザーの入力を監視
    user_input = st.chat_input("聞きたいことを入力してね！")
    if user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))
        st.chat_message("user").markdown(user_input)
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container(), collapse_completed_thoughts=False)
            with get_openai_callback() as cb:
                response = llm.invoke(messages, {'callbacks': [st_callback]})
            st.session_state.costs.append(cb.total_cost)
        st.session_state.messages.append(AIMessage(content=response.content))

    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.9f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.9f}")


if __name__ == '__main__':
    main()
