# import utils
import sqlite3
import streamlit as st
from pathlib import Path
from sqlalchemy import create_engine

from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    AIMessage,
    HumanMessage,
)
import pandas as pd

from dotenv import load_dotenv
from pathlib import Path

HERE = Path(__file__).parent.resolve()
load_dotenv()


def init_page():
    if st.session_state.get('page') != __file__:
        # ページ遷移時に状態を初期化しておく
        st.session_state.clear()
        st.session_state.page = __file__
    st.set_page_config(page_title="ChatSQL", page_icon="🛢")
    st.header('Chat with SQL database')
    st.write('日本の各都道府県の特産品に関する質問をできます')
    st.write('参考テーブル： https://ja.wikipedia.org/wiki/%E7%89%B9%E7%94%A3%E5%93%81')


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]
        st.session_state.costs = []


class SqlChatbot:
    def __init__(self):
        # utils.sync_st_session()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

    def setup_db(_self):
        db_filepath = (HERE.parent / "assets/tokusan.db").absolute()

        def creator(): return sqlite3.connect(f"file:{db_filepath}?mode=ro", uri=True)
        engine = create_engine(f'sqlite:////', creator=creator)
        db = SQLDatabase(engine)

        with st.sidebar.expander('Database tables', expanded=True):
            st.info('\n- '+'\n- '.join(db.get_usable_table_names()))

        with engine.connect() as conn:
            st.dataframe(pd.read_sql_query('SELECT * FROM tokusan', con=conn))

        return db

    def setup_sql_agent(_self, db):
        agent = create_sql_agent(
            llm=_self.llm,
            db=db,
            top_k=10,
            verbose=False,
            agent_type="openai-tools",
            handle_parsing_errors=True,
            handle_sql_errors=True
        )
        return agent

    # @utils.enable_chat_history
    def main(self):
        db = self.setup_db()
        agent = self.setup_sql_agent(db)

        # チャット履歴の表示
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

        user_query = st.chat_input(placeholder="Ask me anything!")

        if user_query:
            st.session_state.messages.append(HumanMessage(user_query))
            st.chat_message("user").write(user_query)

            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container())
                result = agent.invoke(
                    {"input": user_query},
                    {"callbacks": [st_cb]}
                )
                response = result["output"]
                st.session_state.messages.append(AIMessage(response))
                st.write(response)
                # utils.print_qa(SqlChatbot, user_query, response)


if __name__ == "__main__":
    init_page()
    init_messages()
    obj = SqlChatbot()
    obj.main()
