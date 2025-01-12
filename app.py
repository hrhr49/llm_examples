import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to LLM Samples! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
"""
このStreamlitアプリでは、LLMを使ったサンプルを用意しています。
動かしてみたいデモを選んで試しに動かしてみましょう！
"""
)
