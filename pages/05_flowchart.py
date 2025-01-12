import streamlit as st
import streamlit_mermaid as stmd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI  # 修正: ChatOpenAIを使用
import os
from dotenv import load_dotenv

load_dotenv()


# Streamlitアプリの設定
st.title("ソースコードからフローチャートと処理概要を生成するアプリ")
st.write("ソースコードを入力またはアップロードして処理概要とフローチャートを生成します。")

# ソースコードの入力またはアップロード
source_code = ""
input_method = st.radio("ソースコードの入力方法を選択してください", ("直接入力", "ファイルアップロード"))

if input_method == "直接入力":
    source_code = st.text_area("ソースコードを入力してください (Pythonのみ対応)", height=300)
elif input_method == "ファイルアップロード":
    uploaded_file = st.file_uploader("ソースコードをアップロードしてください (Pythonのみ対応)", type=["py"])
    if uploaded_file:
        source_code = uploaded_file.read().decode("utf-8")

if source_code:
    # 処理概要生成のプロンプト
    summary_prompt = PromptTemplate(
        input_variables=["code"],
        template="""
        以下のPythonコードを読んで、処理の概要を簡潔に説明してください：
        ```
        {code}
        ```
        """
    )

    # フローチャート生成のプロンプト
    flowchart_prompt = PromptTemplate(
        input_variables=["code"],
        template="""
        以下のPythonコードを読んで、フローチャートの構造をMermaid.jsの構文で出力してください。
        Mermaid.jsの構文例(「```mermaid」から「```」の間の部分だけが出力です):
        ```mermaid
        graph TD
        A[Start] -->|条件1| B[処理1]
        B --> C[End]
        ```
        コード:
        ```
        {code}
        ```
        """
    )

    # OpenAI LLMの初期化
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # 修正: ChatOpenAIを使用

    # 処理概要の生成
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
    with st.spinner("処理概要を生成中..."):
        summary = summary_chain.run(code=source_code)
    st.subheader("処理概要")
    st.write(summary)

    # フローチャート構造の生成
    flowchart_chain = LLMChain(llm=llm, prompt=flowchart_prompt)
    with st.spinner("フローチャートを生成中..."):
        flowchart_text = flowchart_chain.run(code=source_code)
    st.subheader("フローチャート")

    flowchart_text = (
        flowchart_text
        .replace('```mermaid', '')
        .replace('```', '')
        .strip()
    )

    st.code(flowchart_text, language='mermaid')
    # Mermaid.js構文をStreamlitに埋め込む
    try:
        stmd.st_mermaid(flowchart_text)
    except Exception as e:
        st.error("フローチャートの描画に失敗しました。")
        st.error(str(e))
