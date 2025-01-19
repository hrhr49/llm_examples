import streamlit as st
import streamlit_mermaid as stmd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from dotenv import load_dotenv

load_dotenv()


def main():
    if st.session_state.get('page') != __file__:
        # ページ遷移時に状態を初期化しておく
        st.session_state.clear()
        st.session_state.page = __file__
    st.set_page_config(
        page_title="プロンプト相談",
        page_icon="🤗",
        layout='wide',
    )
    st.header("プロンプト相談 🤗")
    st.write("AIチャットに投げるプロンプトを作るための補助ツールです。")
    st.write("作成したプロンプトを入力すると、AIがそのプロンプトの内容を評価しアドバイスしてくれます。")

    prompt_template = PromptTemplate(
        input_variables=["user_input_prompt"],
        template='''
### 前提 ###
私はChatGPTモデルに与えるプロンプトを作成しています。

### 命令 ###
あなたは私が作ったプロンプトの品質をより良くするためのアドバイスを私に提供してください。
私は抽象的な説明が苦手なので具体例を交えながら説明ください。

### 出力形式 ###
* 出力はMarkdown形式
* 各アドバイスは箇条書きで記載
* アドバイスの後に改善後のプロンプトの提案
* 出力は必ず日本語

### 補足 ###
チェックしてほしい観点は [プロンプトに必要な観点 START] から[プロンプトに必要な観点 END]の間に記載した内容です。
チェックしてほしいプロンプトは [プロンプト START] から[プロンプト END]の間に記載した内容です。

[プロンプトに必要な観点 START]
* 命令と入力データを明確に分けて記述する
  - 「"""」や「###」などといった区切り記号を使うと良い
* 目的を明確にする
* 具体的に書いている
  - 不足する情報がない
  - 入出力の例を挙げることも有効
* 質問の文脈を具体的に説明する
  - どのような前提条件があるか記載する
  - 出力に求める条件は何かを記載する
  - 自分がどのような立場でどういったことをしたいのかを記載する
* 出力形式を指定する
  - 箇条書き、プレーンテキスト、JSON、CSV、ソースコード、表形式など
  - 出力データに期待する構造を記載している
  - 日本語、英語など
* 使用されている言葉に一貫性を持たせる
  - 表記ゆれが起きないようにする
* AIにロールを付与してロールプレイさせる
[プロンプトに必要な観点 END]

[プロンプト START]
{user_input_prompt}
[プロンプト END]
'''
    )

    user_input_prompt = st.text_area("作成中のプロンプトを入力してください", height=300)
    if not user_input_prompt:
        return

    # OpenAI LLMの初期化
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # 修正: ChatOpenAIを使用

    # 処理概要の生成
    adviser_chain = LLMChain(llm=llm, prompt=prompt_template)
    with st.spinner("処理概要を生成中..."):
        advice = adviser_chain.run(user_input_prompt=user_input_prompt)
    st.subheader("アドバイス")
    st.write(advice)

if __name__ == '__main__':
    main()
