import pandas as pd
from pathlib import Path
import sqlite3

HERE = Path(__file__).parent.resolve()

html = (HERE / 'tokusan.html').read_text(encoding='utf-8')

# HTMLのテーブルをデータフレームに変換
dfs = pd.read_html(html)

# # 最初のテーブルをCSVとして保存
# dfs[0].to_csv("output.csv", index=False)

df=dfs[0]

conn = sqlite3.connect('./tokusan.db')
df.to_sql('tokusan', conn, index=None)
conn.close()
