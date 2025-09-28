# ---- 📦 ライブラリ読み込み ----
import os
import io
from io import BytesIO
import base64
from typing import Tuple
import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from openai import OpenAI
from openai import RateLimitError,APIStatusError

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
from openai import OpenAI
import datetime


# ---- ページ設定（最初に！）----
st.set_page_config(page_title="未来ひろば", page_icon="🌱", layout="centered")

####【追加】CSS デザイン用---
st.markdown("""

</style>
""", unsafe_allow_html=True)

# ---- APIキー取得 ----
def get_api_key(env_key: str = "OPENAI_API_KEY") -> str | None:
    key = os.getenv(env_key)
    if key:
        return key
    try:
        return st.secrets[env_key]
    except Exception:
        return None

API_KEY = get_api_key()
if not API_KEY:
    st.error(
        "OpenAI APIキーが見つかりません。\n\n"
        "■ 推奨（ローカル学習向け）\n"
        "  1) .env を作成し OPENAI_API_KEY=sk-xxxx を記載\n"
        "  2) このアプリを再実行\n\n"
        "■ 参考（secrets を使う場合）\n"
        "  .streamlit/secrets.toml に OPENAI_API_KEY を記載（※リポジトリにコミットしない）\n"
        "  公式: st.secrets / secrets.toml の使い方はドキュメント参照"
    )
    st.stop()

client = OpenAI(api_key=API_KEY)



######【追加】 安全対策　意味あるのか？？
def sanitize_html(html: str) -> str:
    # <script> と on* ハンドラを除去して安全側に寄せる
    html = re.sub(r"(?is)<script.*?>.*?</script>", "", html)
    html = re.sub(r'(?is)\son\w+\s*=\s*(["\']).*?\1', "", html)  # onClick 等
    return html


# ---- タイトル ----
#【変更点】
# 'header.mp4'をバイナリモードで開く
with open("header.mp4", "rb") as f:
    video_file = f.read()
    video_bytes = base64.b64encode(video_file).decode()

# HTML文字列の作成
video_html = f"""
    <video width="100%" autoplay loop muted playsinline>
        <source src="data:video/mp4;base64,{video_bytes}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    """

# markdownとしてHTMLを表示
st.markdown(video_html, unsafe_allow_html=True)


# ---- サンプルデータ（男の子・女の子の平均身長とSD） ---- #
data = {
    "age": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    "M_mean": [110.3, 116.5, 122.6, 128.1, 133.5, 139.0, 145.2, 152.8, 160.0, 165.4, 168.3, 169.9, 170.6],
    "M_sd":   [4.71,  4.94,  5.20,  5.41,  5.73,  6.07,  7.13,  8.03,  7.62,  6.72,  5.93,  5.85,  5.87],
    "F_mean": [109.4, 115.6, 121.4, 127.3, 133.4, 140.2, 146.6, 151.9, 154.8, 156.5, 157.2, 157.7, 157.9],
    "F_sd":   [4.70,  4.92,  5.14,  5.55,  6.16,  6.80,  6.59,  5.89,  5.48,  5.32,  5.33,  5.37,  5.34]
}
df = pd.DataFrame(data)

# 【変更点】2つのカラムを作成します 
col1, col2 = st.columns(2)

# col1（左側のカラム）に子どもの現在の情報を配置
with col1:
    st.subheader("お子さんの情報") #
    sex = st.selectbox("性別を選んでください", ["男の子", "女の子"], key="gender_selectbox")
    age_now = st.number_input("年齢", 5, 18, 10, key="age_now")
    height_now = st.number_input("身長 (cm)", 90.0, 200.0, 140.0, format="%.1f", key="height_now")

# col2（右側のカラム）に両親の情報を配置
with col2:
    st.subheader("ご両親の情報") 
    father_height = st.number_input("お父さんの身長 (cm)", 150.0, 200.0, 175.0, format="%.1f", key="father_height") 
    mother_height = st.number_input("お母さんの身長 (cm)", 140.0, 180.0, 160.0, format="%.1f", key="mother_height") 

# 【変更点】サブタイトル
st.subheader("お子さんの生活習慣について教えてください")
st.caption("[はい]の場合はオン（オレンジの状態）に、[いいえ]の場合はオフ（グレーの状態）にしてください")


#st.radioをst.toggleに変更
sleep = st.toggle("睡眠で休養が十分とれていますか？", value=True)
exercise = st.toggle("1回30分以上の軽く汗をかく運動を週2日以上、1年以上実施？", value=True)
breakfast = st.toggle("朝食を抜くことが週に3回以上ありますか？") # デフォルトはFalse (オフ)
family_meal = st.toggle("週に3回以上家族とご飯を食べていますか？", value=True)



# 両親平均身長（Mid-Parental Height）
if sex == "男の子":
    mph = (father_height + mother_height + 13) / 2
    mean_col, sd_col = "M_mean", "M_sd"
else:
    mph = (father_height + mother_height - 13) / 2
    mean_col, sd_col = "F_mean", "F_sd"

min_future = mph - 7.5
max_future = mph + 7.5

# ---- 成長曲線グラフ ----
st.subheader("⭐️ 未来身長予測グラフ")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["age"], y=df[mean_col], mode="lines", name="平均身長", line=dict(color="#50ABA2")))
fig.add_trace(go.Scatter(
    x=pd.concat([df["age"], df["age"][::-1]]),
    y=pd.concat([df[mean_col]-df[sd_col], (df[mean_col]+df[sd_col])[::-1]]),
    fill='toself', fillcolor='rgba(80,171,162,0.3)',    
    line=dict(color='rgba(255,255,255,0)'), name="標準偏差"
))
fig.add_trace(go.Scatter(x=[age_now], y=[height_now], mode="markers",
                         marker=dict(color="#D65A91", size=12), name="現在の身長"))

# MIN-MAX身長の範囲を薄いピンクで塗りつぶし
fig.add_shape(
    type="rect",
    x0=df["age"].min(),
    y0=min_future,
    x1=df["age"].max(),
    y1=max_future,
    fillcolor="rgba(214, 90, 145, 0.1)",
    line=dict(color="rgba(214, 90, 145, 0)"),
    layer="below"
)

fig.add_hline(y=mph, line_dash="dash", line_color="#D65A91", annotation_text="両親平均身長")
fig.add_hline(y=min_future, line_dash="dot", line_color="#D65A91", annotation_text="MIN身長")
fig.add_hline(y=max_future, line_dash="dot", line_color="#D65A91", annotation_text="MAX身長")


fig.update_layout(
    #title="成長予測と基準レンジ",
    xaxis_title="年齢",
    yaxis_title="身長 (cm)",
    template="plotly_white",
    plot_bgcolor="#f5f5f5",   # グラフエリアの背景色
    paper_bgcolor="#f5f5f5",  # 囲い全体の背景色
    font=dict(color="#000"),   # 追加：全体の文字色を黒に
    xaxis=dict(
        title_font=dict(color="#000"),
        tickfont=dict(color="#000"),
        showline=True,  # X軸の線を表示
        linewidth=1,    # X軸の線の太さ
        linecolor='black'  # X軸の線の色
    ),
    yaxis=dict(
        title_font=dict(color="#000"),
        tickfont=dict(color="#000"),
        showline=True,  # Y軸の線を表示
        linewidth=1,    # Y軸の線の太さ
        linecolor='black'  # Y軸の線の色
    )
)
st.plotly_chart(fig, use_container_width=True)
st.caption("※18歳以降は身長の伸びが止まるため予測を表示していません。")

#######【変更】未来予測 ----最小値、最大値、初期値
st.subheader("⭐️ 未来の予測身長")
target_height_age = st.slider("予測したい年齢を選んでください", age_now, 25, 15, key="slider_target_height")

# 全国平均の最終身長
final_mean = df[mean_col].iloc[-1]

# 各年齢の成長率（全国平均 / 最終平均）
df["growth_ratio"] = df[mean_col] / final_mean

# 選択年齢での成長率を補間
ratio = np.interp(target_height_age, df["age"], df["growth_ratio"])

# 予測身長 = 両親平均 × 成長率
pred_height = mph * ratio

######【変更】結果表示
# st.success(f"{target_height_age}歳の予測身長は {pred_height:.1f} cm です！")
st.markdown(f"""
<div style="padding: 0.75rem 1rem; background-color: #FFBDDA; border: 1.5px solid #D65A91; border-radius: 0.375rem; color: black;">
    {target_height_age}歳の予測身長は  <span style="font-size: 1.5em; font-weight: bold;">{pred_height:.1f}cm</span> です！
</div>
""", unsafe_allow_html=True)

# 余白を追加
st.markdown("<br>", unsafe_allow_html=True)

# レンジ判定
if pred_height < min_future:
    st.info("これからの成長が楽しみだね💪")
elif pred_height > max_future:
    st.info("未来は両親越えだね✨")
else:
    st.info("順調に成長しているね🙌")


# -------------------------
# 画像生成
# -------------------------

##一旦削除
# st.header("⭐️ 未来の子どもイメージを生成")

# -------------------------
# 画像生成　<<< 顔　>>>
# -------------------------
st.subheader("⭐️ 未来の顔のイメージ画像を生成")


# 画像アップロード
st.caption("画像を2枚アップロードしてください（両親各1枚、またはお子さん本人2枚）。")
col1, col2 = st.columns(2)
with col1:
    img1_file = st.file_uploader("画像1をアップロード", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
with col2:
    img2_file = st.file_uploader("画像2をアップロード", type=["png", "jpg", "jpeg"], accept_multiple_files=False)



# DALLE3用関数
def image_to_data_url(img: Image.Image) -> str:
    """Convert PIL image to a data URL string usable by OpenAI vision (image_url)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def analyze_parents_to_traits(img1: Image.Image, img2: Image.Image, style: str, target_age: int, sex: str) -> str:
    """Use GPT-4o-mini to extract visual traits from two parent images and craft a child description."""
    data_url_1 = image_to_data_url(img1)
    data_url_2 = image_to_data_url(img2)

    user_prompt = (
        "以下の2枚の画像から、共通/中間的な特徴を踏まえ、"
        f"『{target_age}歳の子ども{sex}』の外見的特徴を日本語で箇条書きにしてください。\n"
        "髪/目/肌の色味、顔の輪郭、鼻や唇の傾向、眉・まつ毛、そばかすやえくぼ等、印象のトーンも含めて。\n"
        "実在人物の特定につながる記述は避け、合成的・一般的な表現で。6〜10行。"
    )

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_prompt},
                {"type": "input_image", "image_url": data_url_1},
                {"type": "input_image", "image_url": data_url_2},
            ],
        }],
        temperature=0.7,
        max_output_tokens=500,
    )
    # 新SDKの簡易取得
    try:
        traits_text = resp.output_text.strip()
    except Exception:
        # 念のための後方互換
        traits_text = resp.choices[0].message.content.strip()

    # DALL·E 3 向けプロンプト
    dalle_prompt = f"""
親しみやすい上半身のポートレート。年齢は約{target_age}歳。{sex}らしいイラスト。


画像タイプ:
- フラットな2Dイラスト、上半身のポートレイト、正面
- 明るく温かみのある背景（余計な装飾はなし）
- 遠近感なし（平行投影）

構図ルール:
- 中央に被写体が一人、周囲に装飾はなし、背景はシンプル


表情・雰囲気:
- 明るく温かみのあるシンプルな無地の背景。遠近感なし（平行投影）

禁止事項:
- 画像内に数字、矢印、定規、文字や余計な記号を描かない

以下の特徴を適度にブレンドして表現してください（一般化・合成的に）:
{traits_text}

スタイル: {style}
やわらかいライティング、文字・テキストは入れない。
    """.strip()
    return dalle_prompt

def generate_with_dalle3(prompt: str, size: str = "1024x1024") -> Image.Image:
    """Generate an image with DALL·E 3 from a text prompt and return a PIL image."""
    img_resp = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,
        n=1,
        response_format="b64_json",
    )
    b64 = img_resp.data[0].b64_json
    binary = base64.b64decode(b64)
    return Image.open(io.BytesIO(binary)).convert("RGB")




# 画像生成用スタイル選択
style = "普通"  # ← 追加
out_size ="1024x1024"

# セッションステートに画像を保持する変数を初期化
if "out_face_img" not in st.session_state:
    st.session_state["out_face_img"] = None
if "dalle_prompt" not in st.session_state:
    st.session_state["dalle_prompt"] = None
if "out_body_img" not in st.session_state:  # ← 追加
    st.session_state["out_body_img"] = None

# 画像生成用年齢の初期値を設定
target_age = 20  # または age_now など、適切な初期値

if img1_file and img2_file:
    img1 = Image.open(img1_file).convert("RGB")
    img2 = Image.open(img2_file).convert("RGB")

    st.subheader("プレビュー")
    p1, p2 = st.columns(2)
    with p1:
        st.image(img1, caption="画像1")
    with p2:
        st.image(img2, caption="画像2")

    # ---- 画像生成用年齢設定 ----最小値、最大値、初期値
    target_age = st.slider("生成したい未来の年齢を選んでください", age_now, 25, 15, key="slider_target_image")

    # 体型イメージ選択-----------
    style = st.selectbox(
        "体型イメージを選んでください",
        ["細身", "普通", "筋肉質", "がっしり", "ふくよか"],
        index=0,
        key="body_type"
    )


    #### 顔生成 -----------
    # 顔イメージ生成ボタンの色をピンクにするCSS
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #D65A91 !important;
        color: black !important;
        border: none !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #D65A91 !important;
        color: black !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if st.button(f"{target_age}歳の顔イメージ生成", type="primary"):
        try:
            with st.spinner("特徴抽出中..."):
                dalle_prompt = analyze_parents_to_traits(
                    img1, img2, style=style, target_age=target_age, sex=sex)

            # 禁止ワードを置換
            dalle_prompt = dalle_prompt.replace("顔", "イラスト")
            dalle_prompt = dalle_prompt.replace("写真", "イラスト")
            dalle_prompt = dalle_prompt.replace("実在", "架空")
            
            dalle_prompt = dalle_prompt.replace("子ども", "キャラクター")

            st.success("特徴抽出完了。DALL·E 3で画像を生成します")
            # st.code(dalle_prompt, language="markdown")  # プロンプトを非表示

            with st.spinner("DALL·E 3で画像を生成中…"):
                out_face_img = generate_with_dalle3(dalle_prompt, size=out_size)

            # セッションステートに保存--------
            st.session_state["out_face_img"] = out_face_img
            st.session_state["dalle_prompt"] = dalle_prompt


        except Exception as e:
            st.error("生成に失敗しました。APIキーや権限、請求設定、組織の検証状態（DALL·E 3は要件が厳しめ）を確認してください。")
            with st.expander("エラーメッセージ"):
                st.exception(e)
else:
    st.info("画像を2枚アップロードしてください。")

# --- 顔画像を表示する ---
if st.session_state["out_face_img"] is not None:
    st.subheader("顔イメージ")
    st.image(st.session_state["out_face_img"], caption=f"{target_age}歳の顔イメージ", use_container_width=True)
    st.success("生成が完了しました！")  # ← ここを追加
    #st.caption("2枚の画像を分析→テキスト化→DALL·E 3で合成ポートレートを生成しています。実在の子ではなく**架空**のイメージです。")
    buf = io.BytesIO()
    st.session_state["out_face_img"].save(buf, format="PNG")
    st.download_button(
        label="顔画像をダウンロード (PNG)",
        data=buf.getvalue(),
        file_name="predicted_child_face.png",
        mime="image/png",
    )
    st.caption("2枚の画像を分析→テキスト化→DALL·E 3で合成ポートレートを生成しています。実在の子ではなく**架空**のイメージです。")


# -------------------------
# 画像生成　<<< 身長　>>>
# -------------------------
st.subheader("⭐️ 未来の身長のイメージ画像を生成")
body_type = style  # 体型イメージをstyleから取得

prompt_text = f"""
画像タイプ:
- フラットな2Dイラスト、全身、正面
- 明るく温かみのある背景（余計な装飾はなし）
- 遠近感なし（平行投影）

登場人物:
- 左: お父さん（身長 {father_height} cm）
- 中央: 子ども（{sex}, {target_age} 歳, 予測身長 {pred_height:.1f} cm、体型は「{body_type}」）
- 右: お母さん（身長 {mother_height} cm）

構図ルール:
- 家族3人が横に並び、背比べをしているような姿
- 3人の足元は同じ床ラインに揃える
- 各人物の頭頂の高さは身長に比例して描く
- 子どもは中央で識別しやすい服装にする

子どもの特徴:
- 男の子: 青いTシャツとジーンズ
- 女の子: 青いトップスとスカート
- 少し誇らしげに背筋を伸ばして立つ姿

表情・雰囲気:
- 両親は柔らかく微笑み、子どもの成長を温かく見守っている
- 家族のつながりや成長の喜びが伝わる雰囲気

禁止事項:
- 画像内に数字、矢印、定規、文字や余計な記号を描かない
"""


prompt_text = prompt_text.replace(f"{target_age}歳", "若い")

# 身長イメージ生成ボタンの色をピンクにするCSS
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #D65A91 !important;
    color: black !important;
    border: none !important;
}
div.stButton > button:first-child:hover {
    background-color: #D65A91 !important;
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

if st.button(f"{target_age}歳の身長イメージ生成", type="primary"):
    with st.spinner("AIが画像を生成しています..."):
        try:
            resp = client.images.generate(
                model="gpt-image-1",
                prompt=prompt_text,
                size="1024x1024"
            )

            # URLが返らないケースに対応
            out_body_img = None 
            if resp.data and hasattr(resp.data[0], "b64_json") and resp.data[0].b64_json:
                image_base64 = resp.data[0].b64_json
                image_bytes = base64.b64decode(image_base64)
                out_body_img = Image.open(BytesIO(image_bytes))
            elif resp.data and hasattr(resp.data[0], "url") and resp.data[0].url:
                # URLの場合は画像をダウンロードして保存
                import requests
                response = requests.get(resp.data[0].url)
                out_body_img = Image.open(BytesIO(response.content))
            else:
                st.error("画像データが返されませんでした。レスポンスを確認してください。")
                st.write(resp)

            st.success("生成が完了しました！")
            # セッションステートに保存
            if out_body_img is not None:
                st.session_state["out_body_img"] = out_body_img

        except Exception as e:
            st.error(f"画像生成でエラーが発生しました: {e}")

# --- 身長画像を表示する ---
if st.session_state["out_body_img"] is not None:
    st.image(st.session_state["out_body_img"], use_container_width=True)
    st.markdown(
    f"""
    <div style="text-align: center;">
    {target_age}歳の予測身長は {pred_height:.1f}cm です。<br>
    左右にご両親、真ん中にお子さんがいます。
    </div>
    """, unsafe_allow_html=True
)
    buf = io.BytesIO()
    st.session_state["out_body_img"].save(buf, format="PNG")            
    st.download_button(
        label="身長画像をダウンロード (PNG)",
        data=buf.getvalue(),
        file_name="predicted_child_body.png",
        mime="image/png",
    )

# Footer
st.markdown(
    """
---
**注意**: これは合成イメージであり、特定の人物を示すものではありません。\
第三者の画像を使う場合は、必ず本人同意を得てください。
    """
)



# ---- ### ウソかマコトか豆情報 ----
st.subheader("⭐️ ウソかマコトか身長にまつわる豆情報")

# CSVファイルのパス（同じフォルダ内にある想定）
csv_path = "trivia.csv"

try:
    # CSV読み込み
    df = pd.read_csv(csv_path)

    # 🎲 ランダム表示ボタン
    if st.button("ランダム表示"):
        random_row = df.sample(1).iloc[0]
        st.subheader(f"📝 {random_row['title']}")
        st.write(random_row['content'])

except FileNotFoundError:
    st.error(f"❌ ファイルが見つかりません：{csv_path}")
except Exception as e:
    st.error(f"⚠️ 読み込みエラー: {e}")


#######【追加】今月の旬の食材画像表示 ----
import streamlit as st
import os
import datetime
# フォルダ名（全角注意）
image_dir = "freshfood"
# 今月の月番号を取得（1〜12）
current_month = datetime.datetime.now().month
# ファイル名を生成（page_1.png ～ page_12.png）
image_filename = f"page_{current_month}.png"
image_path = os.path.join(image_dir, image_filename)
# タイトルと説明
st.subheader("⭐️今月の旬の食材")
#st.markdown(f"### {current_month}月の画像: {image_filename}")
# 画像表示
if os.path.exists(image_path):
    st.image(image_path, use_container_width=True)
else:
    st.warning("該当月の画像がまだ用意されていません。")

#######【追加】成長サポート情報 ----
st.subheader("⭐️ 成長サポート情報")
# CSVファイルのリスト（同じフォルダ内にある想定）
files = {
    "training.csv": "🏋️ トレーニング",
    "sleep.csv": "💤 睡眠",
    "food.csv": "🍴 食事"
}
try:
    # 🎲 ランダム表示ボタン
    if st.button("情報をランダム表示"):
        for file, label in files.items():
            df = pd.read_csv(file)
            random_row = df.sample(1).iloc[0]
            
            st.subheader(f"{label} - {random_row['title']}")
            st.markdown(f"<div style='font-size: 18px;'>{random_row['content']}</div>", unsafe_allow_html=True)
except FileNotFoundError as e:
    st.error(f"❌ ファイルが見つかりません: {e}")
except Exception as e:
    st.error(f"⚠️ 読み込みエラー: {e}")