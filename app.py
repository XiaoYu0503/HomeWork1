
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score

# --- OLS 核心函式 ---
def ols_fit(x, y):
    """
    使用 OLS 閉式解計算線性回歸係數。
    β = (X^T X)^(-1) X^T y
    """
    # 建立設計矩陣 X (加上截距項)
    X = np.vstack([x, np.ones(len(x))]).T
    try:
        # 計算係數
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        return beta[0], beta[1]  # a_hat, b_hat
    except np.linalg.LinAlgError:
        # 如果矩陣不可逆，回傳 None
        st.error("OLS 計算失敗：矩陣為奇異方陣 (singular matrix)，無法求逆。")
        return None, None

# --- Streamlit 介面 ---
st.set_page_config(page_title="簡單線性回歸模擬器", layout="wide")

st.title("📈 簡單線性回歸互動模擬器")
st.write("---")

# --- 資料來源選擇 ---
source_option = st.radio(
    "選擇資料來源：",
    ("自動生成資料", "上傳 CSV 檔案"),
    horizontal=True,
    help="您可以動態生成資料來觀察參數效果，或上傳自己的 CSV 檔進行擬合。"
)

# --- 側邊欄：參數控制 ---
with st.sidebar:
    st.header("⚙️ 參數設定")

    if source_option == "自動生成資料":
        st.subheader("資料生成參數")
        a_true = st.slider("真實 a (斜率)", -10.0, 10.0, 2.5, 0.1)
        b_true = st.slider("真實 b (截距)", -10.0, 10.0, 5.0, 0.1)
        noise_sigma = st.slider("雜訊標準差 σ", 0.0, 20.0, 3.0, 0.5)
        num_points = st.slider("資料點數量", 10, 1000, 100, 10)
        x_min = st.number_input("x 範圍 (最小值)", value=0.0)
        x_max = st.number_input("x 範圍 (最大值)", value=10.0)
        random_seed = st.number_input("隨機亂數種子", value=42, step=1)

    else: # 上傳 CSV
        st.subheader("CSV 上傳")
        uploaded_file = st.file_uploader(
            "選擇一個 CSV 檔案",
            type="csv",
            help="CSV 需包含名為 'x' 和 'y' 的欄位。"
        )

# --- 資料準備 ---
data = None
if source_option == "自動生成資料":
    if x_min >= x_max:
        st.error("x 範圍的最小值必須小於最大值。")
    else:
        # 設定亂數種子
        np.random.seed(random_seed)
        # 生成 x
        x = np.linspace(x_min, x_max, num_points)
        # 生成雜訊
        epsilon = np.random.normal(0, noise_sigma, num_points)
        # 生成 y
        y = a_true * x + b_true + epsilon
        # 建立 DataFrame
        data = pd.DataFrame({"x": x, "y": y})
        st.sidebar.success(f"已成功生成 {num_points} 筆資料。")

elif source_option == "上傳 CSV 檔案" and uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        if 'x' not in data.columns or 'y' not in data.columns:
            st.error("上傳的 CSV 檔案中必須包含 'x' 和 'y' 欄位。")
            data = None
        else:
            st.sidebar.success(f"已成功讀取 {len(data)} 筆資料。")
    except Exception as e:
        st.error(f"讀取檔案時發生錯誤：{e}")
        data = None

# --- 主畫面佈局 ---
if data is not None:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 資料視覺化與擬合結果")

        # --- 建模與評估 ---
        x_data = data['x'].values
        y_data = data['y'].values
        a_hat, b_hat = ols_fit(x_data, y_data)

        if a_hat is not None:
            # 產生擬合線的 y 值
            y_pred = a_hat * x_data + b_hat

            # --- 繪圖 ---
            fig = px.scatter(data, x='x', y='y', title="資料散點圖與 OLS 擬合線", labels={'x': 'X', 'y': 'Y'})
            fig.add_trace(go.Scatter(x=x_data, y=y_pred, mode='lines', name='OLS 擬合線', line=dict(color='red', width=3)))

            # 如果是生成資料，也畫出真實的線
            if source_option == "自動生成資料":
                y_true_line = a_true * x_data + b_true
                fig.add_trace(go.Scatter(x=x_data, y=y_true_line, mode='lines', name='真實線', line=dict(color='green', dash='dash')))

            fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📝 模型評估")
        if a_hat is not None:
            # 計算指標
            mse = mean_squared_error(y_data, y_pred)
            r2 = r2_score(y_data, y_pred)

            st.markdown("#### OLS 估計係數")
            st.metric(label="估計斜率 (â)", value=f"{a_hat:.4f}")
            st.metric(label="估計截距 (b̂)", value=f"{b_hat:.4f}")

            if source_option == "自動生成資料":
                st.markdown("---")
                st.markdown("#### 真實參數")
                st.metric(label="真實斜率 (a)", value=f"{a_true:.4f}")
                st.metric(label="真實截距 (b)", value=f"{b_true:.4f}")

            st.markdown("---")
            st.markdown("#### 模型效能")
            st.metric(label="均方誤差 (MSE)", value=f"{mse:.4f}")
            st.metric(label="R² (決定係數)", value=f"{r2:.4f}")

    # --- 資料下載 ---
    st.subheader("📥 資料下載")
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="下載資料為 CSV",
        data=csv,
        file_name='linear_regression_data.csv',
        mime='text/csv',
    )

    # --- 顯示資料表 ---
    with st.expander("點此查看原始資料"):
        st.dataframe(data)

else:
    st.info("請在左側側邊欄設定參數以生成資料，或上傳一個 CSV 檔案。")

