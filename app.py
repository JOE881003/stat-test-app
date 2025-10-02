import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import rcParams
# ========= 頁面設定 =========
st.set_page_config(
    page_title="統計檢定小工具 📊",
    page_icon="📊",
    layout="centered"
)

# ========= 中文字型設定 =========
rcParams['font.family'] = 'Microsoft JhengHei'
rcParams['axes.unicode_minus'] = False

st.title("假設檢定小工具 (Z / t / 卡方 / F 檢定)")

# ========== 選擇檢定類型 ==========
test_type = st.selectbox("選擇檢定類型", ["Z 檢定", "t 檢定", "卡方檢定 (Chi-square)", "F 檢定 (變異數比)"])
with st.expander("📚 各種檢定的使用時機"):
    st.markdown("""
    ### 🔵 Z 檢定 (Z-test)
    - 適用：母體標準差已知，或樣本數大 (n ≥ 30)，資料近似常態  
    - 常見：檢定樣本均值是否等於母體均值  

    ### 🟢 t 檢定 (t-test)
    - 適用：母體標準差未知，小樣本 (n < 30)，資料近似常態  
    - 常見：檢定樣本均值是否等於母體均值；比較兩組均值差異  

    ### 🟡 卡方檢定 (Chi-square test)
    - 適用：類別資料  
    - 常見：檢查觀測值 vs 期望值差異（適合度）；檢查兩個類別變數是否獨立  

    ### 🟠 F 檢定 (F-test)
    - 適用：比較兩組變異數是否相等；作為 ANOVA 基礎  
    - 常見：不同條件下的變異數比較；單因子/多因子變異數分析  
    """)

alpha = st.selectbox("顯著水準 α", [0.1, 0.05, 0.01])

# ========== Z 檢定 ==========
if test_type == "Z 檢定":
    x_bar = st.number_input("樣本均值 (x̄)", value=5.0)
    mu_0 = st.number_input("假設母體均值 (μ₀)", value=5.0)
    sigma = st.number_input("母體標準差 (σ)", value=1.0)
    n = st.number_input("樣本數 (n)", value=30, step=1)

    se = sigma / np.sqrt(n)
    z = (x_bar - mu_0) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    critical = stats.norm.ppf(1 - alpha/2)

    st.subheader("檢定結果")
    st.write(f"Z 統計量 = {z:.3f}")
    st.write(f"p-value = {p_value:.3f}")
    if abs(z) > critical:
        st.error("❌ 拒絕虛無假設 H₀")
    else:
        st.success("✅ 無法拒絕虛無假設 H₀")

    # 畫圖
    x = np.linspace(-4, 4, 500)
    y = stats.norm.pdf(x, 0, 1)
    fig, ax = plt.subplots()
    ax.plot(x, y, label="Z 檢定 分布曲線")
    ax.axvline(z, color="red", linestyle="--", label=f"Z={z:.2f}")
    ax.axvline(critical, color="blue", linestyle="--", label=f"正向臨界值={critical:.2f}")
    ax.axvline(-critical, color="blue", linestyle="--", label=f"負向臨界值={-critical:.2f}")
    ax.set_title("Z 檢定 分布與檢定結果")
    ax.legend()
    st.pyplot(fig)

# ========== t 檢定 ==========
elif test_type == "t 檢定":
    x_bar = st.number_input("樣本均值 (x̄)", value=5.0)
    mu_0 = st.number_input("假設母體均值 (μ₀)", value=5.0)
    s = st.number_input("樣本標準差 (s)", value=1.0)
    n = st.number_input("樣本數 (n)", value=10, step=1)

    se = s / np.sqrt(n)
    t = (x_bar - mu_0) / se
    p_value = 2 * (1 - stats.t.cdf(abs(t), df=n-1))
    critical = stats.t.ppf(1 - alpha/2, df=n-1)

    st.subheader("檢定結果")
    st.write(f"t 統計量 = {t:.3f}")
    st.write(f"p-value = {p_value:.3f}")
    if abs(t) > critical:
        st.error("❌ 拒絕虛無假設 H₀")
    else:
        st.success("✅ 無法拒絕虛無假設 H₀")

    # 畫圖
    x = np.linspace(-4, 4, 500)
    y = stats.t.pdf(x, df=n-1)
    fig, ax = plt.subplots()
    ax.plot(x, y, label="t 分布曲線")
    ax.axvline(t, color="red", linestyle="--", label=f"t={t:.2f}")
    ax.axvline(critical, color="blue", linestyle="--", label=f"正向臨界值={critical:.2f}")
    ax.axvline(-critical, color="blue", linestyle="--", label=f"負向臨界值={-critical:.2f}")
    ax.set_title("t 檢定 分布與檢定結果")
    ax.legend()
    st.pyplot(fig)

# ========== 卡方檢定 ==========
elif test_type == "卡方檢定 (Chi-square)":
    observed = st.text_input("輸入觀測值 (以逗號分隔)", "10, 20, 30")
    expected = st.text_input("輸入期望值 (以逗號分隔)", "15, 15, 30")

    if st.button("執行卡方檢定"):
        obs = np.array(list(map(float, observed.split(","))))
        exp = np.array(list(map(float, expected.split(","))))

        chi2, p_value = stats.chisquare(obs, f_exp=exp)
        df = len(obs) - 1
        critical = stats.chi2.ppf(1 - alpha, df)

        st.subheader("檢定結果")
        st.write(f"卡方統計量 χ² = {chi2:.3f}, 自由度 = {df}")
        st.write(f"p-value = {p_value:.3f}")
        if chi2 > critical:
            st.error("❌ 拒絕虛無假設 H₀")
        else:
            st.success("✅ 無法拒絕虛無假設 H₀")

# ========== F 檢定 ==========
elif test_type == "F 檢定 (變異數比)":
    var1 = st.number_input("樣本 1 變異數", value=2.0)
    n1 = st.number_input("樣本 1 樣本數", value=10, step=1)
    var2 = st.number_input("樣本 2 變異數", value=1.0)
    n2 = st.number_input("樣本 2 樣本數", value=10, step=1)

    F = var1 / var2
    df1, df2 = n1 - 1, n2 - 1
    p_value = 1 - stats.f.cdf(F, df1, df2)
    critical = stats.f.ppf(1 - alpha, df1, df2)

    st.subheader("檢定結果")
    st.write(f"F 統計量 = {F:.3f}, df1={df1}, df2={df2}")
    st.write(f"p-value = {p_value:.3f}")
    if F > critical:
        st.error("❌ 拒絕虛無假設 H₀")
    else:
        st.success("✅ 無法拒絕虛無假設 H₀")
