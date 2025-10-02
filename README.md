# 📊 統計檢定小工具 (Z / t / 卡方 / F 檢定)

這是一個使用 **Streamlit** 開發的互動式統計檢定工具，支援：
- Z 檢定 (Z-test)
- t 檢定 (t-test)
- 卡方檢定 (Chi-square test)
- F 檢定 (F-test)

## 🚀 功能特色
- 輸入參數即可自動計算統計量與 p-value
- 直觀的檢定分布圖，標示臨界值與檢定統計量
- 自動生成檢定結論
- 附加「統計小抄」說明各檢定的使用時機

## 📂 專案結構

stat_test_app/
│── app.py
│── requirements.txt
│── README.md

## ▶️ 執行方法
```bash
pip install -r requirements.txt
streamlit run app.py
