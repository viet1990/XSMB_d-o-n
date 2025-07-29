import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
from collections import Counter
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

DATA_FILE = "xsmb_history.csv"

# -----------------------------
# Hàm xử lý dữ liệu đầu vào
# -----------------------------
def clean_last_2_digits(numbers):
    return [str(n)[-2:].zfill(2) for n in numbers if str(n).strip().isdigit()]

# -----------------------------
# Hàm tính xác suất
# -----------------------------
def compute_probabilities(df):
    all_numbers = df.iloc[:, 1:].values.flatten()
    all_numbers = clean_last_2_digits(all_numbers)
    counter = Counter(all_numbers)
    total = sum(counter.values())
    prob_df = pd.DataFrame(counter.items(), columns=["Loto", "Count"])
    prob_df["Probability (%)"] = prob_df["Count"] / total * 100
    prob_df = prob_df.sort_values(by="Probability (%)", ascending=False).reset_index(drop=True)
    return prob_df

# -----------------------------
# Hàm phân tích chu kỳ xuất hiện
# -----------------------------
def compute_cycle_analysis(df):
    numbers = df.iloc[:, 1:]
    flat = clean_last_2_digits(numbers.values.flatten())
    cycles = {}
    last_seen = {}

    for i, row in enumerate(df.itertuples(index=False)):
        day_numbers = clean_last_2_digits(row[1:])
        for num in day_numbers:
            if num in last_seen:
                diff = i - last_seen[num]
                if num in cycles:
                    cycles[num].append(diff)
                else:
                    cycles[num] = [diff]
            last_seen[num] = i

    result = []
    for num, diffs in cycles.items():
        avg_cycle = round(sum(diffs) / len(diffs), 2)
        result.append((num, avg_cycle))

    cycle_df = pd.DataFrame(result, columns=["Loto", "Avg Cycle"])
    cycle_df = cycle_df.sort_values(by="Avg Cycle")
    return cycle_df

# -----------------------------
# Hàm tính số ngày chưa ra
# -----------------------------
def days_since_last_seen(num, df):
    reversed_df = df.iloc[::-1]
    for i, row in enumerate(reversed_df.itertuples(index=False)):
        day_numbers = clean_last_2_digits(row[1:])
        if num in day_numbers:
            return i
    return len(df)

# -----------------------------
# Chuẩn hóa
# -----------------------------
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# -----------------------------
# Huấn luyện mô hình AI Voting
# -----------------------------
def train_ensemble_model(df):
    all_data = []
    for i in range(30, len(df)):
        past_df = df.iloc[i-30:i]
        today_numbers = clean_last_2_digits(df.iloc[i, 1:].tolist())
        prob_df = compute_probabilities(past_df)
        cycle_df = compute_cycle_analysis(past_df)

        merged = pd.merge(prob_df, cycle_df, on="Loto", how="inner")
        merged["LastSeen"] = merged["Loto"].apply(lambda x: days_since_last_seen(x, past_df))
        merged["NormProb"] = normalize(merged["Probability (%)"])
        merged["NormCycle"] = normalize(1 / merged["Avg Cycle"])
        merged["NormLast"] = normalize(merged["LastSeen"])
        merged["Label"] = merged["Loto"].apply(lambda x: 1 if x in today_numbers else 0)

        all_data.append(merged[["NormProb", "NormCycle", "NormLast", "Label"]])

    full_data = pd.concat(all_data).dropna()
    X = full_data[["NormProb", "NormCycle", "NormLast"]]
    y = full_data["Label"]

    lr = LogisticRegression()
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model = VotingClassifier(estimators=[('lr', lr), ('rf', rf)], voting='soft')
    model.fit(X, y)
    return model

# -----------------------------
# Dự đoán Top 4 dùng model
# -----------------------------
def suggest_top4_with_model(prob_df, cycle_df, df, model):
    recent_day = clean_last_2_digits(df.iloc[-1, 1:].dropna().tolist())
    merged = pd.merge(prob_df, cycle_df, on="Loto", how="inner")
    merged["LastSeen"] = merged["Loto"].apply(lambda x: days_since_last_seen(x, df))
    merged = merged[~merged["Loto"].isin(recent_day)]
    merged["NormProb"] = normalize(merged["Probability (%)"])
    merged["NormCycle"] = normalize(1 / merged["Avg Cycle"])
    merged["NormLast"] = normalize(merged["LastSeen"])

    X = merged[["NormProb", "NormCycle", "NormLast"]]
    merged["ModelScore"] = model.predict_proba(X)[:, 1]
    return merged.sort_values(by="ModelScore", ascending=False)["Loto"].head(4).tolist()

# -----------------------------
# Gợi ý xiên từ Top 4
# -----------------------------
def suggest_xiens_from_top4(top4):
    xiens = {
        "Xiên 2": list(combinations(top4, 2)),
        "Xiên 3": list(combinations(top4, 3)),
        "Xiên 4": [tuple(top4)],
    }
    return xiens

# -----------------------------
# Giao diện Streamlit
# -----------------------------
st.set_page_config(page_title="🔢 XSMB Predictor", layout="centered")
st.title("🎯 Dự đoán XSMB & Gợi ý Xiên")

uploaded_file = st.file_uploader("📂 Tải lên file dữ liệu 300 ngày đầu (CSV)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.to_csv(DATA_FILE, index=False)
    st.success("✅ Đã lưu dữ liệu ban đầu!")

if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)

    st.subheader("📅 Nhập kết quả mới nhất (27 số)")
    cols = st.columns(9)
    new_numbers = []
    for i in range(27):
        with cols[i % 9]:
            num = st.text_input(f"G{i+1}", key=f"num_{i}")
            new_numbers.append(num.strip())

    if st.button("📌 Lưu kết quả hôm nay"):
        last_date = pd.to_datetime(df.iloc[-1, 0])
        today = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        cleaned = clean_last_2_digits(new_numbers)
        if len([today] + cleaned) == len(df.columns):
            df.loc[len(df)] = [today] + cleaned
            df.to_csv(DATA_FILE, index=False)
            st.success(f"✅ Đã lưu kết quả ngày {today}!")
            st.experimental_rerun()
        else:
            st.error(f"❌ Số lượng số nhập ({len(cleaned)}) không khớp với dữ liệu ({len(df.columns) - 1} số).")

    st.subheader("📅 Lịch sử 10 ngày gần nhất")
    df_sorted = df.sort_values(by=df.columns[0], ascending=False).reset_index(drop=True)
    st.dataframe(df_sorted.head(10))

    if not df.empty:
        prob_df = compute_probabilities(df)
        cycle_df = compute_cycle_analysis(df)
        model = train_ensemble_model(df)

        st.subheader("📊 Xác suất các cặp số")
        st.dataframe(prob_df.head(20))

        st.subheader("🔁 Phân tích chu kỳ xuất hiện")
        st.dataframe(cycle_df.head(20))

        st.subheader("🌟 Dự đoán 4 con số mạnh nhất hôm nay (AI Voting)")
        predicted_top4 = suggest_top4_with_model(prob_df, cycle_df, df, model)
        for i, num in enumerate(predicted_top4, 1):
            st.write(f"{i}. {num}")

        st.subheader("🎯 Gợi ý Xiên từ Top 4")
        xiens = suggest_xiens_from_top4(predicted_top4)
        for xi_type, pairs in xiens.items():
            st.markdown(f"**{xi_type}:**")
            for p in pairs:
                sub_df = prob_df[prob_df['Loto'].isin(p)]
                total_prob = sub_df['Probability (%)'].sum()
                st.write(f"{' - '.join(p)} | Tổng xác suất: {total_prob:.2f}%")

        st.download_button("⬇️ Tải toàn bộ dữ liệu hiện tại", df.to_csv(index=False), "xsmb_current.csv")
