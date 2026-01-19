import streamlit as st
import pandas as pd
import os
from datetime import date

FILE_NAME = "mood_data.csv"
COLUMNS = ["Date", "Mood", "SleepHours", "Stress", "Energy", "ScreenTime", "Exercise", "Notes"]

# st.subheader("🗑️ Manage Data")

# if st.button("Clear All Mood Logs"):
#     df = pd.DataFrame(columns=COLUMNS)
#     df.to_csv(FILE_NAME, index=False)
#     st.success("✅ All mood logs cleared!")
#     st.rerun()

st.set_page_config(page_title="MoodGuard", page_icon="🧠", layout="centered")

st.title("🧠 MoodGuard - Mental Health Mood Tracker")

st.sidebar.title("🧠 MoodGuard Menu")

page = st.sidebar.radio(
    "Go to",
    [
        "➕ Add Entry",
        "📊 Dashboard",
        "🚨 Burnout Risk",
        "🧾 Recovery Plan",
        "📝 Sentiment Analysis",
        "📄 PDF Report",
        "🗑️ Manage Data"
    ]
)

st.write("Log your daily mood + habits and track patterns over time ✅")

# ✅ Create file if not exists
if not os.path.exists(FILE_NAME):
    df = pd.DataFrame(columns=[
        "Date", "Mood", "SleepHours", "Stress", "Energy", "ScreenTime", "Exercise", "Notes"
    ])
    df.to_csv(FILE_NAME, index=False)

# ✅ Load data
df = pd.read_csv(FILE_NAME)

st.subheader("🗑️ Manage Data")

if st.button("Clear All Mood Logs", key="clear_logs_btn"):
    df = pd.DataFrame(columns=COLUMNS)
    df.to_csv(FILE_NAME, index=False)
    st.success("✅ All mood logs cleared!")
    st.rerun()


st.subheader("📌 Add Today's Entry")

with st.form("mood_form"):
    entry_date = st.date_input("Date", value=date.today())
    mood = st.slider("Mood (1 = worst, 10 = best)", 1, 10, 5)
    sleep = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)
    stress = st.slider("Stress Level (1 = low, 10 = high)", 1, 10, 5)
    energy = st.slider("Energy Level (1 = low, 10 = high)", 1, 10, 5)
    screen_time = st.number_input("Screen Time (hours)", min_value=0.0, max_value=24.0, value=4.0)
    exercise = st.selectbox("Did you exercise today?", ["No", "Yes"])
    notes = st.text_area("Notes (optional)", placeholder="Write 1-2 lines about your day...")

    submitted = st.form_submit_button("✅ Save Entry")

    if submitted:
        new_entry = {
            "Date": entry_date.strftime("%Y-%m-%d"),
            "Mood": mood,
            "SleepHours": sleep,
            "Stress": stress,
            "Energy": energy,
            "ScreenTime": screen_time,
            "Exercise": exercise,
            "Notes": notes
        }

        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        df.to_csv(FILE_NAME, index=False)

        st.success("✅ Entry saved successfully!")

st.subheader("📊 Your Mood Logs")

if df.empty:
    st.info("No data yet. Add your first entry above ✅")
else:
    st.dataframe(df.tail(10), use_container_width=True)
import plotly.express as px

st.subheader("📈 Mood Analytics Dashboard")

if not df.empty:
    # convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # sort by date
    df = df.sort_values("Date")

    # ✅ Mood Trend Graph
    fig1 = px.line(df, x="Date", y="Mood", markers=True, title="Mood Trend Over Time")
    st.plotly_chart(fig1, use_container_width=True)

    # ✅ Weekly Summary
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    weekly_avg = df.groupby("Week")["Mood"].mean().reset_index()

    fig2 = px.bar(weekly_avg, x="Week", y="Mood", title="Weekly Average Mood")
    st.plotly_chart(fig2, use_container_width=True)

    # ✅ Correlation Insights
    st.subheader("🔍 Mood Relationship Insights")

    col1, col2 = st.columns(2)

    with col1:
        fig3 = px.scatter(df, x="SleepHours", y="Mood", title="Mood vs Sleep Hours", trendline="ols")
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        fig4 = px.scatter(df, x="Stress", y="Mood", title="Mood vs Stress Level", trendline="ols")
        st.plotly_chart(fig4, use_container_width=True)

    # ✅ Best day / Worst day
    best_day = df.loc[df["Mood"].idxmax()]
    worst_day = df.loc[df["Mood"].idxmin()]

    st.subheader("✅ Quick Highlights")
    st.success(f"🌟 Best Day: {best_day['Date'].date()} | Mood: {best_day['Mood']}")
    st.error(f"⚠️ Worst Day: {worst_day['Date'].date()} | Mood: {worst_day['Mood']}")
else:
    st.info("Add some entries to see analytics ✅")

st.subheader("🚨 Burnout Risk Score (Today)")

if not df.empty:
    latest = df.sort_values("Date").iloc[-1]

    mood = latest["Mood"]
    sleep = latest["SleepHours"]
    stress = latest["Stress"]
    energy = latest["Energy"]
    screen = latest["ScreenTime"]

    # ✅ Simple burnout score logic (0 to 100)
    burnout_score = 0

    if mood <= 4:
        burnout_score += 25
    if sleep < 6:
        burnout_score += 25
    if stress >= 7:
        burnout_score += 25
    if energy <= 4:
        burnout_score += 15
    if screen >= 7:
        burnout_score += 10

    burnout_score = min(burnout_score, 100)

    st.metric("Burnout Risk Score", f"{burnout_score}/100")

    if burnout_score <= 30:
        st.success("✅ Low Risk: You are doing good! Keep it up 💚")
    elif burnout_score <= 70:
        st.warning("⚠️ Medium Risk: Take small breaks + improve sleep 💛")
    else:
        st.error("🚨 High Risk: You may be burning out. Take rest + reduce stress ❤️")

else:
    st.info("Add entries to calculate burnout risk ✅")

from sklearn.ensemble import IsolationForest

st.subheader("🕵️ Anomaly Detection (Stress/Mood Alert)")

if len(df) >= 7:  # needs some data
    temp = df.copy()

    # Convert Exercise to 0/1
    temp["Exercise"] = temp["Exercise"].map({"No": 0, "Yes": 1})

    # Features for anomaly detection
    features = temp[["Mood", "SleepHours", "Stress", "Energy", "ScreenTime", "Exercise"]]

    # Train Isolation Forest
    iso = IsolationForest(contamination=0.15, random_state=42)
    temp["Anomaly"] = iso.fit_predict(features)  # -1 = anomaly, 1 = normal

    latest_row = temp.sort_values("Date").iloc[-1]

    if latest_row["Anomaly"] == -1:
        st.error("🚨 Alert: Today looks unusual compared to your normal pattern!")
        st.write("Possible reasons: high stress, low mood, low sleep, or too much screen time.")
    else:
        st.success("✅ Today looks normal compared to your past pattern.")
else:
    st.info("Add at least 7 entries to enable anomaly detection ✅")
st.subheader("🧾 Personalized Recovery Plan (AI Suggestions)")

if not df.empty:
    latest = df.sort_values("Date").iloc[-1]

    mood = latest["Mood"]
    sleep = latest["SleepHours"]
    stress = latest["Stress"]
    energy = latest["Energy"]
    screen = latest["ScreenTime"]
    exercise = latest["Exercise"]

    plan = []

    # ✅ Sleep suggestions
    if sleep < 6:
        plan.append("😴 Sleep goal: Try to get **7–8 hours** tonight (low sleep detected).")
    else:
        plan.append("✅ Sleep goal: Keep maintaining **7–8 hours** sleep.")

    # ✅ Stress suggestions
    if stress >= 7:
        plan.append("🧘 Stress fix: Do **5–10 min deep breathing** or a short meditation.")
        plan.append("📵 Try to take **a break from phone** for 30 minutes.")
    else:
        plan.append("✅ Stress level looks okay today. Keep taking small breaks.")

    # ✅ Screen time suggestions
    if screen >= 7:
        plan.append("📱 Screen time: Reduce screen time by **1–2 hours** tomorrow.")
    else:
        plan.append("✅ Screen time is under control 👍")

    # ✅ Energy suggestions
    if energy <= 4:
        plan.append("⚡ Energy boost: Drink water + take a **10 min walk**.")
    else:
        plan.append("✅ Energy level is good today!")

    # ✅ Exercise suggestion
    if exercise == "No":
        plan.append("🏃 Activity: Add a small workout (even **10 mins stretching** is enough).")
    else:
        plan.append("✅ Great! You exercised today 💪")

    # ✅ Mood suggestions
    if mood <= 4:
        plan.append("💛 Mood support: Talk to a friend / write journal / listen to calm music.")
    else:
        plan.append("🌟 Mood is stable today. Keep going!")

    st.write("### ✅ Your Recovery Plan for Tomorrow:")
    for i, step in enumerate(plan, start=1):
        st.write(f"**{i}.** {step}")

else:
    st.info("Add entries to generate recovery plan ✅")
from textblob import TextBlob

st.subheader("📝 Notes Sentiment Analysis (NLP)")

if not df.empty:
    latest_note = str(df.sort_values("Date").iloc[-1]["Notes"]).strip()

    if latest_note == "" or latest_note.lower() == "nan":
        st.info("No notes written for the latest entry. Add notes to analyze sentiment ✅")
    else:
        polarity = TextBlob(latest_note).sentiment.polarity  # -1 to +1

        if polarity > 0.1:
            sentiment = "😊 Positive"
            st.success(f"Sentiment: {sentiment}")
        elif polarity < -0.1:
            sentiment = "😟 Negative"
            st.error(f"Sentiment: {sentiment}")
        else:
            sentiment = "😐 Neutral"
            st.warning(f"Sentiment: {sentiment}")

        st.write(f"**Sentiment Score:** `{polarity:.2f}`")
        st.write(f"**Your Note:** {latest_note}")
else:
    st.info("Add entries to analyze notes sentiment ✅")

from fpdf import FPDF
from datetime import datetime

st.subheader("📄 Download Weekly Report (PDF)")

if not df.empty:
    # Ensure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Select last 7 days data
    last7 = df.sort_values("Date").tail(7)

    avg_mood = last7["Mood"].mean()
    avg_sleep = last7["SleepHours"].mean()
    avg_stress = last7["Stress"].mean()

    if st.button("📥 Generate Weekly PDF Report", key="pdf_btn"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=14)

        pdf.cell(200, 10, txt="MoodGuard - Weekly Mood Report", ln=True, align="C")
        pdf.ln(10)

        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.ln(5)

        pdf.cell(200, 10, txt=f"Average Mood (Last 7 Days): {avg_mood:.2f}", ln=True)
        pdf.cell(200, 10, txt=f"Average Sleep Hours: {avg_sleep:.2f}", ln=True)
        pdf.cell(200, 10, txt=f"Average Stress Level: {avg_stress:.2f}", ln=True)
        pdf.ln(8)

        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Last 7 Days Logs:", ln=True)
        pdf.ln(5)

        pdf.set_font("Arial", size=10)
        for _, row in last7.iterrows():
            line = f"{row['Date'].date()} | Mood: {row['Mood']} | Sleep: {row['SleepHours']} | Stress: {row['Stress']} | Energy: {row['Energy']}"
            pdf.multi_cell(0, 8, txt=line)

        pdf_file = "weekly_report.pdf"
        pdf.output(pdf_file)

        # ✅ Download Button
        with open(pdf_file, "rb") as f:
            st.download_button(
                label="✅ Download Report",
                data=f,
                file_name="MoodGuard_Weekly_Report.pdf",
                mime="application/pdf",
                key="download_pdf_btn"
            )
else:
    st.info("Add at least 1 entry to generate report ✅")
import joblib

st.subheader("🤖 ML Burnout Risk Prediction")

if not df.empty:
    try:
        model = joblib.load("burnout_model.pkl")

        latest = df.sort_values("Date").iloc[-1].copy()
        latest["Exercise"] = 1 if latest["Exercise"] == "Yes" else 0

        X_latest = [[
            latest["Mood"],
            latest["SleepHours"],
            latest["Stress"],
            latest["Energy"],
            latest["ScreenTime"],
            latest["Exercise"]
        ]]

        pred = model.predict(X_latest)[0]
        prob = model.predict_proba(X_latest)[0][1]  # probability of burnout=1

        st.metric("Burnout Probability", f"{prob*100:.2f}%")

        if pred == 1:
            st.error("🚨 ML Prediction: High Burnout Risk")
        else:
            st.success("✅ ML Prediction: Low Burnout Risk")

    except:
        st.warning("⚠️ Train model first by running: python train_model.py")
else:
    st.info("Add entries to get ML burnout prediction ✅")
