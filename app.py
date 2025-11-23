import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns               # For enhanced plots
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("StudentPerformanceFactors.csv")

# Page config
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📚",
    layout="wide"
)

# Sidebar Branding
st.sidebar.title("EduAnalytics")
st.sidebar.markdown("Smart Student Performance Prediction Platform")

# Sidebar Navigation
page = st.sidebar.selectbox("Navigation", [
    "Home",
    "About",
    "Dataset Insights",
    "Exploratory Analysis",
    "Predictive Model Performance",
    "Exam Score Estimation"
])

# HOME PAGE
if page == "Home":
    st.markdown("<h1 style='text-align: center; color: black; font-size: 48px; font-weight: bold;'>Welcome to EduAnalytics</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; font-size: 20px;'>
        This platform uses real-world data and machine learning to predict student exam performance based on study habits, attendance, motivation, and more.  
        Explore insights, train models, and make informed predictions — all in one interactive dashboard.
    </div>
    """, unsafe_allow_html=True)

# ABOUT PAGE
elif page == "About":
    st.markdown("<h2 style='text-align: center; color: black; font-size: 36px; font-weight: bold;'>About EduAnalytics</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; font-size: 18px;'>
        This app helps predict student exam scores using data like study hours, attendance, and motivation.  
        It guides you through exploring the data, building a model, checking its accuracy, and making predictions — all in one easy-to-use dashboard.
    </div>
    """, unsafe_allow_html=True)

# DATASET INSIGHTS
elif page == "Dataset Insights":
    import seaborn as sns

    st.markdown("<h2 style='text-align: center; color: black; font-size: 32px; font-weight: bold;'>Dataset Insights</h2>", unsafe_allow_html=True)
    # 🔹 First 20 Records
    st.subheader(" First 20 Records")
    st.dataframe(df.head(20))

    # 🔹 Summary Statistics
    st.subheader(" Summary Statistics")
    st.write(df.describe())


    # 🔹 Grouped Stats: Motivation Level
    st.subheader(" Average Exam Score by Motivation Level")
    avg_motivation = df.groupby("Motivation_Level")["Exam_Score"].mean().reset_index()
    st.dataframe(avg_motivation)

    fig1, ax1 = plt.subplots()
    sns.barplot(x="Motivation_Level", y="Exam_Score", data=df, ax=ax1, palette="Blues")
    ax1.set_title("Average Exam Score by Motivation Level")
    ax1.set_xlabel("Motivation Level")
    ax1.set_ylabel("Average Exam Score")
    st.pyplot(fig1)

    # 🔹 Grouped Stats: School Type (if available)
    if "School_Type" in df.columns:
        st.subheader(" Average Exam Score by School Type")
        avg_school = df.groupby("School_Type")["Exam_Score"].mean().reset_index()
        st.dataframe(avg_school)

        fig2, ax2 = plt.subplots()
        sns.barplot(x="School_Type", y="Exam_Score", data=df, ax=ax2, palette="Greens")
        ax2.set_title("Average Exam Score by School Type")
        ax2.set_xlabel("School Type")
        ax2.set_ylabel("Average Exam Score")
        st.pyplot(fig2)

# EXPLORATORY ANALYSIS
elif page == "Exploratory Analysis":
    import seaborn as sns

    st.markdown("<h2 style='text-align: center; color: black; font-size: 32px; font-weight: bold;'>Exploratory Analysis: Key Questions</h2>", unsafe_allow_html=True)
    # Question 1
    st.markdown(" **1. Does studying more hours improve exam scores?**")
    fig1, ax1 = plt.subplots()
    sns.regplot(x=df["Hours_Studied"], y=df["Exam_Score"], ax=ax1,
                scatter_kws={"alpha":0.5}, line_kws={"color":"red"})
    ax1.set_title("Hours Studied vs Exam Score")
    ax1.set_xlabel("Hours Studied")
    ax1.set_ylabel("Exam Score")
    st.pyplot(fig1)
    st.caption("Graph: Regression plot showing correlation between study hours and exam scores.")

    # Question 2
    st.markdown(" **2. Does attendance percentage affect exam performance?**")
    fig2, ax2 = plt.subplots()
    sns.regplot(x=df["Attendance"], y=df["Exam_Score"], ax=ax2,
                scatter_kws={"alpha":0.5}, line_kws={"color":"green"})
    ax2.set_title("Attendance vs Exam Score")
    ax2.set_xlabel("Attendance (%)")
    ax2.set_ylabel("Exam Score")
    st.pyplot(fig2)
    st.caption("Graph: Regression plot showing impact of attendance on exam scores.")

    # Question 3
    st.markdown(" **3. Does sleep duration influence exam performance?**")
    fig3, ax3 = plt.subplots()
    sns.regplot(x=df["Sleep_Hours"], y=df["Exam_Score"], ax=ax3,
                scatter_kws={"alpha":0.5}, line_kws={"color":"purple"})
    ax3.set_title("Sleep Hours vs Exam Score")
    ax3.set_xlabel("Sleep Hours")
    ax3.set_ylabel("Exam Score")
    st.pyplot(fig3)
    st.caption("Graph: Regression plot showing relationship between sleep and performance.")

    # Optional: Correlation Matrix
    st.markdown(" **4. Overall correlation between numeric features**")
    numeric_cols = df.select_dtypes(include=np.number).columns
    corr = df[numeric_cols].corr()
    fig4, ax4 = plt.subplots(figsize=(8,6))
    im = ax4.imshow(corr, cmap="Blues")
    ax4.set_xticks(range(len(numeric_cols)))
    ax4.set_yticks(range(len(numeric_cols)))
    ax4.set_xticklabels(numeric_cols, rotation=90)
    ax4.set_yticklabels(numeric_cols)
    fig4.colorbar(im, ax=ax4)
    ax4.set_title("Correlation Matrix")
    fig4.tight_layout()
    st.pyplot(fig4)
    st.caption("Graph: Heatmap showing correlations among numeric features.")
# PREDICTIVE MODEL PERFORMANCE
elif page == "Predictive Model Performance":
    st.markdown("<h2 style='text-align: center; color: black; font-size: 32px; font-weight: bold;'>Train & Evaluate Model</h2>", unsafe_allow_html=True)

    X = df[["Hours_Studied"]]
    y = df["Exam_Score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{rmse:.2f}")
    col2.metric("R² Score", f"{r2:.4f}")
    col3.metric("Intercept", f"{model.intercept_:.2f}")
    st.write("Slope:", model.coef_[0])
    st.success("Model trained successfully!")

# EXAM SCORE ESTIMATION
elif page == "Exam Score Estimation":
    st.markdown("<h2 style='text-align: center; color: black; font-size: 32px; font-weight: bold;'>Predict Exam Score</h2>", unsafe_allow_html=True)
    study_hours = st.number_input("Enter Hours Studied:", min_value=0, max_value=12, value=5)

    # Train model
    X = df[["Hours_Studied"]]
    y = df["Exam_Score"]
    model = LinearRegression()
    model.fit(X, y)

    # Predict and display score
    if st.button("Estimate Score"):
        new_data = pd.DataFrame({"Hours_Studied": [study_hours]})
        score = model.predict(new_data)[0]
        st.success(f"Estimated Exam Score: {score:.2f}")
