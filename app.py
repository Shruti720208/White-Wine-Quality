import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# App title
st.title("🍷 Wine Quality Prediction App")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("winequality-white.csv", sep=";")

df = load_data()
st.subheader("Dataset Sample")
st.dataframe(df.head())

# Features and target
X = df.drop("quality", axis=1)
y = df["quality"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and show metrics
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"Accuracy: **{accuracy:.2%}**")

# Pie chart of predictions
quality_counts = pd.Series(y_pred).value_counts().sort_index()
fig, ax = plt.subplots()
ax.pie(quality_counts, labels=quality_counts.index, autopct='%1.1f%%',
       startangle=140, colors=sns.color_palette("pastel"))
ax.set_title("Predicted Wine Quality Distribution")
ax.axis('equal')
st.pyplot(fig)

# Optional: Custom input form
st.subheader("🔍 Predict Custom Wine Quality")
input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(col, value=float(X[col].mean()))

if st.button("Predict Quality"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted wine quality: **{prediction}**")
