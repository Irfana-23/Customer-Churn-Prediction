import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from pygments.styles.dracula import background
from streamlit import columns
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR / "churn-bigml-20.csv")

st.set_page_config(layout="wide")
#st.sidebar.title('churn')
st.title("📈 Dashboard")
st.subheader("Dataset Overview")
#df=pd.read_csv('../churn-bigml-20.csv')
# data overview
def card(title, value, color, emoji):
    st.markdown(f"""
    <div style="
        background-color:#1E1E1E;
        padding:18px;
        border-radius:15px;
        border-left:6px solid {color};
        box-shadow:2px 2px 10px rgba(0,0,0,0.4);
        text-align:center;
    ">
        <h4>{emoji} {title}</h4>
        <h2 style='color:{color};'>{value}</h2>
    </div>
    """, unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    total = len(df)
    card("Customers", total, "#00BCD4", "👥")
with col2:

        total_features = df.shape[1] - 1
        card(" Features:", total_features,"#FFC107", "📑")
with col3:
            churn_customers = df['Churn'].sum()
            card("Churn:", churn_customers,"#F44336", "❌")
with col4:
             non_churn_customers = total - churn_customers
             card("Non-Churn:", non_churn_customers,"#4CAF50", "✅")
with col5:
               churn_rate = round((churn_customers / total) * 100,2)
               card("Churn Rate:", churn_rate,"#9C27B0", "📉")
#st.divider()


# churn distribution
# ---------------- Churn Distribution & Dataset Preview ----------------
st.divider()
col1, col2 = st.columns([1, 2])

with col1:

    st.subheader("Churn Distribution")

    counts = df["Churn"].value_counts()
    fig, ax = plt.subplots(figsize=(3, 3))

    wedges, texts, autotexts = ax.pie(
        [counts[True], counts[False]],
        labels=None,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#FF4B4B", "#00C853"],
        explode=(0.08, 0),
        radius=0.8,
        textprops={"fontsize":8, "color":"white"}
    )


    # Legend at the top
    ax.legend(
        wedges,
        ["Churn", "Non-Churn"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        fontsize=9,
        frameon=False,
        labelcolor="white"
    )

    ax.set_aspect("equal")

    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    st.pyplot(fig, use_container_width=True)

with col2:

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(9),
        use_container_width=True,
        height=380
    )
     #st.pyplot(fig)

st.subheader("Model Performance")
col1,col2,col3,col4=columns(4)
with col1:
     st.metric("Accuracy", "96.7%")
     with col2:
         st.metric("Precision", "95.4%")
         with col3:
             st.metric("Recall", "94.8%")
             with col4:
                 st.metric("F1 Score", "95.1%")