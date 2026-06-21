import streamlit as st
import pandas as pd
import plotly.express as px
from groq import Groq
import base64
from sklearn.base import BaseEstimator, TransformerMixin

def style_chart(fig):

    fig.update_layout(
        template="plotly_dark",

        paper_bgcolor="rgba(15,23,42,0.75)",
        plot_bgcolor="rgba(15,23,42,0.35)",

        font=dict(
            color="white",
            size=14
        ),

        title_font=dict(
            size=22,
            color="white"
        ),

        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.15)",
            tickfont=dict(color="white")
        ),

        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.15)",
            tickfont=dict(color="white")
        )
    )

    return fig

st.set_page_config(
    page_title="Flight Delay Prediction",
    layout="wide"
)
st.markdown("""
<style>

/* Sidebar Background */
section[data-testid="stSidebar"]{
    background: rgba(0,0,0,0.08) !important;
    backdrop-filter: blur(4px);
}

/* Sidebar Text */
section[data-testid="stSidebar"] *{
    color: white !important;
}

/* Selectbox Labels */
section[data-testid="stSidebar"] label{
    color: black !important;
    font-weight: bold;
}

/* Header */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3{
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

section[data-testid="stSidebar"] div[data-baseweb="select"] *{
    color: black !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Tabs */
.stTabs [data-baseweb="tab-list"]{
    gap: 15px;
}

.stTabs [data-baseweb="tab"]{
    background: rgba(15,23,42,0.75);
    border: 1px solid rgba(59,130,246,0.4);
    border-radius: 12px;
    color: white !important;
    font-size: 18px;
    font-weight: 600;
    padding: 10px 25px;
}

.stTabs [aria-selected="true"]{
    background: linear-gradient(
        90deg,
        #2563eb,
        #1d4ed8
    ) !important;

    color: white !important;

    border: 1px solid #60a5fa !important;

    box-shadow: 0px 0px 15px rgba(37,99,235,0.5);
}

</style>
""", unsafe_allow_html=True)

def add_bg():

    with open("Airplane.jpg", "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>

        .stApp {{
            background:
                linear-gradient(
                    rgba(0,0,0,0.5),
                    rgba(0,0,0,0.5)
                ),
                url("data:image/jpg;base64,{encoded}");

            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

add_bg()
@st.cache_data
def load_data():
    return pd.read_parquet("Clean_flight_Data.parquet")

df = load_data()
dataset_summary = f"""
Total Flights: {len(df):,}

Average Delay: {df['ARR_DELAY'].mean():.2f} minutes

Number of Airlines:
{df['AIRLINE_CODE'].nunique()}

Number of Airports:
{df['ORIGIN'].nunique()}

Most Delayed Airline:
{df.groupby('AIRLINE_CODE')['ARR_DELAY'].mean().idxmax()}

Most Delayed Route:
{df.groupby('ROUTE')['ARR_DELAY'].mean().idxmax()}
"""

@st.cache_data
def create_route_lookup(df):

    return (
        df.groupby(["ORIGIN","DEST"])
        .agg({
            "DISTANCE":"median",
            "CRS_ELAPSED_TIME":"median"
        })
        .reset_index()
    )

route_lookup = create_route_lookup(df)

route_lookup["ROUTE_DISPLAY"] = (
    route_lookup["ORIGIN"]
    + " ➜ "
    + route_lookup["DEST"]
)
class HistoricalFeaturesTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        temp = X.copy()
        temp['ARR_DELAY'] = y

        self.global_avg_ = y.mean()

        self.airline_avg_ = temp.groupby('AIRLINE_CODE')['ARR_DELAY'].mean().to_dict()
        self.origin_avg_ = temp.groupby('ORIGIN')['ARR_DELAY'].mean().to_dict()
        self.dest_avg_ = temp.groupby('DEST')['ARR_DELAY'].mean().to_dict()
        self.month_avg_ = temp.groupby('MONTH_NUM')['ARR_DELAY'].mean().to_dict()
        self.hour_avg_ = temp.groupby('DEP_HOUR')['ARR_DELAY'].mean().to_dict()

        self.route_freq_ = temp['ROUTE'].value_counts().to_dict()
        self.origin_freq_ = temp['ORIGIN'].value_counts().to_dict()
        self.dest_freq_ = temp['DEST'].value_counts().to_dict()
        self.airline_freq_ = temp['AIRLINE_CODE'].value_counts().to_dict()

        return self

    def transform(self, X):

        X = X.copy()

        X['AIRLINE_AVG_DELAY'] = X['AIRLINE_CODE'].map(self.airline_avg_).fillna(self.global_avg_)
        X['ORIGIN_AVG_DELAY'] = X['ORIGIN'].map(self.origin_avg_).fillna(self.global_avg_)
        X['DEST_AVG_DELAY'] = X['DEST'].map(self.dest_avg_).fillna(self.global_avg_)
        X['MONTH_AVG_DELAY'] = X['MONTH_NUM'].map(self.month_avg_).fillna(self.global_avg_)
        X['DEP_HOUR_AVG_DELAY'] = X['DEP_HOUR'].map(self.hour_avg_).fillna(self.global_avg_)

        X['ROUTE_FREQ'] = X['ROUTE'].map(self.route_freq_).fillna(0)
        X['ORIGIN_FREQ'] = X['ORIGIN'].map(self.origin_freq_).fillna(0)
        X['DEST_FREQ'] = X['DEST'].map(self.dest_freq_).fillna(0)
        X['AIRLINE_FREQ'] = X['AIRLINE_CODE'].map(self.airline_freq_).fillna(0)

        def get_season(month):
            if month in [12,1,2]:
                return 'Winter'
            elif month in [3,4,5]:
                return 'Spring'
            elif month in [6,7,8]:
                return 'Summer'
            else:
                return 'Fall'

        X['SEASON'] = X['MONTH_NUM'].apply(get_season)

        return X
import joblib

with st.spinner(
    "Predicting Flight Delay..."
):

    try:

        st.write("Before Historical")

        x1 = model.named_steps["historical_features"].transform(pred_df)

        st.write("After Historical")

        st.write(x1)

        st.stop()

    except Exception as e:

        st.error(str(e))
        st.stop()

def get_season(month):
    if month in [12,1,2]:
        return "Winter"
    elif month in [3,4,5]:
        return "Spring"
    elif month in [6,7,8]:
        return "Summer"
    else:
        return "Fall"

df['SEASON'] = df['MONTH_NUM'].apply(get_season)

st.sidebar.markdown("""
#  Dashboard Filters

Use the filters below to explore flight delay patterns.
""")

selected_year = st.sidebar.selectbox(
    "Year",
    ["All"] + sorted(df['YEAR'].unique().tolist())
)

selected_airline = st.sidebar.selectbox(
    "Airline",
    ["All"] + sorted(df['AIRLINE_CODE'].unique().tolist())
)

selected_month = st.sidebar.selectbox(
    "Month",
    ["All"] + sorted(df['MONTH_NUM'].unique().tolist())
)

selected_season = st.sidebar.selectbox(
    "Season",
    ["All"] + sorted(df['SEASON'].unique().tolist())
)

filtered_df = df.copy()

if selected_year != "All":
    filtered_df = filtered_df[
        filtered_df['YEAR'] == selected_year
    ]

if selected_airline != "All":
    filtered_df = filtered_df[
        filtered_df['AIRLINE_CODE'] == selected_airline
    ]

if selected_month != "All":
    filtered_df = filtered_df[
        filtered_df['MONTH_NUM'] == selected_month
    ]

if selected_season != "All":
    filtered_df = filtered_df[
        filtered_df['SEASON'] == selected_season
    ]


st.markdown("""
<div style="
background: rgba(15,23,42,0.85);
border: 2px solid rgba(59,130,246,0.7);
padding: 35px;
border-radius: 25px;
text-align:center;
margin-bottom:20px;
backdrop-filter: blur(10px);
box-shadow: 0px 4px 20px rgba(0,0,0,0.5);
">

<h1 style="
color:white;
font-size:48px;
margin-bottom:10px;
">
Flight Delay Analytics & Prediction
</h1>

<p style="
color:#cbd5e1;
font-size:20px;
">
Analyze flight delays across airlines, airports, routes and seasons
using Machine Learning & XGBoost
</p>

</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "Analytics Dashboard",
    "ML Prediction",
    "AI Assistant"
])

with tab1:
    st.markdown("""
            <h2 style='color:white'>
            Dashboard Overview
            </h2>
            """, unsafe_allow_html=True)
    

    # ================= KPIs =================

    total_flights = len(filtered_df)

    avg_delay = round(filtered_df['ARR_DELAY'].mean(), 2)

    delay_rate = round(
        (filtered_df['ARR_DELAY'] > 15).mean() * 100,
        2
    )

    total_airlines = filtered_df['AIRLINE_CODE'].nunique()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div style="
            background: rgba(15,23,42,0.85);
            border: 2px solid #3b82f6;
            border-radius: 20px;
            padding: 20px;
            text-align:center;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.5);
        ">
            <h4 style="color:white;">✈️ Total Flights</h4>
            <h1 style="color:#60a5fa;">{total_flights:,}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="
            background: rgba(15,23,42,0.85);
            border: 2px solid #10b981;
            border-radius: 20px;
            padding: 20px;
            text-align:center;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.5);
        ">
            <h4 style="color:white;"> Avg Delay</h4>
            <h1 style="color:#10b981;">{avg_delay}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="
            background: rgba(15,23,42,0.85);
            border: 2px solid #f59e0b;
            border-radius: 20px;
            padding: 20px;
            text-align:center;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.5);
        ">
            <h4 style="color:white;"> Delay Rate</h4>
            <h1 style="color:#f59e0b;">{delay_rate}%</h1>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style="
            background: rgba(15,23,42,0.85);
            border: 2px solid #ef4444;
            border-radius: 20px;
            padding: 20px;
            text-align:center;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.5);
        ">
            <h4 style="color:white;"> Airlines</h4>
            <h1 style="color:#ef4444;">{total_airlines}</h1>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")

    # ================= Airlines =================
    st.markdown("""
        <h2 style='color:white'>
        Airline Performance Analysis
        </h2>
        """, unsafe_allow_html=True)
    airline_delay = (
        filtered_df.groupby('AIRLINE_CODE')['ARR_DELAY']
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig_airline = px.bar(
        airline_delay,
        x='AIRLINE_CODE',
        y='ARR_DELAY',
        title='Top 10 Airlines by Average Delay',
        labels={
            'AIRLINE_CODE': 'Airline',
            'ARR_DELAY': 'Average Delay (Minutes)'
        }
    )
    fig_airline = style_chart(fig_airline)

    st.plotly_chart(fig_airline, use_container_width=True)

    # ================= Row 1 =================

    col1, col2 = st.columns(2)
    st.markdown("""
        <h2 style='color:white'>
        Seasonal Analysis
        </h2>
        """, unsafe_allow_html=True)
    with col1:

        month_delay = (
            filtered_df.groupby('MONTH_NUM')['ARR_DELAY']
            .mean()
            .reset_index()
        )

        fig_month = px.line(
            month_delay,
            x='MONTH_NUM',
            y='ARR_DELAY',
            markers=True,
            title='Average Delay by Month'
        )
        fig_month = style_chart(fig_month)

        st.plotly_chart(fig_month, use_container_width=True)

    with col2:

        season_delay = (
            filtered_df.groupby('SEASON')['ARR_DELAY']
            .mean()
            .reset_index()
        )

        fig_season = px.bar(
            season_delay,
            x='SEASON',
            y='ARR_DELAY',
            title='Average Delay by Season'
        )
        fig_season = style_chart(fig_season)

        st.plotly_chart(fig_season, use_container_width=True)

    # ================= Row 2 =================

    col3, col4 = st.columns(2)
    st.markdown("""
        <h2 style='color:white'>
        Operational Insights
        </h2>
        """, unsafe_allow_html=True)
    with col3:

        hour_delay = (
            filtered_df.groupby('DEP_HOUR')['ARR_DELAY']
            .mean()
            .reset_index()
        )

        fig_hour = px.line(
            hour_delay,
            x='DEP_HOUR',
            y='ARR_DELAY',
            markers=True,
            title='Average Delay by Departure Hour'
        )
        fig_hour = style_chart(fig_hour)

        st.plotly_chart(fig_hour, use_container_width=True)

    with col4:

        route_delay = (
            filtered_df.groupby('ROUTE')['ARR_DELAY']
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig_route = px.bar(
            route_delay,
            x='ROUTE',
            y='ARR_DELAY',
            title='Top 10 Most Delayed Routes'
        )
        fig_route = style_chart(fig_route)

        st.plotly_chart(fig_route, use_container_width=True)

    # ================= Distribution =================

    fig_dist = px.histogram(
        filtered_df,
        x='ARR_DELAY',
        nbins=50,
        title='Arrival Delay Distribution'
    )
    fig_dist = style_chart(fig_dist)

    st.plotly_chart(fig_dist, use_container_width=True)
    pivot_table = (
        filtered_df.pivot_table(
            values='ARR_DELAY',
            index='MONTH_NUM',
            columns='YEAR',
            aggfunc='mean'
        )
    )

    fig_heat = px.imshow(
        pivot_table,
        text_auto=True,
        aspect="auto",
        title="Average Delay by Month and Year"
    )
    fig_heat = style_chart(fig_heat)
    st.plotly_chart(fig_heat, use_container_width=True)
with tab2:

    st.markdown("""
    <style>

    .prediction-card{
        background: rgba(15,23,42,0.80);
        padding: 25px;
        border-radius: 20px;
        border: 1px solid rgba(59,130,246,0.4);
        backdrop-filter: blur(8px);
        margin-top:15px;
    }

    label{
        color:white !important;
        font-size:18px !important;
        font-weight:600 !important;
    }

    .stMarkdown,
    .stMarkdown p,
    .stMarkdown li,
    .stMarkdown span,
    h1,h2,h3,h4,h5,h6{
        color:white !important;
    }

    /* Success Box */
    .stSuccess,
    .stSuccess *{
        color:white !important;
    }

    /* Info Box */
    .stInfo,
    .stInfo *{
        color:white !important;
    }

    /* Metric Cards */
    [data-testid="stMetric"]{
        background: rgba(15,23,42,0.80);
        border: 1px solid rgba(59,130,246,0.4);
        border-radius: 15px;
        padding: 15px;
    }

    [data-testid="stMetric"] *{
        color:white !important;
    }

    /* Metric Labels */
    [data-testid="stMetricLabel"]{
        color:white !important;
    }

    /* Metric Values */
    [data-testid="stMetricValue"]{
        color:white !important;
    }

    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <h2 style='color:white'>
    Flight Delay Prediction
    </h2>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        airline = st.selectbox(
            "Airline",
            sorted(df['AIRLINE_CODE'].unique())
        )

        selected_route = st.selectbox(
            "Route",
            sorted(route_lookup["ROUTE_DISPLAY"].unique())
        )

        selected_row = route_lookup[
            route_lookup["ROUTE_DISPLAY"] == selected_route
        ].iloc[0]

        origin = selected_row["ORIGIN"]
        dest = selected_row["DEST"]

        distance = int(selected_row["DISTANCE"])
        elapsed = int(selected_row["CRS_ELAPSED_TIME"])

        month = st.selectbox(
            "Month",
            sorted(df['MONTH_NUM'].unique())
        )

        day_name = st.selectbox(
            "Day",
            sorted(df["DAY_NAME"].unique())
        )

        day = df[
            df["DAY_NAME"] == day_name
        ]["DAY_NUM"].iloc[0]

    with col2:

        times = []

        for h in range(24):

            times.append(
                pd.Timestamp(f"{h}:00")
                .strftime("%I:%M %p")
            )

        selected_time = st.selectbox(
            "Departure Time",
            times,
            index=12
        )

        dep_hour = times.index(selected_time)

    route = f"{origin}-{dest}"

    st.markdown(f"""
    <div class="prediction-card">

    <h3 style="color:white;">
    Flight Summary
    </h3>

    <p style="color:white;font-size:18px;">

    <b>Airline:</b> {airline}<br>

    <b>Route:</b> {origin} ➜ {dest}<br>

    <b>Departure:</b> {selected_time}<br>

    <b>Distance:</b> {distance:,} Miles<br>

    <b>Duration:</b> {elapsed} Minutes<br>

    <b>Month:</b> {month}

    </p>

    </div>
    """, unsafe_allow_html=True)

    if st.button(
        "Predict Delay",
        use_container_width=True
    ):

        dep_time = dep_hour * 100

        arr_time = dep_time + elapsed

        is_weekend = 1 if day in [6,7] else 0

        pred_df = pd.DataFrame({

            'AIRLINE_CODE': [airline],
            'ORIGIN': [origin],
            'DEST': [dest],
            'CRS_DEP_TIME': [dep_time],
            'CRS_ARR_TIME': [arr_time],
            'CRS_ELAPSED_TIME': [elapsed],
            'DISTANCE': [distance],
            'MONTH_NUM': [month],
            'DAY_NUM': [day],
            'IS_WEEKEND': [is_weekend],
            'ROUTE': [route],
            'DEP_HOUR': [dep_hour]

        })

        with st.spinner(
            "Predicting Flight Delay..."
        ):
        
            st.write(pred_df)
            st.stop()

        delay_minutes = round(prediction)

        if prediction <= 15:

            color = "#10b981"
            status = "🟢 On Time"

            message = """
            Flight is expected to arrive on time.
            """

        elif prediction <= 45:

            color = "#f59e0b"
            status = "🟡 Moderate Delay"

            message = """
            Minor operational delay may occur.
            """

        else:

            color = "#ef4444"
            status = "🔴 High Delay"

            message = """
            Significant delay expected.
            Consider alternative plans.
            """

        st.markdown(
            f"<h3 style='color:white'>{status}</h3>",
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Delay",
                f"{delay_minutes} Minutes"
            )

        with col2:
            st.metric(
                "Flight Status",
                status.replace("🟢 ", "")
                    .replace("🟡 ", "")
                    .replace("🔴 ", "")
            )

        st.markdown(
            f"<p style='color:white;font-size:18px'>{message}</p>",
            unsafe_allow_html=True
        )

        st.subheader("Key Factors Used By Model")

        st.markdown("""
        - Airline Historical Performance
        - Route Historical Delay
        - Departure Time
        - Seasonality
        - Flight Distance
        - Airport Congestion
        """)
with tab3:

    st.markdown("""
    <h2 style='color:white'>
    AI Project Assistant
    </h2>
    """, unsafe_allow_html=True)

    try:

        with open(
            "project_summary.md",
            "r",
            encoding="utf-8"
        ) as f:

            project_summary = f.read()

    except:

        project_summary = "Project documentation not found."

    dataset_summary = f"""

    Total Flights: {len(df):,}

    Average Delay:
    {df['ARR_DELAY'].mean():.2f} minutes

    Number of Airlines:
    {df['AIRLINE_CODE'].nunique()}

    Number of Airports:
    {df['ORIGIN'].nunique()}

    Most Delayed Airline:
    {
        df.groupby('AIRLINE_CODE')
        ['ARR_DELAY']
        .mean()
        .idxmax()
    }

    Most Delayed Route:
    {
        df.groupby('ROUTE')
        ['ARR_DELAY']
        .mean()
        .idxmax()
    }

    """

    SYSTEM_PROMPT = f"""
    You are an AI assistant for a Flight Delay Prediction Project.

    Use the following project documentation:

    {project_summary}

    Dataset Insights:

    {dataset_summary}

    Answer questions about the project,
    dashboard, machine learning model,
    dataset insights and business problem.

    If information is unavailable,
    say so honestly.
    """

    client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
    )

    if "chat_messages" not in st.session_state:

        st.session_state.chat_messages = []

    for msg in st.session_state.chat_messages:

        with st.chat_message(msg["role"]):

            st.write(msg["content"])

    prompt = st.chat_input(
        "Ask me anything about the project..."
    )

    if prompt:

        st.session_state.chat_messages.append(
            {
                "role":"user",
                "content":prompt
            }
        )

        with st.chat_message("user"):

            st.write(prompt)

        with st.spinner(
            "Thinking..."
        ):

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role":"system",
                        "content":SYSTEM_PROMPT
                    }
                ] + st.session_state.chat_messages,
                temperature=0.2
            )

            answer = (
                response
                .choices[0]
                .message
                .content
            )

        st.session_state.chat_messages.append(
            {
                "role":"assistant",
                "content":answer
            }
        )

        with st.chat_message("assistant"):

            st.write(answer)
