#import necessary libraries
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

#page setup
st.set_page_config(page_icon="👥",page_title="CUSTOMER SEGMENTATION",layout="wide")

#load the data
st.subheader('UPLOAD DATA')
st.write('(required column- Annual Income (k$), Spending Score (1-100))')
file = st.file_uploader(" ",type=["csv"])
df = None
if file:
    df = pd.read_csv(file)


with st.sidebar:
    st.title("CUSTOMER SEGMENTATION")
    st.image('https://cdn-icons-png.flaticon.com/512/7111/7111143.png')
    if df is not None:
        features = st.multiselect("Select Features: ",options=df.columns, default=["Annual Income (k$)","Spending Score (1-100)"]) #static for these column names only
        df = df.loc[:,features]

def preprocessing(df):
    #encoding
    encoder = LabelEncoder()
    for col in df.columns:
        if df[col].dtype==object:
            df[col] = encoder.fit_transform(df[col])

def elbow():
    out = []
    k_values = range(1,11)

    for i in k_values:
        model = KMeans(n_clusters=i)
        model.fit(df)
        out.append(model.inertia_)

    KL = KneeLocator(k_values,out,curve="convex",direction="decreasing")
    df1 = pd.DataFrame({"K_val":k_values,"inertia":out})

    st.subheader("ELBOW CURVE")
    fig = st.line_chart(data=df1,x="K_val",y="inertia")

    return KL.elbow

if df is not None:
    st.subheader("Samples of the data uploaded for visualization and clustering-")
    st.write(df.sample(10))

    preprocessing(df) 

    #optimized K value
    K = elbow()

    #model training
    model = KMeans(n_clusters=K)
    model.fit(df)
    labels = model.labels_
    df["clusters"] = labels

    #visualization
    st.subheader("CLUSTERED DATA")

    st.scatter_chart(data=df,x="Annual Income (k$)",y="Spending Score (1-100)",color="clusters")
