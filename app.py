import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model from the file
kmeans_model = joblib.load('KMeanClusteringML/bmx_kmean.joblib')

st.title('K-Means Clustering')

# Upload the dataset and save as csv
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    clustering_columns = ['bmxleg','bmxwaist']
    data = pd.read_csv(uploaded_file)
    data = data.dropna(subset=clustering_columns)
    if data.shape[0] > kmeans_model.n_clusters:
        # Select the columns to be used for clustering
        # X = data[['bmxleg','bmxwaist']].dropna()

        # Predict the cluster for each data point
        clusters = kmeans_model.predict(data[clustering_columns])

        # Add cluster labels to the DataFrame
        data['cluster'] = clusters
        st.write(data)
        #Plot the clusters
        st.set_option('deprecation.showPyplotGlobalUse',False)
        st.write("Scatterplot of BMXLEG and BMXWAIST")
        plt.figure(figsize=(10,6))
        sns.scatterplot(data=data, x='bmxleg', y='bmxwaist',hue='cluster')
        plt.title("KMeans Clustering")
        st.pyplot()
    else:
        st.write("Not Enough data points for clustering")