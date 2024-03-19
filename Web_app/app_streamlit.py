import streamlit as st
import datetime as dt
import pandas as pd
import numpy as np
import seaborn as sns
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pickle
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

matplotlib.use('Agg')

html_temp = """
		<div style="background-color:{};padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Customer segmentation using K-means </h1>
		<h5 style="color:white;text-align:center;">RMF Model</h5>
		</div>
		"""

descriptive_message_temp ="""
	<div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:black;padding:10px">Defination</h3>
		<p>Today as the competition between marketing companies, retail stores, banks to attract newer customers and maintain the old ones is in its peak. So every company is  having their own business strategies to achieve this. But instead the traditional “one size fits all” approach they are trying to have the customer segmentation approach in order to have upper hand in competition. So Our project is based on such segmentation or customer clustering method where we will collect , analyze, process and visualize the customer’s data and build a data science model which will help in forming clusters or segments of customers using the k-means clustering algorithm and RFM model (Recency Frequency Monetary) for already existing customers. At the very simple the customer clusters would be like super customer, intermediate customers, customers on the verge of churning out .The input data can be demographic (age,gender..etc.), behavioral, economic .So using this strategy  it will be easy to target customers accordingly and achieve business strength by maintaining good relationship with the customers </p>
	</div>
	"""

def get_date(x):
    return dt.datetime(x.year, x.month, 1)

def get_date_int(dataframe, column):
    year = dataframe[column].dt.year
    month = dataframe[column].dt.month
    day = dataframe[column].dt.day
    return year, month, day

def check_skew(df, column):
    skew = stats.skew(df[column])
    skewtest = stats.skewtest(df[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df[column])
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return

def join_rfm(x):
    return str(x['Recency_Q']) + str(x['Frequency_Q']) + str(x['MonetaryValue_Q'])

def segment_me(df):
    if df['RFM_Score'] > 10:
        return 'A Class'
    elif (df['RFM_Score'] >= 5) and (df['RFM_Score'] < 8):
        return 'B Class'
    elif (df['RFM_Score'] >= 8) and (df['RFM_Score'] < 11):
        return 'C Class'
    else:
        return 'D Class'


def get_quantile(df, column, start_n_quantiles, end_n_quantiles, step=1):
    category_label = range(start_n_quantiles, end_n_quantiles, step)

    quantiles = pd.qcut(df[column], q=abs(
        end_n_quantiles - start_n_quantiles),duplicates='drop',labels=False)

    df = df.assign(name=quantiles.values)

    new_column_name = column + '_Q'

    return df.rename(columns={"name": new_column_name})

def preprocess(df):
    df['Date'] = df['InvoiceDate'].dt.date
    df['Time'] = df['InvoiceDate'].dt.time

    df.drop(['InvoiceDate'], axis=1, inplace=True)

    for i in range(df.shape[0]):
        if (str(df.loc[i, 'InvoiceNo']).isnumeric()):
            df.loc[i, 'CancelledOrder'] = None
            df.loc[i, 'Invoice_No'] = df.loc[i, 'InvoiceNo']
        elif (str(df.loc[i, 'InvoiceNo']).isalnum()):
            df.loc[i, 'CancelledOrder'] = str(df.loc[i, 'InvoiceNo'])[0]
            df.loc[i, 'Invoice_No'] = str(df.loc[i, 'InvoiceNo'])[1:]
    df.drop(['InvoiceNo'], axis=1, inplace=True)
    df['CancelledOrder'] = df['CancelledOrder'].astype('category')
    df['CancelledOrder'] = df['CancelledOrder'].cat.add_categories([0])
    df['CancelledOrder'].fillna(value=0, inplace=True)
    df['CancelledOrder'].replace(to_replace='C', value=1, inplace=True)
    df['StockCode'] = df.StockCode.astype('category')
    df['TotalSum'] = df['Quantity'] * df['UnitPrice']
    df = df[pd.notnull(df['CustomerID'])]
    df['InvoiceMonth'] = df['Date'].apply(get_date)
    grouping = df.groupby('CustomerID')['InvoiceMonth']
    df['CohortMonth'] = grouping.transform('min')

    invoice_year, invoice_month, invoice_day = get_date_int(df, 'InvoiceMonth')
    cohort_year, cohort_month, cohort_day = get_date_int(df, 'CohortMonth')

    years_diff = invoice_year - cohort_year
    months_diff = invoice_month - cohort_month
    days_diff = invoice_day - cohort_day

    df['CohortIndex'] = (years_diff * 12 + months_diff + 1)

    return df


def optimal_kmeans(dataset, start=2, end=11):
    n_clu = []
    km_ss = []
    inertia = []

    for n_clusters in range(start, end):

        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(dataset)
        silhouette_avg = round(silhouette_score(dataset, labels, random_state=1), 3)
        inertia_score = round(kmeans.inertia_, 2)
        km_ss.append(silhouette_avg)
        n_clu.append(n_clusters)
        inertia.append(inertia_score)

        print("No. Clusters: {}, Silhouette Score(SS): {}, SS Delta: {}, Inertia: {}, Inertia Delta: {}".format(
            n_clusters,
            silhouette_avg,
            (km_ss[n_clusters - start] - km_ss[n_clusters - start - 1]).round(3),
            inertia_score,
            (inertia[n_clusters - start] - inertia[n_clusters - start - 1]).round(3)))

        if n_clusters == end - 1:
            plt.figure(figsize=(9, 6))

            plt.subplot(2, 1, 1)
            plt.title('Within-Cluster Sum-of-Squares / Inertia')
            sns.pointplot(x=n_clu, y=inertia)

            plt.subplot(2, 1, 2)
            plt.title('Silhouette Score')
            sns.pointplot(x=n_clu, y=km_ss)
            plt.tight_layout()
            plt.show()
            st.pyplot()
    return


def kmeans(normalised_df_rfm, clusters_number, original_df_rfm):
    kmeans = KMeans(n_clusters=clusters_number, random_state=1)
    kmeans.fit(normalised_df_rfm)

    cluster_labels = kmeans.labels_

    df_new = original_df_rfm.assign(Cluster=cluster_labels)

    model = TSNE(random_state=1)
    transformed = model.fit_transform(df_new)

    plt.title('Flattened Graph of {} Clusters'.format(clusters_number))
    sns.scatterplot(x=transformed[:, 0], y=transformed[:, 1], hue=cluster_labels, style=cluster_labels, palette="Set1")
    st.pyplot()
    return df_new


def rfm_values(df):
    df_new = df.groupby(['Cluster']).agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'MonetaryValue': ['mean', 'count']
    }).round(0)

    return df_new


def cluster_display(normalised_df_rfm, clusters_number, custid,ind):
    kmeans = KMeans(n_clusters=clusters_number, random_state=1)
    kmeans.fit(normalised_df_rfm)

    cluster_labels = kmeans.labels_
    print(normalised_df_rfm.index)
    st.write("The Given customerID belong to "+cluster_labels[ind] + "with total "+ clusters_number +"clusters")
    return

def main():


    st.markdown(html_temp.format('royalblue'), unsafe_allow_html=True)
    menu = ["Home", "Model"]
    sub_menu = ["Data Visualization","RFM Model","K-Means"]

    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        st.subheader("Home")
        st.text("Why Customer Segmentation?")
        st.markdown(descriptive_message_temp, unsafe_allow_html=True)



    elif choice == "Model":

        file1 = st.file_uploader(label="Please upload your file.",
                                 type=['csv', 'xlsx'])
        if file1 is not None:
            st.subheader("Your Dataset")
            print(file1)
            try:
                df = pd.read_excel(file1)
            except Exception as e:
                print(e)
                df = pd.read_csv(file1)
        try:
            st.write(df)
        except Exception as e:
            print(e)
            st.write("Please upload file")

        activity = st.selectbox("Activity", sub_menu)

        if activity == "Data Visualization":
            st.set_option('deprecation.showPyplotGlobalUse', False)

            df = preprocess(df)

            if st.checkbox("Data Visualize"):
                st.subheader("No of. Customers in Different Countries ")
                df.Country.value_counts()[:10].plot(kind='bar')
                st.pyplot()
                st.subheader("Heatmap of Data")
                sns.heatmap(df.drop("CustomerID", axis=1).corr(), annot=True)
                st.pyplot()
                st.subheader("No of. Customers active per month")
                df.InvoiceMonth.value_counts().plot(kind='bar')
                st.pyplot()
                st.subheader("Cancelled Order")
                a = list(df.CancelledOrder)
                labels = ["Not Cancelled", "Cancelled"]
                plt.pie([a.count(0), a.count(1)], labels=labels, autopct='%1.0f%%')
                st.pyplot()

                st.subheader("Histogram of Total price per transaction")
                ig, ax = plt.subplots(1, 1)
                a = list(df.TotalSum)
                ax.hist(a, bins=[-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80])
                ax.set_xlabel('Total Sum')
                plt.show()
                st.pyplot()


        elif activity == "RFM Model":
            st.subheader("RFM Model Analysis")
            df = pd.read_excel('E:\College\TY\Edi\sample2.xlsx')
            df = preprocess(df)
            earliest_date = df['Date'].min()
            end_date = df['Date'].max()
            start_date = end_date - pd.to_timedelta(40, unit='d')
            df_rfm = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
            snapshot_date = end_date + dt.timedelta(days=1)
            df_rfm = df_rfm.groupby(['CustomerID']).agg({
                'Date': lambda x: (snapshot_date - x.max()).days,
                'Invoice_No': 'count',
                'TotalSum': 'sum'})


            df_rfm.rename(columns={'Date': 'Recency', 'Invoice_No': 'Frequency', 'TotalSum': 'MonetaryValue'}, inplace=True)
            df_rfm_quantile = df_rfm.copy()

            df_rfm_quantile = get_quantile(df_rfm_quantile, 'Recency', 4, 0,-1)
            df_rfm_quantile = get_quantile(df_rfm_quantile, 'Frequency', 1, 5,1)
            df_rfm_quantile = get_quantile(df_rfm_quantile, 'MonetaryValue', 1, 5,2)
            df_rfm_quantile['RFM_Segment'] = df_rfm_quantile.apply(join_rfm, axis=1)
            df_rfm_quantile['RFM_Score'] = df_rfm_quantile[['Recency_Q', 'Frequency_Q', 'MonetaryValue_Q']].sum(axis=1)

            df_rfm_quantile['Customer_Class'] = df_rfm_quantile.apply(segment_me, axis=1)

            df_rfm_custom_segment = df_rfm_quantile.groupby('Customer_Class').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'MonetaryValue': ['mean', 'count']
            }).round(1)
            df_rfm_custom_segment

            if st.checkbox("Distrubition Graph of RMF Model"):
                plt.figure(figsize=(9, 9))

                plt.subplot(3, 1, 1)
                check_skew(df_rfm, 'Recency')

                plt.subplot(3, 1, 2)
                check_skew(df_rfm, 'Frequency')

                plt.subplot(3, 1, 3)
                check_skew(df_rfm, 'MonetaryValue')

                plt.tight_layout()
                plt.savefig('before_transform.png', format='png', dpi=1000)
                st.pyplot()

            df_rfm_log = df_rfm.copy()
            df_rfm_log['MonetaryValue'] = (df_rfm_log['MonetaryValue'] - df_rfm_log['MonetaryValue'].min()) + 1

            if st.checkbox("Normilized Distrubition Graph of RMF Model"):
                df_rfm_log = np.log(df_rfm_log)

                plt.figure(figsize=(9, 9))

                plt.subplot(3, 1, 1)
                check_skew(df_rfm_log, 'Recency')

                plt.subplot(3, 1, 2)
                check_skew(df_rfm_log, 'Frequency')

                plt.subplot(3, 1, 3)
                check_skew(df_rfm_log, 'MonetaryValue')

                plt.tight_layout()
                plt.savefig('after_transform.png', format='png', dpi=1000)
                st.pyplot()

        elif activity == "K-Means":
            st.subheader("K-Means Clustering")
            df = pd.read_excel('E:\College\TY\Edi\sample2.xlsx')
            df = preprocess(df)
            earliest_date = df['Date'].min()
            end_date = df['Date'].max()
            start_date = end_date - pd.to_timedelta(40, unit='d')
            df_rfm = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
            snapshot_date = end_date + dt.timedelta(days=1)
            df_rfm = df_rfm.groupby(['CustomerID']).agg({
                'Date': lambda x: (snapshot_date - x.max()).days,
                'Invoice_No': 'count',
                'TotalSum': 'sum'})

            df_rfm.rename(columns={'Date': 'Recency', 'Invoice_No': 'Frequency', 'TotalSum': 'MonetaryValue'},
                          inplace=True)
            df_rfm_quantile = df_rfm.copy()

            df_rfm_quantile = get_quantile(df_rfm_quantile, 'Recency', 4, 0, -1)
            df_rfm_quantile = get_quantile(df_rfm_quantile, 'Frequency', 1, 5, 1)
            df_rfm_quantile = get_quantile(df_rfm_quantile, 'MonetaryValue', 1, 5, 2)
            df_rfm_quantile['RFM_Segment'] = df_rfm_quantile.apply(join_rfm, axis=1)
            df_rfm_quantile['RFM_Score'] = df_rfm_quantile[['Recency_Q', 'Frequency_Q', 'MonetaryValue_Q']].sum(axis=1)

            df_rfm_quantile['Customer_Class'] = df_rfm_quantile.apply(segment_me, axis=1)

            df_rfm_custom_segment = df_rfm_quantile.groupby('Customer_Class').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'MonetaryValue': ['mean', 'count']
            }).round(1)


            df_rfm_log = df_rfm.copy()
            df_rfm_log['MonetaryValue'] = (df_rfm_log['MonetaryValue'] - df_rfm_log['MonetaryValue'].min()) + 1

            scaler = StandardScaler()
            scaler.fit(df_rfm_log)
            df_rfm_normal = scaler.transform(df_rfm_log)

            df_rfm_normal = pd.DataFrame(df_rfm_normal, index=df_rfm_log.index, columns=df_rfm_log.columns)
            if st.checkbox("Elbow Method"):
                optimal_kmeans(df_rfm_normal)
            if st.checkbox("Clustering Graph"):
                plt.figure(figsize=(9, 9))

                plt.subplot(3, 1, 1)
                df_rfm_k3 = kmeans(df_rfm_normal, 3, df_rfm)

                plt.subplot(3, 1, 2)
                df_rfm_k4 = kmeans(df_rfm_normal, 4, df_rfm)

                plt.subplot(3, 1, 3)
                df_rfm_k5 = kmeans(df_rfm_normal, 5, df_rfm)

                plt.tight_layout()
                plt.savefig('flattened.png', format='png', dpi=1000)

            if st.checkbox("Analysis of average RFM values and size for each cluster"):
                st.write("Average RFM values and size for 3 cluster")
                st.write(rfm_values(df_rfm_k3))
                st.write("Average RFM values and size for 4 cluster")
                st.write(rfm_values(df_rfm_k4))
                st.write("Average RFM values and size for 5 cluster")
                st.write(rfm_values(df_rfm_k5))

            custid = st.text_input("Please Enter CustomerID to find out which Cluster it Belongs")
            if custid is not None:
                custid = float(custid)
                a = list(df_rfm_normal.index)
                b=a.index(custid)
                cluster_display(df_rfm_normal, 3, custid,b)
                cluster_display(df_rfm_normal, 4, custid,b)
                cluster_display(df_rfm_normal, 5, custid,b)



if __name__ == '__main__':

	main()