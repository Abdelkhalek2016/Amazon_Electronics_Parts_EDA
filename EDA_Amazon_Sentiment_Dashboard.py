import streamlit as st
import pandas as pd
import plotly.express as plx
from collections import Counter
import ast
from wordcloud import WordCloud
import bz2

# Page 1 - EDA Project Information
def page_1():
    #st.subheader("Amazon Review Text Sentiment Analysis")
    st.markdown("""<b style='font-size:26px'>Project Overview</b>\n
In this data-driven project, we dive into the vast world of Amazon customer reviews to uncover valuable insights hidden within text data.\nOur main objective is to perform sentiment analysis on these reviews, extracting the emotional tone and polarity of each customer's feedback.

<b style='font-size:26px'>Our Goals</b>\n
<b style='font-size:22px'>Sentiment Analysis</b>: Develop machine learning models to classify reviews as positive, negative, or neutral based on their content.

<b style='font-size:22px'>Exploratory Data Analysis (EDA)</b>:</b> Conduct a comprehensive EDA to gain a deep understanding of the dataset, including trends, patterns, and key statistics.

<b style='font-size:22px'>Dashboard Visualization</b>: Create an interactive dashboard using Streamlit and Plotly to present our findings and allow users to explore the data visually.

<b style='font-size:22px'>Timeline Analysis</b>: Visualize the evolution of sentiments over time by analyzing trends in customer reviews.

<b style='font-size:26px'>Dataset</b>\n
Our dataset comprises a large collection of Amazon reviews for electronic products. Each review contains various attributes, such as reviewer ID, product ASIN, review text, overall rating, and more.

<b style='font-size:26px'>Project Workflow</b>\n
<b style='font-size:22px'>Data Preprocessing</b>: Cleanse and prepare the dataset for analysis by handling missing values, text preprocessing, and feature engineering.

<b style='font-size:22px'>Exploratory Data Analysis (EDA)</b>: Explore the dataset to uncover patterns, trends, and anomalies. This phase involves data visualization, statistical analysis, and data summarization.

<b style='font-size:22px'>Sentiment Analysis Models</b>: Build and evaluate machine learning models for sentiment analysis, including Natural Language Processing (NLP) techniques.

<b style='font-size:22px'>Dashboard Creation</b>: Develop an interactive dashboard using Streamlit and Plotly to present the project's findings and insights.

<b style='font-size:22px'>Timeline Analysis</b>: Investigate how sentiments change over time by creating dynamic timeline charts.

<b style='font-size:26px'>Tools and Technologies</b>\n
<b style='font-size:22px'>Python</b>: For data analysis, machine learning, and dashboard development.\n
<b style='font-size:22px'>Pandas, NumPy</b>: For data manipulation and preprocessing.\n
<b style='font-size:22px'>Scikit-Learn</b>: For machine learning model development.\n
<b style='font-size:22px'>Natural Language Toolkit (NLTK) or spaCy</b>: For NLP tasks.\n
<b style='font-size:22px'>Streamlit and Plotly</b>: For creating the interactive dashboard.\n
<b style='font-size:26px'>Project Deliverables</b>\n
Interactive Streamlit dashboard with multiple tabs.\n
Machine learning models for sentiment analysis.\n
EDA reports and visualizations.\n
Timeline charts showing sentiment trends.\n
<b style='font-size:26px'>Conclusion</b>\n
By the end of this project, we aim to provide a comprehensive analysis of Amazon customer reviews for electronic products,\n helping businesses gain insights into customer sentiment and improve their products and services.

Stay tuned for updates and exciting visualizations as we dive deeper into the world of Amazon reviews!""",unsafe_allow_html=True)

# Page 2 - Timeline Charts
def page_2(df):
    st.subheader("Timeline Charts")

    col1,col2=st.columns(spec=[2,1],gap='large')
    with col1:
        
        # Distribution of reviews across different years
        distribution_reviews_by_years=df['year'].value_counts().reset_index()
        fig=plx.histogram(data_frame=df,x='year',title='Distribution of reviews across different years')
        st.plotly_chart(fig)
        
        
        # Trend of number of reviews across months
        distribution_reviews_by_months=df.groupby('monthName')['reviewerID'].count().sort_values(ascending=False).reset_index().rename(columns={'reviewerID':'count_reviewes'}).sort_values(by='monthName',ascending=True)
        sort_month_mapper={'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}
        distribution_reviews_by_months['month_number']=distribution_reviews_by_months['monthName'].map(sort_month_mapper)
        distribution_reviews_by_months=distribution_reviews_by_months.sort_values(by='month_number')
        fig=plx.line(distribution_reviews_by_months,x='monthName',y='count_reviewes',title="Trend of number of reviews across months")
        st.plotly_chart(fig)
        
        
        # Trend of number of reviews across years
        number_reviews_years=df.groupby('year')['reviewerID'].count().sort_values(ascending=False).reset_index().rename(columns={'reviewerID':'count_reviewes'}).sort_values(by='year',ascending=True)
        fig=plx.line(number_reviews_years,x='year',y='count_reviewes',title="Trend of number of reviews across years")
        st.plotly_chart(fig)
        
        # Number of Reviews class (positive,Negative,Neutral) across days
        reviewes_positive_negative=df.groupby(['dayName','review_classes'])['reviewerID'].count().reset_index().rename(columns={'reviewerID':'reviewes_count'})
        fig=plx.bar(reviewes_positive_negative,x='dayName',y='reviewes_count',color='review_classes',facet_col='review_classes',title='Do certain days have more positive or negative reviews?')
        st.plotly_chart(fig)


        
    with col2:
        # distribution_reviews_by_years
        st.dataframe(distribution_reviews_by_years,hide_index=True)
        
        
        st.markdown('<div style="margin-top:130px;"></div>',unsafe_allow_html=True)
        
        # Trends_reviews_by_months
        st.dataframe(distribution_reviews_by_months,hide_index=True)
        
        
        st.markdown('<div style="margin-top:40px;"></div>',unsafe_allow_html=True)

        
        # Trends_reviews_years
        st.dataframe(number_reviews_years)
        
        st.markdown('<div style="margin-top:20px;"></div>',unsafe_allow_html=True)

        
        # reviews classes with days
        st.dataframe(reviewes_positive_negative,hide_index=True)

    
    
    

# Page 3 - Placeholder Page
def page_3(df):
    st.subheader("Ditribution Charts")
    col1,col2=st.columns(spec=[2,1],gap='large')
    with col1:
        
        # overall rating distribution
        overall_distribution= df['overall'].value_counts().reset_index()
        fig=plx.bar(overall_distribution,y='count',x='overall',title="distribution of the 'overall' ratings")
        st.plotly_chart(fig)
        
        # distribution of 'verified' vs. 'non-verified' reviews
        dist_verified_nonverfied= df['verified'].value_counts().reset_index()
        fig=plx.pie(dist_verified_nonverfied,labels='verified',values='count',title="Distribution of 'verified' vs. 'non-verified' reviews",hover_data='verified')
        st.plotly_chart(fig)

        # distribution of ratings for the top reviewers
        top_reviewrs=df.groupby('reviewerID')[['overall']].count().sort_values(by='overall',ascending=False).head(10).rename(columns={'overall':'count'})
        dit_rating_top_reviewrs=df[df['reviewerID'].isin(top_reviewrs.index)][['reviewerID','overall']]
        fig=plx.histogram(data_frame=dit_rating_top_reviewrs,x='overall',title='Distribution of Ratings for the top Reviewers 364 users')
        st.plotly_chart(fig)
        
        
        # the most common words or phrases used in review texts limited to 100 words?
        
        counter_words=Counter()
        for row in df['cleaned_review_list']:
            row=ast.literal_eval(row)
            counter_words.update(row)
        ## dataframe of most 100 common words
        most_common_words=counter_words.most_common(100)
        df_most_common_words=pd.DataFrame(most_common_words,columns=['word','count'])
        fig=plx.bar(data_frame=df_most_common_words,x='word',y='count',title='Most Common Words in Review Text')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig)
        
        
        # Whats the Most common words used in reviewText column in every class?
        
        # Positive Class Words
        positive_class=df[df['review_classes']=='Positive']['cleaned_review_list']
        word_counter=Counter()
        for row in positive_class:
            row=ast.literal_eval(row)
            word_counter.update(row)
        word_cloud=WordCloud(width=1200,height=600,background_color='white')
        word_cloud.generate_from_frequencies(dict(word_counter))
        fig=plx.imshow(word_cloud,title="Positive Class Common Words")
        fig.update_layout(xaxis_visible=False,yaxis_visible=False)
        st.plotly_chart(fig)

        # Negative Class words
        negative_class=df[df['review_classes']=='Negative']['cleaned_review_list']
        word_counter=Counter()
        for row in negative_class:
            row=ast.literal_eval(row)
            word_counter.update(row)
        word_cloud=WordCloud(width=1200,height=600,background_color='white')
        word_cloud.generate_from_frequencies(dict(word_counter))
        fig=plx.imshow(word_cloud,title="Negative Class Common Words")
        fig.update_layout(xaxis_visible=False,yaxis_visible=False)
        st.plotly_chart(fig)


        # Neutral Class words
        
        Neutral_class=df[df['review_classes']=='Neutral']['cleaned_review_list']
        word_counter=Counter()
        for row in Neutral_class:
            row=ast.literal_eval(row)
            word_counter.update(row)
        word_cloud=WordCloud(width=1200,height=600,background_color='white')
        word_cloud.generate_from_frequencies(dict(word_counter))
        fig=plx.imshow(word_cloud,title="Neutral Class Common Words")
        fig.update_layout(xaxis_visible=False,yaxis_visible=False)
        st.plotly_chart(fig)
        
        # Correlation betweens Numeric Features
        df_numeric_corr=df.corr(numeric_only=True)
        df_numeric_corr.rename(columns={'review_text_length':'review_lenght'},index={'review_text_length':'review_lenght'},inplace=True)
        fig=plx.imshow(df_numeric_corr,title="Correlation for Numeric Columns")
        st.plotly_chart(fig)
        

    
    with col2:
        
        st.markdown('<div style="margin-top : 100px;"></div>',unsafe_allow_html=True)
        
        # overall rating distribution
        st.dataframe(overall_distribution,hide_index=True)
        
        st.markdown('<div style="margin-top : 250px;"></div>',unsafe_allow_html=True)
        
        # distribution of 'verified' vs. 'non-verified' reviews
        st.dataframe(dist_verified_nonverfied,hide_index=True)
        
        st.markdown('<div style="margin-top : 250px;"></div>',unsafe_allow_html=True)
        
        # distribution of ratings for the top reviewers
        st.dataframe(dit_rating_top_reviewrs.sort_values(by='reviewerID'),hide_index=True)
        
        st.markdown("<div style='margin-top : 50px;'></div>",unsafe_allow_html=True)
        
        # the most common words or phrases used in review texts?
        st.dataframe(df_most_common_words,hide_index=True,use_container_width=True)
        
        st.markdown("<div style='margin-top : 1550px;'></div>",unsafe_allow_html=True)
        
        # Correlation Datafram
        st.dataframe(df_numeric_corr)

        
        
# Main App
def main():
    st.set_page_config(layout='wide')
    st.title("EDA for Amazon Reviews Sentiment Analysis ")
    
    df=pd.read_csv("cleaned_amazon_dataset.csv.bz2")
    #df=pd.read_csv("cleaned_amazon_dataset.csv") # loading Dataset

    tab1,tab2,tab3 = st.tabs(["Overview", "Timeline Charts", "Features Distribution Charts"])
    
    with tab1:
        page_1()
    with tab2:
        page_2(df)
    with tab3:
        page_3(df)

if __name__ == "__main__":
    main()
