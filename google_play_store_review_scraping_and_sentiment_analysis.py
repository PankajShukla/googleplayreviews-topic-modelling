#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Libraries

# In[ ]:


get_ipython().system('pip install google_play_scraper')
get_ipython().system('pip install sklearn')

import pandas as pd
from google_play_scraper.features.reviews import Sort, reviews_all, reviews
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# In[280]:





# # Reviews data extraction

# In[293]:


result = reviews_all('com.bt.bms',sleep_milliseconds=0,lang='en', country='us')


# # Create dataframe of the reviews

# In[294]:


df = pd.DataFrame(result)


# In[295]:


df.head(2)


# In[296]:


print(f'Total textual reviews: {len(result)} \n')

unique_users  = len(df['userName'].unique())
unknown_users = len(df[df['userName']=='A Google user'])
total_reviews = len(df)

print(f'Total unique users : {unique_users}')
print(f'Total unknown users: {unknown_users}')
print(f'Total users who gave multiple reviews: {total_reviews - unique_users - unknown_users}\n')


mean = df['score'].mean()
print(f'Average rating for this app based on the textual reviews: {round(mean,2)} \n')


# # Extract all reviews with rating below 4

# In[403]:


df_tm = df[df['score']<=3]


# In[404]:


df_tm = df_tm[df_tm.content.str.len()>=30]


# In[405]:


print(f'Remaining textual reviews: {len(df_tm)} \n')


# # Get the relevant columns for topic modelling

# In[406]:


df_tm = df_tm[['reviewId','content']].drop_duplicates()
df_tm.dropna(inplace=True)
df_tm = df_tm.reset_index().drop(columns='index')


# In[407]:


print(f'Remaining textual reviews: {len(df_tm)} \n')


# # Use CountVectorizer to create document term matrix

# In[409]:


cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')

# max_df : discard words that occur more than 95% documents
# min_df : include only those words that occur atleast in 2 documents


# In[410]:


dtm = cv.fit_transform(df_tm['content'])


# In[411]:


dtm
#shows 9510 terms and 56561 articles


# In[412]:


len(cv.get_feature_names())


# # Using LDA for topic modelling

# In[413]:


LDA = LatentDirichletAllocation(n_components=5,random_state=1)

LDA.fit(dtm)


# In[ ]:





# # Extract data

# In[414]:


for index,topic in enumerate(LDA.components_):
    print(f'topic #{index} : ')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-30:]])


# In[ ]:





# In[ ]:





# 0. App/OverallExp
# 1. Delivery-CommitmentIssue
# 2. FoodQuality
# 3. Offers
# 4. App/Coupons
# 5. Delivery-RestaurantIssue
# 6. CustomerSupport
# 7. Competitors
# 8. Refund-ChatSupport
# 9. Refund-Cancellation

# In[416]:


topic_results = LDA.transform(dtm)


# In[417]:


topic_results


# In[456]:


df_topic_results = pd.DataFrame(topic_results, columns=[
'0_InternetCharges',
'1_Payment/Offers' ,
'2_App'            ,
'3_Booking-Refund/Ticket'  ,
'4_Booking-Location/language' 
])


# In[ ]:





# In[457]:


df_topic_results.head(3)


# In[458]:


df_result = pd.merge(df_tm,df_topic_results,  how='inner', left_index=True, right_index=True )


# In[459]:


df_result.drop(columns='reviewId').iloc[175,]['content']


# In[ ]:





# In[461]:


df_output = pd.merge(df, df_result,  how='left', on='reviewId' )


# In[462]:


len(df_output)


# In[463]:


df_output.to_csv('app_reviews_bms.csv')


# In[ ]:





# In[ ]:





# In[ ]:




