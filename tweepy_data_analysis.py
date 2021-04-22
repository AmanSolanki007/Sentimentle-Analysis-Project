import tweepy
import csv
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.functions import udf,col
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt


import Login

spark = SparkSession.builder.appName("my_app").getOrCreate()
authenticate=tweepy.OAuthHandler(Login.consumerKey,Login.consumerSecret)
authenticate.set_access_token(Login.accessToken,Login.accessTokenSecret)

api = tweepy.API(authenticate,wait_on_rate_limit=True)
posts = api.user_timeline(screen_name='BillGates',count=100,lang='en',tweet_mode='extended')

i = 1
for tweet in posts:
    data=str(i) +')'+" "+tweet.full_text,'\n'
    # data1=str(data)
    i = i+1
    with open('tweets_data.csv','a') as f:
        csvWriter = csv.writer(f)
        csvWriter.writerow(data)

df = spark.read.csv("tweets_data.csv")

df1 = df.select("_c0")
df2 = df1.withColumn('_c0',regexp_replace('_c0',r'http\S+',''))
df3 = df2.withColumn('_c0',regexp_replace('_c0',r'@[A-Za-z8-9]+',''))
df4 = df3.withColumn('_c0',regexp_replace('_c0','RT[\s]+',''))
df5 = df4.withColumn('_c0',regexp_replace('_c0',':',''))
df6 = df5.withColumn('_c0',regexp_replace('_c0',r'https?:\/\/\S+',''))
df7 = df6.withColumn('_c0',regexp_replace('_c0',r'"',''))

def getsubjectivity(text):
    return TextBlob(text).subjectivity

def getpolarity(text):
    return TextBlob(text).sentiment.polarity

Func =udf(getsubjectivity)

sp_dataframe = df7.withColumnRenamed("_c0","tweets")
data_frame = sp_dataframe.toPandas()

data_frame["subjectivity"]=data_frame['tweets'].apply(getsubjectivity)
data_frame["polarity"]=data_frame['tweets'].apply(getpolarity)

final_df= spark.createDataFrame(data_frame).toDF("tweets","subjectivity","polarity")
final_df.show()

#convert final_df to pandas object
PD_DF=final_df.toPandas()
PD_DF.head(5)

all_word = ' '.join([tw for tw in PD_DF['tweets']])
wordCloud = WordCloud(width=400,height=200,random_state=21,max_font_size=119).generate(all_word)

plt.imshow(wordCloud,interpolation='bilinear')
plt.axis('off')

##############################################
def getanlysis(score):
    if score < 0 :
        return "Negative"
    elif score==0:
        return "Neutral"
    elif score > 0:
        return "Positive"

PD_DF["Analysis"]=PD_DF["polarity"].apply(getanlysis)
#now i am going to print all positive tweets
sorted_df=PD_DF.sort_values(by='polarity')
j=1
for i in range(0,sorted_df.shape[0]):
    if(sorted_df['Analysis'][i] == 'Positive'):
        print(sorted_df['tweets'][i])
        print()
        j=j+1

#now i am going to print all negative tweets
j=1
sorted_df=PD_DF.sort_values(by=['polarity'],ascending='False')
for i in range(0,sorted_df.shape[0]):
    if (sorted_df['Analysis'][i]=="Negative"):
        print(sorted_df["tweets"][i])
        print()
        j=j+1

# now i plot a figure

plt.figure(figsize=(8,6))
for i in range(0,PD_DF.shape[0]):
    plt.scatter(PD_DF['polarity'][i],PD_DF['subjectivity'][i],color='Blue')
    plt.title('Sentimental Analysis')
    plt.xlabel('polarity')
    plt.ylabel('subjectivity')
    plt.grid()
    # plt.show()

#percentage of positive tweets
ptweets = PD_DF[PD_DF.Analysis=='Positive']
ptweets = ptweets['tweets']
ptweets=round(lit(ptweets.shape[0]/PD_DF.shape[0]*100),1)

#percentage of negative tweets
ntweets = PD_DF[PD_DF.Analysis == 'Negative']
ntweets = ntweets['tweets']
ntweets=round(lit(ntweets.shape[0]/PD_DF.shape[0]*100),1)

#show value count
PD_DF['Analysis'].value_counts()

#plot final graph

plt.title("Sentimental Analysis")
plt.xlabel("Sentiment")
plt.ylabel("Counts")
PD_DF['Analysis'].value_counts().plot(kind='bar')
plt.show()





