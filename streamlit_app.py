!pip install matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import streamlit as st
import re 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import helper
from collections import Counter
import emojis
from sorted_months_weekdays import *
from sort_dataframeby_monthorweek import *
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go


st.title("WHATSUP CHAT ANALYSIS")
st.balloons()
data = open("dummy.txt","r")

#####################################################################################################
###
try:
  def preprocess_data(data):
    pattern = "\d{1,2}/\d{1,2}/\d{1,2},\s\d{1,2}:\d{2}\s[ap][m]\s"
    message = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)
    df = pd.DataFrame({"dates":dates,"messages":message})
    df["dates"] = pd.to_datetime(df["dates"])
    df["messages"] = df["messages"].apply(lambda x:x.replace("-",""))
    return df

  def extract_user(item):
    users = []
    if ":" not in item:
      users.append("group notification")
    else:
      item = item.split(":")
      users.append(item[0])
    return str(users[0])
  def extract_messages(item):
    messages = []
    if ":" not in item:
      messages.append(item)
    else:
      item = item.split(":")
      messages.append(item[1])
    return str(messages[0])

  uploaded_file = st.sidebar.file_uploader("Choose file")
  if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")


  from PIL import Image
  image = Image.open('image1.jpg')
  st.sidebar.image(image,width = 200)


  st.sidebar.write("Click the audio below to have some entertainment till analysis is complete")
  audio_file = open('To Brazil 320 Kbps.mp3', 'rb')
  st.sidebar.audio(audio_file)

  pattern = "\d{1,2}/\d{1,2}/\d{1,2},\s\d{1,2}:\d{2}\s[ap][m]\s"
  message = re.split(pattern, data)[1:]
  dates = re.findall(pattern, data)
  df = pd.DataFrame({"dates":dates,"messages":message})
  df2 = df
  df["dates"] = pd.to_datetime(df["dates"])
  df["messages"] = df["messages"].apply(lambda x:x.replace("-",""))
  df["users"] = df["messages"].apply(extract_user)
  df["message"] = df["messages"].apply(extract_messages)
  df["year"] = df["dates"].dt.year
  df["month"] = df["dates"].dt.month_name()
  df["day"] = df["dates"].dt.day_name()
  df["hour"] = df["dates"].dt.hour
  df["minute"] = df["dates"].dt.minute
  final_dataframe = df[["users","message","year","month","day","hour","minute"]]
  user_list = df["users"].unique().tolist()
  user_list.remove("group notification")
  user_list.sort()
  user_list.insert(0,"Overall")
  selected_user = st.sidebar.selectbox("Show Analysis wrt",user_list)
  if selected_user == "Overall":
  	st.write(final_dataframe)
  else:
  	final_dataframe = final_dataframe[final_dataframe["users"] == selected_user]
  	st.write(final_dataframe)
except:
  pass

  
################################## STARTING ANALYSIS ########################################################

if st.sidebar.button("Show Analysis"):

  st.title("Top Statistics")
  try:
    num_messages, num_words, num_mediafiles_shared, num_links = helper.fetch_stats(selected_user,df)
    col1,col2,col3,col4 = st.columns(4)
    with col1:
      st.header("Total Messages")
      st.title(num_messages)
    with col2:
      st.header("Total Words")
      st.title(num_words)
    with col3:
      st.header("Total Media")
      st.title(num_mediafiles_shared)
    with col4:
      st.header("Total Links")
      st.title(num_links)
  except:
    pass

################################################################################################

##finding top 10 active users

  if selected_user == "Overall":
    try:
      st.title("Top 10 Most Users")
      top = df[df["users"] != "group notification"]
      top = top["users"].value_counts()
      top_df = pd.DataFrame({"name":top.index,"values":top.values})
      s = top_df["values"].sum()
      top_df["percent_usage"] = top_df["values"].apply(lambda x:round((x/s)*100,2))
      col1, col2 = st.columns(2)
      with col2:
        fig, ax = plt.subplots()
        ax.bar(top_df["name"][0:9],top_df["values"][0:9], color="Orange", linewidth = 5,edgecolor = "black")
        plt.xticks(rotation ="90")
        plt.title("Top 10 most active users",fontweight = "bold")
        plt.xlabel("users",fontweight = "bold")
        plt.ylabel("Number of messages", fontweight = "bold")
        st.pyplot(fig)
      with col1:
        st.write(top_df[0:10])
    except:
      pass
##############################################################################################################

  try:
    st.title("Top 10 Most Media sent users")
    if selected_user == "Overall":
  	  a = final_dataframe[final_dataframe["message"] == " <Media omitted>\n"]
  	  a = a["users"].value_counts()
  	  top_media = pd.DataFrame({"name":a.index,"values":a.values})
    else:
  	  final_dataframe = final_dataframe[final_dataframe["users"] == selected_user]
  	  a = final_dataframe[final_dataframe["message"] == " <Media omitted>\n"]
  	  a = a["users"].value_counts()
  	  top_media = pd.DataFrame({"name":a.index,"values":a.values})
    if top_media.empty:
  	  st.write("No media is sent by the user")
    else:
      col1, col2 = st.columns(2)
      with col2:
        fig, ax = plt.subplots()
        ax.bar(top_media["name"][0:9],top_media["values"][0:9], color="purple", linewidth = 5,edgecolor = "black")
        plt.xticks(rotation ="90")
        plt.title("Top 10 most Media sent users",fontweight = "bold")
        plt.xlabel("users",fontweight = "bold")
        plt.ylabel("Number of Medias", fontweight = "bold")
        st.pyplot(fig)
      with col1:
        st.write(top_media[0:9])
  except:
     pass
#####################################################################################################

## Generating the wordcloud
  try:
    st.header("Wordcloud")
    df_ = df[df["message"] != " <Media omitted>\n"]
    df_ = df_[df_["users"] != "group notification"]
    if selected_user == "Overall":
      wc = WordCloud(width = 500, height = 500, min_font_size = 10,background_color = "black")
      df_wc = wc.generate(df_["message"].str.cat(sep = " "))
      fig, ax = plt.subplots()
      ax.imshow(df_wc)
      st.pyplot(fig)
    else:
      df_ = df_[df_["users"] == selected_user]
      if df_.empty:
  	    st.write("No messages have been sent by the user")
      else:
  	    wc = WordCloud(width = 500, height = 500, min_font_size = 10,background_color = "black")
  	    df_wc = wc.generate(df_["message"].str.cat(sep = " "))
  	    fig, ax = plt.subplots()
  	    ax.imshow(df_wc)
  	    st.pyplot(fig)
  except:
  	pass
      
 ##########################################################################################################

##Generating top 20 words
  try:
    st.header("Top 20 Most used words")
    temp_df = df[df["users"] != "group notification"]
    temp_df = temp_df[temp_df["message"] != " <Media omitted>\n"]
    words = []
    if selected_user == "Overall":
      for message in temp_df["message"]:
  	    message = re.sub(pattern='[^a-zA-Z]',repl=' ', string=message)
  	    for word in message.lower().split():
  	      if word not in set(stopwords.words('english')):
  	      	words.append(word)
  	    	 
  	    
    else:
      temp_df = temp_df[temp_df["users"] == selected_user]
      for message in temp_df["message"]:
  	    message = re.sub(pattern='[^a-zA-Z]',repl=' ', string=message)
  	    for word in message.lower().split():
  	      if word not in set(stopwords.words('english')):
  	      	words.append(word)
  	    	 
  	    
  	    	
    top_20 = pd.DataFrame(Counter(words).most_common(20))
    top_20 = top_20.rename(columns = {0:"Words", 1:"Count"})
    if top_20.empty:
  	  st.write("No messages have been sent by the user")
    else:
      col1, col2 = st.columns(2)
      with col2:
        fig,ax = plt.subplots()
        ax.bar(top_20["Words"], top_20["Count"], color = "green", linewidth = 4,edgecolor = "black")
        plt.xticks(rotation = "90")
        plt.title("Top 20 most used words",fontweight = "bold")
        st.pyplot(fig)
      with col1:
        st.write(top_20)
  except:
  	pass

#####################################################################################################

#Generating the top 10 most widely used emojis

  try:
    if selected_user == "Overall":
  	  l = []
  	  for message in df["message"]:
  	  	l.append(message)
  	  emoji = []
  	  for item in l:
  	  	emoji.extend(emojis.get(item))
    else:
      df = df[df["users"] == selected_user]
      l = []
      for message in df["message"]:
      	l.append(message)
      emoji = []
      for item in l:
      	emoji.extend(emojis.get(item))
    top_emoji = pd.DataFrame(Counter(emoji).most_common(len(Counter(emoji))))
    top_emoji= top_emoji.rename(columns = {0:"Emoji", 1: "Count"})
    st.header("Top 10 most used emojis")
    if top_emoji.empty:
  	  st.write("No emojis are used by the user")
    else:
      col1, col2 = st.columns(2) 
      with col1:
        st.write(top_emoji[0:10])
      with col2:
        fig= px.pie(top_emoji[0:10], values = 'Count', names='Emoji')
        fig.update_traces(textposition='inside', textinfo='percent+label')
        col2.write(fig)
  except:
  	pass
 ##########################################################################################################
  # Generating the yearly trends according to month
  try:
    st.header("Trend of data for past 2 years")
    t = []
    if selected_user == "Overall":
      df2["month_num"] = df2["dates"].dt.month
      timeline = df2.groupby(["year","month","month_num"]).count()["message"].reset_index()
      for i in range(timeline.shape[0]):
        t.append(timeline["month"][i] + "-" + str(timeline["year"][i]))
    else:
      df2 = df2[df2["users"] == selected_user]
      df2["month_num"] = df2["dates"].dt.month
      timeline = df2.groupby(["year","month","month_num"]).count()["message"].reset_index()
      for i in range(timeline.shape[0]):
        t.append(timeline["month"][i]+"-"+str(timeline["year"][i]))
    timeline["month_year"] = t
    timeline = Sort_Dataframeby_Month(df = timeline,monthcolumnname = "month")
    if timeline.empty:
  	  st.write("The user is not active in the group")
    else:
      l = timeline["year"].unique()
      l.sort()
      if len(l) == 1:
        e = timeline[timeline["year"]== l[-1]]
        st.header(l[-1])
        fig,ax = plt.subplots()
        ax.plot(e["month_year"], e["message"])
        plt.xticks(rotation = "90")
        st.pyplot(fig)
      else:
        for i in l:
          d = timeline[timeline["year"]== l[-2]]
          e = timeline[timeline["year"]== l[-1]]
        col1,col2 = st.columns(2)
        with col1:
          st.header(l[-2])
          fig,ax = plt.subplots()
          ax.plot(d["month_year"], d["message"])
          plt.xticks(rotation = "90")
          st.pyplot(fig)
        with col2:
          st.header(l[-1])
          fig,ax = plt.subplots()
          ax.plot(e["month_year"], e["message"])
          plt.xticks(rotation = "90")
          st.pyplot(fig)
  except:
  	pass

############################################################################################################
## Generating the date trend
  try:
    st.header("Trend of data by date")
    if selected_user =="Overall":
  	  df2["only_date"]  = df2["dates"].dt.date
  	  daily_timeline = df2.groupby("only_date").count()["message"].reset_index()
    else:
  	  df2 = df2[df2["users"] == selected_user]
  	  df2["only_date"]  = df2["dates"].dt.date
  	  daily_timeline = df2.groupby("only_date").count()["message"].reset_index()
    if daily_timeline.empty:
  	  st.write("The user is not active")
    else:
      fig,ax = plt.subplots()
      ax.plot(daily_timeline["only_date"], daily_timeline["message"],color = "green")
      plt.xticks(rotation = "90")
      st.pyplot(fig)
  except:
  	pass

 ############################################################################################################
 #################Finding the most busy hour,day, month, year#############################################################
  try:
    st.title("Trends by hour,day,month and year")
    if selected_user != "Overall":
  	  df2 = df2[df2["users"] == selected_user]
    #busy day
    df2["day_name"] = df2["dates"].dt.day_name()
    day_trend = df2[['day_name',"message"]]
 

    a = day_trend["day_name"].value_counts()
    day_trend1 = pd.DataFrame(a).reset_index()

    #busy month
    busy_month = df2[["month",'message']]
    b = busy_month["month"].value_counts()
    busy_month1 = pd.DataFrame(b).reset_index()
  

    #busy hour
    hour_trend = df2[["hour","message"]]
    c = hour_trend["hour"].value_counts()
    hour_trend = pd.DataFrame(c).reset_index()
  

    # busy year
    busy_year = df2[["year","message"]]
    o = busy_year["year"].value_counts()
    busy_year1 = pd.DataFrame(o).reset_index()
  
    if hour_trend.empty and day_trend.empty:
  	  st.write("The user is not active")
    else:
      col1,col2 = st.columns(2)
      with col1:
        st.header("Most busy hours of a day")
        fig,ax = plt.subplots()
        ax.bar(hour_trend["index"], hour_trend["hour"],color = "orange",linewidth = 4,edgecolor = "black")
        plt.xticks(rotation = "90")
        st.pyplot(fig)
      with col2:
        st.header("Most busy weekdays")
        fig,ax = plt.subplots()
        ax.bar(day_trend1["index"],day_trend1["day_name"],color = "green",linewidth = 4,edgecolor = "black")
        plt.xticks(rotation = "90")
        st.pyplot(fig)

    if busy_month1.empty and busy_year1.empty:
  	  st.write("The user is not active")
    else:
      col1,col2 = st.columns(2)
      with col1:
        st.header("Most busy Months")
        fig,ax = plt.subplots()
        ax.bar(busy_month1["index"],busy_month1["month"],color = "red",linewidth = 4,edgecolor = "black")
        plt.xticks(rotation = "90")
        st.pyplot(fig)

      with col2:
        st.header("Most busy years")
        fig,ax = plt.subplots()
        ax.bar(busy_year1["index"], busy_year1["year"],color = "blue",linewidth = 4,edgecolor = "black")
        plt.xticks(rotation = "90")
        st.pyplot(fig)
  except:
    pass
#############################################################################################################3
 
  ##Weekly and hourly trend
  try:
    period = []
    for hour in df[["day_name","hour"]]["hour"]:
      if hour == 23:
        period.append(str(hour) + "-"+ str("00"))
      elif hour == 0:
        period.append(str("00")+ "-" +str(hour+1))
      else:
        period.append(str(hour)+ "-" + str(hour+1))

    df2["period"] = period
    st.header("weekly and hourly trend")
    fig,ax = plt.subplots()
    ax = sns.heatmap(df.pivot_table(index = "day_name", columns = "period", values = "message", aggfunc = "count").fillna(0), cmap= "seismic_r")
    plt.yticks(rotation = "0")
    st.pyplot(fig)

    st.header("Monthly and hourly trend")
    fig,ax = plt.subplots()
    ax = sns.heatmap(df.pivot_table(index = "month", columns = "period", values = "message", aggfunc = "count").fillna(0), cmap= "seismic")
    plt.yticks(rotation = "0")
    st.pyplot(fig)
  except:
  	pass
##############################################################################################################
#st.snow()
#st.balloons()  



  

  

  

 



  









 
  
    
    
    
      
    
  
	
	

    
      

   
   
   
  
   



    	








    	











      
      
        
      
        
        
      

    
      
      
        
      






















