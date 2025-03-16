#!/usr/bin/env python
# coding: utf-8

# In[345]:


import sys
print(sys.executable)  # Check which Python is running
print(sys.version)      # Verify Python version


# In[346]:


get_ipython().system('pip install nltk')


# In[347]:


import sys
get_ipython().system('{sys.executable} -m pip install --upgrade pip')


# In[348]:


pip install nltk


# In[349]:


import nltk
print(nltk.__version__)


# In[350]:


pip install plotly


# In[351]:


pip install scikit-learn


# In[352]:


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

import webbrowser
import os


# In[353]:


apps_df=pd.read_csv('googleplaystore.csv')
apps_df


# In[354]:


apps_df.head()


# In[355]:


reviews_df=pd.read_csv('googleplaystore_user_reviews.csv')


# In[356]:


reviews_df.head()


# In[357]:


# pd.read_csv() : csv files
# pd.read_excel()
# pd.read_sql()
#pd.read_json()


# In[358]:


# recognize missing values
# df.isnull()
# df.dropna() : remove row and column that contain missing values
# df.fillna() : fills missing values


# In[359]:


# handle duplicate
#df.duplicated() : identifies duplicates
# df.drop_duplicates() : removes duplicates


# In[360]:


# step2   Data cleaning
apps_df=apps_df.dropna(subset=['Rating'])
for column in apps_df.columns:
    apps_df[column].fillna(apps_df[column].mode()[0],inplace=True)
    
apps_df.drop_duplicates(inplace=True)
apps_df=apps_df=apps_df[apps_df['Rating']<=5]
reviews_df.dropna(subset=['Translated_Review'],inplace=True)


# In[361]:


# data transformation
# converts installs column to numeric by removing , and +
apps_df['Installs'] = apps_df['Installs'].astype(str).str.replace('+', '', regex=False).str.replace(',', '').astype(int)




# In[362]:


apps_df


# In[363]:


apps_df.dtypes


# In[364]:


apps_df['Price'] = apps_df['Price'].astype(str).str.replace('$', '').astype(float)


# In[365]:


apps_df.dtypes


# In[366]:


# merging apps and reviews data sets
merged_df=pd.merge(apps_df,reviews_df,on ='App',how='inner')


# In[367]:


merged_df


# In[368]:


# size conversion
def convert_size(size):
    if'M'in size:
        return float(size.replace('M',''))
    elif 'k' in size:
        return float(size.replace('k',''))/1024
    else:
        return np.nan
apps_df['Size']=apps_df['Size'].apply(convert_size)


# In[369]:


apps_df


# In[370]:


# log transformation
apps_df['log_Installs']=np.log1p(apps_df['Installs'])
#apps_df['Reviews']=np.log1p(apps_df['Reviews'])


# In[371]:


apps_df


# In[372]:


apps_df['Reviews']=apps_df['Reviews'].astype(int)
apps_df['Reviews']=np.log1p(apps_df['Reviews'])


# In[373]:


apps_df


# In[374]:


#categarize the rating
def rating_group(rating):
    if rating>=4:
        return'top rate'
    elif rating>=3:
        return 'above avg'
    elif rating>=2:
        return 'avg'
    else:
        return 'below avg'
apps_df['Rating_Group']=apps_df['Rating'].apply(rating_group)


# In[375]:


print(apps_df.dtypes)


# In[376]:


apps_df


# In[377]:


apps_df["Revenue"] = apps_df["Price"] * apps_df["Installs"]
print("Revenue column created successfully!")


# In[378]:


apps_df


# In[379]:


# creating  new revenue column
#apps_df['Revenue']=apps_df['Price']*apps_df['Installs']


# In[380]:


### NLP##############
#SIA--sentiment intensity analyser
# polarity scores in sia (positive ,negetive,neutral,compound:-1 very negetive,+1 very positive)
sia=SentimentIntensityAnalyzer()
review="this app is amazing! i love the new features"
sentiment_scores=sia.polarity_scores(review)
print(sentiment_scores)


# In[381]:


review="this app is very bad! i hate the new features"
sentiment_scores=sia.polarity_scores(review)
print(sentiment_scores)


# In[382]:


review="this app is okay"
sentiment_scores=sia.polarity_scores(review)
print(sentiment_scores)


# In[383]:


reviews_df['Sentiment_score']=reviews_df['Translated_Review'].apply(lambda x:sia.polarity_scores(str(x))['compound'])


# In[384]:


reviews_df.head()


# In[385]:


apps_df['Last Updated']=pd.to_datetime(apps_df['Last Updated'],errors='coerce')


# In[386]:


apps_df['Year']=apps_df['Last Updated'].dt.year


# In[387]:


apps_df


# In[388]:


import plotly.express as px


# In[389]:


figure=px.bar(x=['A','B','C'],y=[1,2,3],title='simple bar chart')
figure.show()


# In[390]:


figure.write_html('interactive_plot.html')


# In[391]:


# static visualizations: fixed images or plaots,non interactive
# interactive visualization
#


# In[392]:


# create a directiory to store html files
html_files_path='./'
if not os.path.exists(html_files_path):
    os.makedirs(html_files_path)


# In[393]:


plot_containers=""


# In[394]:


def save_plot_as_html(fig,filename,insight):
    global plot_containers
    filepath=os.path.join(html_files_path,filename)
    html_content=pio.to_html(fig,full_html=False,include_plotlyjs='inline')
    plot_containers+=f"""
    <div class="plot-container" id="{filename}"onclick="openPlot('{filename}')">
        <div class='plot'>{html_content}</div>
        <div class='insights'>{insight}</div>
    </div>
    """
    fig.write_html(filepath,full_html=False,include_plotlyjs='inline')


# In[395]:


plot_width=400
plot_height=300
plot_bg_color='black'
text_color='white'
title_font={'size':16}
axis_font={'size':12}


# In[396]:


# fig1(categarize the top 10 apps in play store)

category_counts=apps_df['Category'].value_counts().nlargest(10)
fig1=px.bar(
    x=category_counts.index,
    y=category_counts.values,
    labels={'x':'Category','y':'count'},
    title='top categories on play store',
    color=category_counts.index,
    color_discrete_sequence=px.colors.sequential.Plasma,
    width=400,
    height=300
)
fig1.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
fig1.update_traces(marker=dict(line=dict(color=text_color,width=1)))
save_plot_as_html(fig1,"Category Graph 1.html","the top categories on the play store are dominated by tools,entertainment,and productivity apps")


# In[397]:


apps_df


# In[398]:


# type analysis plot (distribution vs free and paid apps)
#figure2
# for distribution we use pi chart
type_counts=apps_df['Type'].value_counts()
fig2=px.pie(
    values=type_counts.values,
    names=type_counts.index,
    
    title='App Type Distribution',
    color_discrete_sequence=px.colors.sequential.RdBu,
    width=400,
    height=300
)
fig2.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    
    margin=dict(l=10,r=10,t=30,b=10)
)
save_plot_as_html(fig2,"Type Graph 2.html","Most apps on playstore are free strategy to attract users and monatize through ads")


# In[399]:


apps_df


# In[400]:


# histogram  for Rating distribution
# figure3
fig3=px.histogram(
    apps_df,
    x='Rating',
    nbins=20,
    
    
    title='App Rating Distribution',
    color_discrete_sequence=['#636EFA'],
    width=400,
    height=300
)
fig3.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
save_plot_as_html(fig3,"Rating Graph 3.html","Rating are skewed towards higher values,suggesting that most apps arerated favorable bu users")


# In[401]:


apps_df


# In[402]:


#figure4
# bar chart
sentiment_counts=reviews_df['Sentiment_score'].value_counts()
fig4=px.bar(
    x=sentiment_counts.index,
    y=sentiment_counts.values,
    labels={'x':'Sentiment_score','y':'count'},
    title='Sentiment Distribution',
    color=sentiment_counts.index,
    color_discrete_sequence=px.colors.sequential.RdPu,
    width=400,
    height=300
)
fig4.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig1.update_traces(market=dict(market=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig4,"Sentiment Graph 4.html","Sentiment in reviews show a mix of positive and negetive feedback.with a slight lean towards positive sentiment")


# In[403]:


apps_df


# In[404]:


# category plot
# figure5
installs_by_category=apps_df.groupby('Category')['Installs'].sum().nlargest(10)
fig5=px.bar(
    x=installs_by_category.index,
    y=installs_by_category.values,
    orientation='h',  # h=horizontal
    labels={'x':'Installs','y':'Category'},
    title='installs_by_category',
    color=installs_by_category.index,
    color_discrete_sequence=px.colors.sequential.Blues,
    width=400,
    height=300
)
fig5.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig1.update_traces(market=dict(market=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig5,"installs Graph 4.html","the categories with most installs are social and communication apps,reflecting their broad appeal")


# In[405]:


apps_df


# In[406]:


apps_df


# In[407]:


# Convert "Last Updated" to datetime format
apps_df["Last Updated"] = pd.to_datetime(apps_df["Last Updated"], errors="coerce")

# Drop rows where "Last Updated" is missing
apps_df = apps_df.dropna(subset=["Last Updated"])

# Count updates per year
updates_per_year = apps_df["Last Updated"].dt.year.value_counts().sort_index()

# Create line graph
fig6 = px.line(
    x=updates_per_year.index,
    y=updates_per_year.values,
    labels={"x": "Year", "y": "Number of Updates"},
    title="Number of Updates over the Years",
    color_discrete_sequence=["#AB63FA"],
    width=400,
    height=300
)

# Customize layout
fig6.update_layout(
    plot_bgcolor="black",
    paper_bgcolor="black",
    font_color="white",
    title_font={"size": 16},
    xaxis=dict(title_font={"size": 12}),
    yaxis=dict(title_font={"size": 12}),
    margin=dict(l=10, r=10, t=30, b=10)
)

# Save plot as HTML
save_plot_as_html(fig6, "Updates_Graph_6.html", "Updates have been increasing over the years, showing that developers are actively maintaining and improving the apps.")


# In[408]:


print(apps_df.columns)


# In[409]:


apps_df


# In[410]:


print(apps_df.dtypes)


# In[411]:


apps_df['Installs'] = apps_df['Installs'].astype(str).str.replace('+', '', regex=False).str.replace(',', '').astype(int)


# In[412]:


apps_df


# In[413]:


print(apps_df.dtypes)


# In[414]:


apps_df['Price'] = apps_df['Price'].astype(str).str.replace('$', '').astype(float)


# In[415]:


print(apps_df.dtypes)


# In[416]:


apps_df["Revenue"] = apps_df["Price"] * apps_df["Installs"]
print("Revenue column created successfully!")


# In[417]:


apps_df


# In[418]:


#figure7
# comparing the revenue using bar graph

revenue_by_category=apps_df.groupby('Category')['Revenue'].sum().nlargest(10)
fig7=px.bar(
    x=revenue_by_category.index,
    y=revenue_by_category.values,
    labels={'x':'Category','y':'Revenue'},
    title='revenue_by_category',
    color=revenue_by_category.index,
    color_discrete_sequence=px.colors.sequential.Greens,
    width=400,
    height=300
)
fig7.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig1.update_traces(market=dict(market=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig7,"Revenue Graph 7.html", "The highest revenue-generating app categories are shown, indicating market trends and profitability")


# In[419]:


apps_df


# In[420]:


# figure 8
# count top 10 genre

genre_counts=apps_df['Genres'].str.split(';',expand=True).stack().value_counts().nlargest(10)
fig8=px.bar(
    x=genre_counts.index,
    y=genre_counts.values,
    labels={'x':'Genres','y':'Count'},
    title='Top Genres',
    color=genre_counts.index,
    color_discrete_sequence=px.colors.sequential.OrRd,
    width=400,
    height=300
)
fig8.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig1.update_traces(market=dict(market=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig8,"Genre Graph 8.html","Action nad Casual genres are the most common,reflecting users prefer for enganing and easy-to-play games") 


# In[421]:


apps_df


# In[422]:


# figure 9
# scatter plot

fig9=px.scatter(
    apps_df,
    x='Last Updated',
    y='Rating',
    color='Type',
    
    title='Impact of last Update on Rating',
    
    color_discrete_sequence=px.colors.qualitative.Vivid,
    width=400,
    height=300
)
fig9.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig1.update_traces(market=dict(market=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig9,"Update Graph 9.html","The Scatter Plot shows a weak coreraltion between the last update and rating,suggesting that more frequent updates donot always result in better ratings") 


# In[423]:


apps_df


# In[424]:


#box plot
#figure 10
fig10=px.box(  

    apps_df,
    x='Type',
    y='Rating',
    color='Type',
    
    title='Rating for Paid vs Free Apps',
    
    color_discrete_sequence=px.colors.qualitative.Pastel,
    width=400,
    height=300
)
fig10.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig1.update_traces(market=dict(market=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig10,"Paid Free Graph 10.html","Paid apps generally have higher rating compared to free apps,suggesting that user expect higher quality from paid apps")


# In[425]:


apps_df


# In[426]:


# fig 11
# Filter only paid apps & drop NaN values in Revenue
apps_df_paid = apps_df[(apps_df["Type"] == "Paid") & (apps_df["Revenue"].notna())]

# Scatter plot with trendline
fig11 = px.scatter(
    apps_df_paid,
    x="Installs",
    y="Revenue",
    color="Category",
    trendline="ols",  # Ordinary Least Squares regression (linear trendline)
    title="Revenue vs Installs for Paid Apps",
    labels={"Installs": "Number of Installs", "Revenue": "Revenue ($)"},
    width=1000,
    height=500
)

# Update layout for better visualization
fig11.update_layout(
    plot_bgcolor="black",
    paper_bgcolor="black",
    font_color="white",
    title_font={"size": 18},
    xaxis=dict(title_font={"size": 14}),
    yaxis=dict(title_font={"size": 14}),
)

# Show the plot
save_plot_as_html(fig11,"Revenue vs Installs.html","This plot shows the relationship between installs and revenue for paid apps, with color-coded categories and a trendline indicating correlation.")


# In[427]:


apps_df


# In[428]:


print(apps_df.dtypes)


# In[429]:


# Convert "Installs" to numeric (remove commas and "+" if present)
apps_df["Installs"] = (
    apps_df["Installs"]
    .astype(str)  # Ensure it's a string before replacement
    .str.replace(",", "")
    .str.replace("+", "")
    .astype(float)  # Convert to numeric
)

# Convert "Price" to numeric (remove "$" if present)
apps_df["Price"] = (
    apps_df["Price"]
    .astype(str)  # Ensure it's a string before replacement
    .str.replace("$", "")
    .astype(float)  # Convert to numeric
)


# In[430]:


import pandas as pd
import plotly.express as px
import datetime
import pytz
# fig 12
# Load dataset (Replace "googleplaystore.csv" with your actual file)
apps_df = pd.read_csv("googleplaystore.csv")

# Convert "Installs" to numeric (handle missing values and remove non-numeric characters)
apps_df["Installs"] = (
    apps_df["Installs"]
    .astype(str)
    .str.replace(",", "")
    .str.replace("+", "")
    .str.replace("Free", "0")  # Handle cases where 'Free' appears
    .astype(float)
)

# Filter out categories starting with A, C, G, or S
apps_df = apps_df[~apps_df["Category"].str.startswith(("A", "C", "G", "S"))]

# Select the top 5 categories based on installs
top_categories = apps_df.groupby("Category")["Installs"].sum().nlargest(5).index
apps_df = apps_df[apps_df["Category"].isin(top_categories)]

# Check if 'Country' column exists
if "Country" not in apps_df.columns or apps_df["Country"].isnull().all():
    country_data_available = False
else:
    country_data_available = True

# Get current time in IST
ist = pytz.timezone("Asia/Kolkata")
current_time = datetime.datetime.now(ist).time()

# Define allowed time window (6 PM - 8:00 PM IST)
start_time = datetime.time(18, 00)
end_time = datetime.time(20, 00)



# Check if current time is within allowed window
if start_time <= current_time <= end_time:
    if country_data_available:
        # Create Choropleth map if country data is available
        fig = px.choropleth(
            apps_df,
            locations="Country",
            locationmode="country names",
            color="Installs",
            hover_name="Category",
            title="Global Installs by App Category",
            color_continuous_scale="Viridis",
        )
    else:
        # Show an empty map with a message
        fig = px.choropleth(
            locations=[],
            title="Global Installs -Country Data Unavailable",
        )
    
    # Save the plot
    save_plot_as_html(fig, "Global_Installs_Choropleth.html", 
        "This interactive map shows installs for selected categories where installs exceed 1 million, but only displays between 6:00 AM and 8:00 PM IST."
    )
    
    

else:
    print("Choropleth map is only available between 6 PM and 8 PM IST.")


# In[ ]:





# In[431]:


import pandas as pd
import plotly.express as px
import datetime
import pytz
# fig 13
# Load dataset
apps_df = pd.read_csv("googleplaystore.csv")  # Replace with actual dataset file

# Convert "Reviews" to numeric (fixes TypeError)
apps_df["Reviews"] = pd.to_numeric(apps_df["Reviews"], errors="coerce")

# Step 1: Filter categories with more than 50 apps
category_counts = apps_df['Category'].value_counts()
valid_categories = category_counts[category_counts > 50].index
apps_df_filtered = apps_df[apps_df['Category'].isin(valid_categories)]

# Step 2: Filter apps where the name contains the letter "C" (case insensitive)
apps_df_filtered = apps_df_filtered[apps_df_filtered['App'].str.contains('C', case=False, na=False)]

# Step 3: Exclude apps with fewer than 10 reviews
apps_df_filtered = apps_df_filtered[apps_df_filtered['Reviews'] >= 10]  # Fixed

# Step 4: Include only apps with a rating below 4.0
apps_df_filtered = apps_df_filtered[apps_df_filtered['Rating'] < 4.0]

# Step 5: Check current time in IST (Indian Standard Time)
ist = pytz.timezone('Asia/Kolkata')
current_time_ist = datetime.datetime.now(ist).time()

start_time = datetime.time(16, 0)  # 4:00 PM IST
end_time = datetime.time(18, 0)    # 6:00 PM IST

# Step 6: Display the violin plot only within the specified time window
if start_time <= current_time_ist <= end_time:
    # Create the violin plot
    fig13 = px.violin(
        apps_df_filtered,
        x="Category",
        y="Rating",
        box=True,  # Show box plot inside the violin
        points="all",  # Show all data points
        title="Distribution of Ratings for Each App Category (Filtered)",
        color="Category",
        width=1000,
        
        height=500
    )
    save_plot_as_html(fig13, "Apps Ratings.html", 
                      "The violin plot shows the distribution of app ratings across categories, filtered to exclude high-rated apps. Categories with wider distributions indicate inconsistent user satisfaction, while tightly packed ones suggest uniform ratings.")
   
    # Show the plot
    fig13.show()
else:
    print("Violin plot is hidden. Available only between 4 PM - 6 PM IST.")


# In[432]:


plot_containers_split=plot_containers.split('</div>')


# In[433]:


if len(plot_containers_split)>1:
    final_plot=plot_containers_split[-2]+'</div>'
else:
    final_plot=plot_containers
    


# In[434]:


dashboard_html="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset='UTF-8'>
    <meta name="viewport" content="width=device-width,initial-scale-1.0">
    <title> Google Play Store Review Analytics</title>
    <style>
        body {{
            font-family: Arial,sans-serif;
            background-color:#333;
            color:#fff;
            margin:0;
            padding:0;
        }}
        .header {{
            display:flex;
            align-items:center;
            justify-content:center;
            padding:20px;
            background-color:#444
        }}
        .header img{{
            margin:0 10px;
            height:50px;
        }}
        
        .container {{
            display:flex;
            flex-wrap:wrap;
            justify_content:center;
            padding:20px;
        }}
        .plot-container{{
            border:2px solid #555
            margin:10px;
            padding:10px;
            width:{plot_width}px;
            height:{plot_height}px;
            overflow:hidden;
            position:relative;
            cursor:pointer;
        }}
        .insights {{
            display:none;
            position:absolute;
            right:10px;
            top:10px;
            background-color:rgba(0,0,0,0.7);
            padding:5px;
            border-radius:5px;
            color:#fff;
        }}
        .plot-container:hover .insights {{
            display:block;
        }}
     </style>
        <script>
            function openPlot(filename){{
                window.open(filename, '_blank');
                }}
        </script>
      </head>
      <body>
          <div class ="header">
              <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Logo_2013_Google.png/800px-Logo_2013_Google.png" alt="Google Logo">
              <h1>Google Play Store Reviews Analytics</h1>
             <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Google_Play_Store_badge_EN.svg/1024px-Google_Play_Store_badge_EN.svg.png" alt="Google Play Store">
          </div>
          <div class="container">
              {plots}
          </div>
        </body>
        </html>
        """


# In[435]:


final_html=dashboard_html.format(plots=plot_containers,plot_width=plot_width,plot_height=plot_height)


# In[436]:


dashboard_path=os.path.join(html_files_path,"web page.html")


# In[437]:


with open(dashboard_path,"w",encoding="utf-8") as f:
    f.write(final_html)


# In[438]:


webbrowser.open('file://'+os.path.realpath(dashboard_path))


# In[439]:


final_html=dashboard_html.format(plots=plot_containers,plot_width=plot_width,plot_height=plot_height)


# In[440]:


dashboard_path=os.path.join(html_files_path,"web page.html")


# In[441]:


with open(dashboard_path,"w",encoding="utf-8") as f:
    f.write(final_html)


# In[442]:


webbrowser.open('file://'+os.path.realpath(dashboard_path))


# In[443]:


apps_df


# In[444]:


apps_df_paid = apps_df[apps_df["Type"] == "Paid"]


# In[445]:


apps_df_paid


# In[ ]:





# In[446]:


pip install statsmodels


# In[447]:


apps_df


# In[448]:


apps_df


# In[449]:


apps_df


# In[ ]:





# In[450]:


print(apps_df.dtypes)


# In[451]:


apps_df


# In[452]:


# Convert "Installs" to numeric, handling errors
apps_df["Installs"] = (
    apps_df["Installs"]
    .astype(str)  # Ensure it's a string
    .str.replace(r"[^\d]", "", regex=True)  # Remove non-numeric characters
)

# Convert to float (if possible), set errors='coerce' to avoid issues
apps_df["Installs"] = pd.to_numeric(apps_df["Installs"], errors="coerce").fillna(0)


# In[453]:


apps_df


# In[454]:


print(apps_df.dtypes)


# In[455]:


pip install pycountry


# In[456]:


import pandas as pd

# Load the dataset
apps_df = pd.read_csv("googleplaystore.csv")

# Convert "Installs" to numeric (removing commas and "+")
apps_df["Installs"] = (
    apps_df["Installs"]
    .astype(str)
    .str.replace(",", "")
    .str.replace("+", "")
    .str.strip()
)

# Handle cases where "Installs" is not numeric
apps_df["Installs"] = pd.to_numeric(apps_df["Installs"], errors="coerce")

# Convert "Price" to numeric (removing "$")
apps_df["Price"] = (
    apps_df["Price"]
    .astype(str)
    .str.replace("$", "")
    .str.strip()
)

# Handle cases where "Price" is not numeric
apps_df["Price"] = pd.to_numeric(apps_df["Price"], errors="coerce")

# Fill NaN values with 0 (optional)
apps_df["Installs"].fillna(0, inplace=True)
apps_df["Price"].fillna(0, inplace=True)

print(apps_df[["Installs", "Price"]].dtypes)  # Verify conversion


# In[457]:


apps_df_filtered = apps_df_filtered[apps_df_filtered['App'].str.contains('C', case=False, na=False)]
apps_df_filtered


# In[458]:


apps_df_filtered = apps_df_filtered[apps_df_filtered['Reviews'] >= 10]


# In[459]:


apps_df_filtered


# In[460]:


apps_df_filtered = apps_df_filtered[apps_df_filtered['Rating'] < 4.0]
apps_df_filtered


# In[461]:


print(apps_df_filtered.head(100))  # Prints the first 50 rows


# In[462]:


apps_df_filtered = apps_df_filtered[apps_df_filtered['App'].str.contains('C', case=False, na=False)]
apps_df_filtered


# In[ ]:




