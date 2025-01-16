#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import yfinance as yf
import smtplib
import warnings
from io import BytesIO
from datetime import datetime, timedelta
import plotly.graph_objects as go
warnings.filterwarnings('ignore')
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage


# In[2]:


def fetch_historical_prices_batch(symbol, dates):
    try:
        unique_dates = pd.to_datetime(dates).drop_duplicates().sort_values()
        start_date = unique_dates.min()
        end_date = unique_dates.max() + pd.Timedelta(days=1)

        stock = yf.Ticker(symbol)
        history = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

        history['Date'] = history.index.date
        prices = history[['Date', 'Close']].set_index('Date')['Close']
        return prices.to_dict()
    
    except Exception as e:
        return {}


# In[3]:


def fetch_current_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        return stock.history(period="1d")["Close"].iloc[-1]
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None


# In[4]:


def calculate_holdings(group):
    buy_units = group.loc[group["Action"] == "Buy", "Units"].sum()
    sell_units = group.loc[group["Action"] == "Sell", "Units"].sum()
    remaining_units = buy_units - sell_units

    # Total investment: sum of buy amounts minus sold stock amounts
    buy_amount = group.loc[group["Action"] == "Buy", "Total Amount"].sum()
    sell_amount = (group.loc[group["Action"] == "Sell", "Units"] * 
                   group.loc[group["Action"] == "Sell", "Price per Unit"]).sum()

    total_investment = buy_amount - sell_amount

    return pd.Series({"Total Units": remaining_units, "Total Investment": total_investment})


# In[5]:


def piechart(dff):
    fig = go.Figure(data=[go.Pie(labels=dff["Stock Name"], values=dff["Total Investment"], hole=0.5, textfont=dict(size=14), direction='clockwise',
                                 sort=True)])
    fig.update_layout(title=dict(text="Current Holdings", x=0.5, xanchor="center", font=dict(size=26)), margin=dict(t=40,b=10,l=5,r=5),
                      legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"), showlegend=True, width=600, height=550)
    return fig


# In[6]:


def barchart(stocklist, dayreturns):
    dayreturnsstr = [str(i)+"%" for i in dayreturns]
    colors = ['green' if i > 0 else 'red' for i in dayreturns]
    fig = go.Figure(data=[go.Bar(x=stocklist, y=dayreturns, text=dayreturnsstr, marker=dict(color=colors))])
    fig.update_layout(title=dict(text="Today's Returns", x=0.5, xanchor="center", font=dict(size=22)), showlegend=False, width=750, height=450,
                      xaxis=dict(title="Stocks",tickangle=0), margin=dict(t=40,b=10,l=15,r=10))
    return fig


# In[7]:


def barchart2(stocklist, dff):
    ovrlreturnslist = []
    for i in range(len(stocklist)):
        dfff = dff[dff['Stock Name']==stocklist[i]].reset_index(drop=True)
        ovrlreturnslist.append(np.round(dfff['Returns'].mean(),1))
    ovrlreturnsstrlist = [str(i)+"%" for i in ovrlreturnslist]
    colors = ['green' if i > 0 else 'red' for i in ovrlreturnslist]
    fig = go.Figure(data=[go.Bar(x=stocklist, y=ovrlreturnslist, text=ovrlreturnsstrlist, marker=dict(color=colors))])
    fig.update_layout(title=dict(text="Overall Returns", x=0.5, xanchor="center", font=dict(size=22)), showlegend=False, width=750, height=450,
                      xaxis=dict(title="Stocks",tickangle=0), margin=dict(t=40,b=10,l=15,r=10))
    return fig


# In[8]:


def linechart(inv_jrny_df):
    fig_inv = go.Figure()
    cur_date = inv_jrny_df['Date'][len(inv_jrny_df)-1]
    cur_value = inv_jrny_df['Net Value'][len(inv_jrny_df)-1].round(2)
    inv_value = inv_jrny_df['Cumulative Investment'][len(inv_jrny_df)-1].round(2)
    tomorrow = (datetime.today() + timedelta(days=1))
    inv_jrny_df.loc[len(inv_jrny_df),'Date'] = tomorrow
    inv_jrny_df['Date'] = inv_jrny_df['Date'].dt.date
    if cur_value>=inv_value:
        cur_text_pos = 'top left'
        inv_text_pos = 'bottom left'
    else:
        cur_text_pos = 'bottom left'
        inv_text_pos = 'top left'
    fig_inv.add_trace(go.Scatter(x=inv_jrny_df['Date'], y=inv_jrny_df['Cumulative Investment'], mode='lines', name='Investment',
                                 line=dict(color='blue')))
    fig_inv.add_trace(go.Scatter(x=inv_jrny_df['Date'], y=inv_jrny_df['Net Value'], mode='lines', name='Performance',
                                 line=dict(color='orange')))
    fig_inv.add_trace(go.Scatter(x=[cur_date], y=[cur_value], mode='markers+text', name='Current Value', marker=dict(color='orange', size=13),
                                 text=[f"Cur: ₹{cur_value}"], textposition=cur_text_pos, textfont=dict(size=15)))
    fig_inv.add_trace(go.Scatter(x=[cur_date], y=[inv_value], mode='markers+text', name='Invested Value', marker=dict(color='blue', size=13),
                                 text=[f"Inv: ₹{inv_value}"], textposition=inv_text_pos, textfont=dict(size=15)))
    fig_inv.update_layout(title=dict(text="Stock Investment Journey", x=0.5, xanchor='center', font=dict(size=22)), xaxis_title="Date",
                          yaxis_title="Rupees (₹)", template="plotly_white", xaxis=dict(showgrid=True), yaxis=dict(showgrid=True), showlegend=False,
                          width=900, height=500, margin=dict(t=40,b=10,l=15,r=10), xaxis_range=[pd.Timestamp("2024-11-01"),
                          pd.Timestamp((datetime.today()+timedelta(days=1)).strftime("%Y-%m-%d"))])
    return fig_inv


# In[9]:


def stockperformance(dff, stocklist):
    string = ""
    Dreturnslist = []
    for stock in stocklist:
        string += stock+":  "
        minindex = dff[dff['Stock Name']==stock].index.max()-1
        maxindex = dff[dff['Stock Name']==stock].index.max()
        returns = np.round(dff['Net Value'][maxindex]-dff['Net Value'][minindex],2)
        Dreturnsperc = np.round(100*(dff['Net Value'][maxindex]-dff['Net Value'][minindex])/dff['Net Value'][minindex],1)
        Dreturnslist.append(Dreturnsperc)
        string += f"+₹{str(returns)}  (+{Dreturnsperc}%)<br>" if returns>0 else f"-₹{str(abs(returns))}  ({Dreturnsperc}%)<br>"
    return string, Dreturnslist


# In[10]:


def send_email_with_chart(sender_email, recipient_email, subject, body_text, body_color, app_password, dataframe1, dataframe2, mystocks, invjrnydf):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    stockprfm, dayreturnslist = stockperformance(dataframe2, mystocks)
    stockanalysisdburl = "https://stockanalysispy-xfufzzdrzumnhucw3emrcw.streamlit.app/"
    
    buf1 = BytesIO()
    fig_pie = piechart(dataframe1)
    fig_pie.write_image(buf1, format="png")
    buf1.seek(0)
    buf2 = BytesIO()
    fig_bar = barchart(mystocks,dayreturnslist)
    fig_bar.write_image(buf2, format="png")
    buf2.seek(0)
    buf3 = BytesIO()
    fig_bar2 = barchart2(mystocks,dataframe1)
    fig_bar2.write_image(buf3, format="png")
    buf3.seek(0)
    buf4 = BytesIO()
    fig_inv = linechart(invjrnydf)
    fig_inv.write_image(buf4, format="png")
    buf4.seek(0)

    html_body = f"""
    <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.5; margin: 0; padding: 0;">
            <h1 style="font-size: 20px; color: {body_color};">{body_text}</h1>
            <h4 style="font-size: 16px; color: #555;">{stockprfm}</h4>
            <h4 style="font-size: 14px; color: #555;">
                Stock Analysis Dashboard link: <a href="{stockanalysisdburl}" style="color: #1a73e8;">{stockanalysisdburl}</a>
            </h4>
            <img src="cid:bar_chart" alt="Bar Chart" style="display: block; width: 750px; height: auto; margin: 0 auto;">
            <img src="cid:line_chart" alt="Line Chart" style="display: block; width: 900px; height: auto; margin: 0 auto;">
            <table style="width: 100%; text-align: center; border-spacing: 20px;">
                <tr>
                    <td><img src="cid:bar_chart_" alt="Bar Chart" style="display: block; width: 750px; height: auto; margin: 0 auto;"></td>
                    <td><img src="cid:pie_chart" alt="Pie Chart" style="display: block; width: 300px; height: auto; margin: 0 auto;"></td>
                </tr>
            </table>
        </body>
    </html>
    """
    
    msg.attach(MIMEText(html_body, 'html'))
    img = MIMEImage(buf1.getvalue())
    img.add_header('Content-ID', '<pie_chart>')
    msg.attach(img)
    img = MIMEImage(buf2.getvalue())
    img.add_header('Content-ID', '<bar_chart>')
    msg.attach(img)
    img = MIMEImage(buf4.getvalue())
    img.add_header('Content-ID', '<line_chart>')
    msg.attach(img)
    img = MIMEImage(buf3.getvalue())
    img.add_header('Content-ID', '<bar_chart_>')
    msg.attach(img)
    
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, app_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
            print(f"Email sent to {recipient_email} successfully.")
    except Exception as e:
        print(f"Error: {e}")


# In[11]:


df = pd.read_excel("Stock_Database.xlsx")
df = df.drop(columns=['S.No.'])
mystocks = df['Stock Name'].unique().tolist()
df["Total Amount"] = df["Units"] * df["Price per Unit"]
df["Cumulative Investment"] = df["Total Amount"].cumsum()
holdings = df.groupby("Stock Name").apply(calculate_holdings).reset_index()
holdings = holdings[holdings["Total Units"]>0]
stock_symbols = df[["Stock Name", "Stock Symbol"]].drop_duplicates()
holdings = holdings.merge(stock_symbols, on="Stock Name")
holdings["Current Stock Price"] = holdings["Stock Symbol"].apply(fetch_current_price)
holdings["Current Value"] = holdings["Current Stock Price"]*holdings["Total Units"]
holdings["Returns"] = 100*(holdings["Current Value"]-holdings["Total Investment"])/holdings["Total Investment"]

start_date = df['Date'].min().date()
end_date = datetime.today()
date_range = pd.date_range(start=start_date, end=end_date)
dff = pd.DataFrame({'Date': date_range})

stockdf_list = []
for i in range(len(mystocks)):
    dfsub = df[df['Stock Name'] == mystocks[i]].reset_index(drop=True)
    stockdf = pd.merge(dff, dfsub, on='Date', how='left')
    stockdf = stockdf[stockdf[stockdf['Stock Name'].notna()].index.min():].reset_index(drop=True)
    stockdf[['Sector', 'Stock Name', 'Stock Symbol']] = stockdf[['Sector', 'Stock Name', 'Stock Symbol']].fillna(method='ffill')
    stockdf['Cumulative Units'] = stockdf['Units'].cumsum()

    stock_symbol = stockdf['Stock Symbol'].iloc[0]
    dates = stockdf['Date']
    historical_prices = fetch_historical_prices_batch(stock_symbol, dates)
    stockdf["Closing Price"] = stockdf["Date"].map(historical_prices)
    stockdf[['Cumulative Investment','Cumulative Units','Closing Price']] = stockdf[['Cumulative Investment','Cumulative Units','Closing Price']
                                                                                   ].fillna(method='ffill')
    stockdf['Net Value'] = stockdf['Cumulative Units']*stockdf['Closing Price']
    stockdf_list.append(stockdf)
stockdffull = pd.concat(stockdf_list,ignore_index=True)
investment_journey_df = stockdffull.groupby('Date').agg({'Cumulative Investment':'max','Net Value':'sum'}).reset_index()


# In[12]:


portfolio_performance = np.round(investment_journey_df['Net Value'][len(investment_journey_df)-1
                                 ]-investment_journey_df['Net Value'][len(investment_journey_df)-2],2)
portfolio_performance_str = "+₹"+str(portfolio_performance) if portfolio_performance>0 else "-₹"+str(abs(portfolio_performance))


# In[13]:


sender_email = "stockanalysiswithesh@gmail.com"
recipient_email = "muruganeswaraprasath@gmail.com"
subject = f"Stock Portfolio performance - {datetime.today().strftime('%d-%b-%Y')}"
body = f"Today's Stock Portfolio performance: {portfolio_performance_str}/-"
bodycolor = "#41c443" if portfolio_performance>0 else "#e3343a"
app_password = "zseo rzfw zgwr aduv"


# In[14]:


send_email_with_chart(sender_email, recipient_email, subject, body, bodycolor, app_password, holdings, stockdffull, mystocks, investment_journey_df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




