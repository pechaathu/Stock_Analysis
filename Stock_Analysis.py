#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import io
import re
import sys
import time
import smtplib
import requests
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from io import BytesIO
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from nselib import capital_market
from bs4 import BeautifulSoup as bs
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from contextlib import redirect_stderr
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
pio.templates["custom_template"] = pio.templates["plotly"]
pio.templates["custom_template"]["layout"]["colorway"] = px.colors.qualitative.Plotly
pio.templates.default = "custom_template"
color_palette = px.colors.qualitative.Plotly
warnings.filterwarnings('ignore')


# # Page Settings

# In[ ]:


st.set_page_config(layout='wide')


# In[ ]:


def page_navigation():
    st.markdown("""<style>.menu-container {display: flex; justify-content: center; align-items: center; margin-top: 1px;
                }.menu-button {padding: 0.2px 0.60x; font-size: 24px; font-weight: bold; background-color: #007BFF; color: white; border: none; 
                border-radius: 0.1px; margin: 0 0.2px; cursor:pointer; text-align:center; text-decoration: none; transition: background-color 0.3s ease;
                }.menu-button.selected {background-color: #0056b3;}.menu-button:hover {background-color: #0056b3;}</style>""", unsafe_allow_html=True)
    st.markdown('<div class="menu-container">', unsafe_allow_html=True)
    colmenu1, colmenu2, colmenu3, colmenu4, colmenu5, colmenu6, colmenu7 = st.columns([1,1,1,1,1,1,1])
    with colmenu2:
        if st.button("Stock Analysis", key="stockanalysis", use_container_width=True):
            st.session_state.page = "Stock Analysis"
    with colmenu3:
        if st.button("Index Performance", key="indexperformance", use_container_width=True):
            st.session_state.page = "Index Performance"
    with colmenu4:
        if st.button("Strategy Indices", key="strategyindices", use_container_width=True):
            st.session_state.page = "Strategy Indices"
    with colmenu5:
        if st.button("FII DII Activity", key="fiidiiactivity", use_container_width=True):
            st.session_state.page = "FII DII Activity"
    with colmenu6:
        if st.button("Union Budget", key="unionbudget", use_container_width=True):
            st.session_state.page = "Union Budget"
    st.markdown('</div>', unsafe_allow_html=True)


# # Functions

# In[ ]:


def calculate_holdings(group):
    buy_units = group.loc[group["Action"] == "Buy", "Units"].sum()
    sell_units = group.loc[group["Action"] == "Sell", "Units"].sum()
    remaining_units = buy_units - sell_units

    buy_amount = group.loc[group["Action"] == "Buy", "Total Amount"].sum()
    sell_amount = (group.loc[group["Action"] == "Sell", "Units"] * 
                   group.loc[group["Action"] == "Sell", "Price per Unit"]).sum()

    total_investment = buy_amount - sell_amount

    return pd.Series({"Total Units": remaining_units, "Total Investment": total_investment})


# In[ ]:


def fetch_current_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        return stock.history(period="1d")["Close"].iloc[-1]
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None


# In[ ]:


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


# In[ ]:


def webscrap(webcontent, label):
    label_tag = webcontent.find(string=lambda text: text and label in text)
    if label_tag:
        parent = label_tag.parent
        value_tag = parent.find_next(class_='number')
        if value_tag:
            value_string = value_tag.text.strip()
            return value_string


# # Login Page

# In[ ]:


def load_valid_emails():
    try:
        df = pd.read_excel("AccessList.xlsx")
        return set(df['MailID'].str.strip().str.lower())
    except Exception as e:
        st.error(f"Error loading email list: {e}")
        return set()


# In[ ]:


def mail(recipient_email, otpnum):
    msg = MIMEMultipart()
    msg['From'] = "stockanalysiswithesh@gmail.com"
    msg['To'] = recipient_email
    msg['Subject'] = "OTP for Logging into Stock Analysis dashboard"

    msg.attach(MIMEText(f'OTP: {otpnum}', 'plain'))
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login("stockanalysiswithesh@gmail.com", "zseo rzfw zgwr aduv")
            text = msg.as_string()
            server.sendmail("stockanalysiswithesh@gmail.com", recipient_email, text)
    except Exception as e:
        st.write(f"Error: {e}")


# In[ ]:


def login_page(valid_emails):
    st.markdown("""<style>.title {text-align:center; font-size:34px; font-weight:bold;}</style><div class="title">Stock Analysis dashboard<br></div>""",
                unsafe_allow_html=True)
    logincol1, logincol2, logincol3 = st.columns([1,1,1])
    logincol10, logincol11, logincol12 = st.columns([1.15,0.35,1])
    logincol4, logincol5, logincol6 = st.columns([1,1,1])
    logincol7, logincol8, logincol9 = st.columns([1.2,0.3,1])
    with logincol2:
        email = st.text_input("Email ID", placeholder="Enter your email")
    with logincol11:
        if st.button("Generate OTP"):
            if email.strip().lower() in valid_emails:
                otp = np.random.randint(100000, 1000000)
                st.session_state.generated_otp = otp
                st.session_state.email = email
                mail(email, otp)
            else:
                st.error("You don't have access to the dashboard. Please contact Admin.")

    if "generated_otp" in st.session_state:
        with logincol5:
            otpentry = st.text_input("OTP", placeholder="Enter the OTP received in mail")
        with logincol8:
            if st.button("Login"):
                if otpentry == str(st.session_state.generated_otp):
                    st.success("Login successful!")
                    time.sleep(3)
                    st.session_state.page = "Stock Analysis"
                    st.experimental_rerun()
                else:
                    st.error("Invalid OTP!...")


# # Stock Analysis Page

# ### Stock Filters & Price Chart

# In[ ]:


def prologue(companies):
    comparison = False
    headcol1, headcol2, headcol3 = st.columns([6.5,1.5,2])
    industries = list(companies.keys())
    with headcol1:
        industry = st.radio("Select Sector:", industries, index=0, horizontal=True)
    stocks = list(companies[industry].keys())
    if industry!="Index":
        with headcol2:
            st.markdown('<p style="font-size:14px;">Index Comparison:</p>',unsafe_allow_html=True)
            comparison = st.toggle("Yes", value=False)
        if comparison:
            with headcol3:
                selected_index = st.selectbox("Select the Index:", companies['Index'].keys(), index=0)
    
    col1, col2, col3, col4 = st.columns([2.25, 2, 1.5, 1.5])
    
    with col1:
        stock_name = st.selectbox("Select Stock:", stocks)
        
    with col2:
        stock_symbol = companies[industry][stock_name]
        st.text_input("Stock Symbol:", stock_symbol, disabled=True)
    
    stock_data = yf.download(stock_symbol)
    
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = ['_'.join(col).strip() for col in stock_data.columns.values]
    stock_data['Date'] = stock_data.index
    stock_data = stock_data.reset_index(drop=True)
    stock_data['50_DMA'] = stock_data['Close_'+stock_symbol].rolling(window=50).mean()
    stock_data['200_DMA'] = stock_data['Close_'+stock_symbol].rolling(window=200).mean()
    
    alltimehigh = np.round(max(stock_data[['High_'+stock_symbol,'Low_'+stock_symbol,'Open_'+stock_symbol,'Close_'+stock_symbol]].max().tolist()),2)
    alltimelow = np.round(min(stock_data[['High_'+stock_symbol,'Low_'+stock_symbol,'Open_'+stock_symbol,'Close_'+stock_symbol]].min().tolist()),2)
    high52week = np.round(stock_data[stock_data['Date']>=(datetime.today()-timedelta(weeks=52))]['High_'+stock_symbol].max(),2)
    low52week = np.round(stock_data[stock_data['Date']>=(datetime.today()-timedelta(weeks=52))]['Low_'+stock_symbol].min(),2)
    high3month = np.round(stock_data[stock_data['Date']>(datetime.today()-timedelta(weeks=12))]['High_'+stock_symbol].max(),2)
    low3month = np.round(stock_data[stock_data['Date']>(datetime.today()-timedelta(weeks=12))]['Low_'+stock_symbol].min(),2)
    
    default_start_date = (stock_data['Date'].max()).to_pydatetime()-timedelta(days=1)
    
    col5, col6, col7 = st.columns([2.25,1.5,1])
    
    date_ranges = ["1D", "1W", "1M", "1Y", "3Y", "5Y", "Max", "Custom"]
    with col5:
        date_range = st.radio("Select Date Range:", date_ranges, index=3, horizontal=True)
    
    if date_range == "1D":
        start_date = default_start_date
    elif date_range == "1W":
        start_date = default_start_date - timedelta(weeks=1)
    elif date_range == "1M":
        start_date = default_start_date - timedelta(days=30)
    elif date_range == "1Y":
        start_date = default_start_date - timedelta(days=365)
    elif date_range == "3Y":
        start_date = default_start_date - timedelta(days=3 * 365)
    elif date_range == "5Y":
        start_date = default_start_date - timedelta(days=5 * 365)
    elif date_range == "Max":
        start_date = (stock_data['Date'].min()).to_pydatetime()
    
    with col3:
        if date_range == "Custom":
            start_date = datetime.combine(st.date_input("Start Date", pd.to_datetime("2024-01-01")), datetime.min.time())
        else:
            st.text_input("Start Date", start_date.strftime("%d-%b-%Y"), disabled=True)
    
    with col4:
        if date_range == "Custom":
            end_date = datetime.combine(st.date_input("End Date", pd.to_datetime("today")), datetime.min.time())
        else:
            st.text_input("End Date", (stock_data['Date'].min()).to_pydatetime().strftime("%d-%b-%Y"), disabled=True)
    
    Alltimehigh = stock_data['High_'+stock_symbol].max()
    varalltimehigh = np.round(100*(stock_data['Close_'+stock_symbol][len(stock_data)-1]-Alltimehigh)/Alltimehigh,2)
    High52week = stock_data[stock_data['Date']>=((stock_data['Date'].max()).to_pydatetime()-timedelta(days=365))]['High_'+stock_symbol].max()
    varhigh52week = np.round(100*(stock_data['Close_'+stock_symbol][len(stock_data)-1]-High52week)/High52week,2)
    High52weekindex = stock_data[(stock_data['Date']>=((stock_data['Date'].max()).to_pydatetime()-timedelta(days=365)))&
                                 (stock_data['High_'+stock_symbol]==High52week)].index.tolist()[0]
    stock_data_sub = stock_data[High52weekindex:].copy().reset_index(drop=True)
    Low52week = stock_data_sub[stock_data_sub['Date']>=((stock_data_sub['Date'].max()).to_pydatetime()-timedelta(days=365))]['Low_'+stock_symbol].min()
    varlow52week = np.round(100*(Low52week-High52week)/High52week,2)
    
    stock_data = stock_data[stock_data['Date']>=start_date].reset_index(drop=True)

    absolute_returns = np.round(100*(stock_data['Close_'+stock_symbol][len(stock_data)-1]-stock_data['Close_'+stock_symbol][0])/stock_data['Close_'+stock_symbol][0],2)
    if absolute_returns>0:
        fontcolor = 'green'
    elif absolute_returns<0:
        fontcolor = 'red'
    else:
        fontcolor = 'white'
    if absolute_returns>0:
        absolute_returns = "+"+str(absolute_returns)+"%" 
    else:
        absolute_returns = str(absolute_returns)+"%"

    if varalltimehigh>=0:
        fontcolor2 = 'green'
    else:
        fontcolor2 = 'red'
    varalltimehigh = str(varalltimehigh)+"%"

    if varhigh52week>=0:
        fontcolor3 = 'green'
    else:
        fontcolor3 = 'red'
    varhigh52week = str(varhigh52week)+"%"

    if varlow52week>=0:
        fontcolor3 = 'green'
    else:
        fontcolor3 = 'red'
    varlow52week = str(varlow52week)+"%"
        
    if comparison==True:
        bm_index_symbol = companies['Index'][selected_index]
        bm_data = yf.download(bm_index_symbol)
        bm_data.columns = ['_'.join(col).strip() for col in bm_data.columns.values]
        bm_data['Date'] = bm_data.index
        bm_data = bm_data[bm_data['Date']>=start_date].reset_index(drop=True)
        bm_data = bm_data.reset_index(drop=True)
        stock_data = pd.merge(stock_data, bm_data, how="left", on='Date')
        stock_data['Variation_'+bm_index_symbol] = np.round(100*(
                            stock_data['Close_'+bm_index_symbol]-stock_data['Close_'+bm_index_symbol][0])/stock_data['Close_'+bm_index_symbol][0],2)
        stock_data['Variation_'+stock_symbol] = np.round(100*(
                                    stock_data['Close_'+stock_symbol]-stock_data['Close_'+stock_symbol][0])/stock_data['Close_'+stock_symbol][0],2)
        
        fig_compline = go.Figure()
        fig_compline.add_trace(go.Scatter(x=stock_data['Date'],y=stock_data['Variation_'+stock_symbol],mode='lines',name=stock_name,
                                          line=dict(color='blue')))
        fig_compline.add_trace(go.Scatter(x=stock_data['Date'],y=stock_data['Variation_'+bm_index_symbol],mode='lines',name=selected_index,
                                          line=dict(color='yellow')))
        fig_compline.update_layout(title=dict(text=f"Benchmark Index ({selected_index})  v/s  "+stock_name,x=0.5,xanchor='center'), xaxis_title="Date",
                                   yaxis_title="% variation", template="plotly_white", xaxis=dict(showgrid=True), yaxis=dict(showgrid=True),
                                   width=1350, height=500)
        st.plotly_chart(fig_compline, use_container_width=True)
    
    else:
        with col6:
            st.markdown('<p style="font-size:14px;">Add on:</p>',unsafe_allow_html=True)
            placeholder = st.empty()
            with placeholder.container():
                cb1, cb2, cb3= st.columns([1,1.2,1.3])
                with cb1:
                    dma50 = st.checkbox("50 DMA")
                with cb2:
                    dma200 = st.checkbox("200 DMA")
    
        with col7:
            chart_type = st.radio("Chart type:", ["Line","Candle"], index=0, horizontal=True)
        
        if chart_type=="Line":
            if stock_data['Close_'+stock_symbol][0] <= stock_data['Close_'+stock_symbol][len(stock_data)-1]:
                chart_color = 'green'
            else:
                chart_color = 'red'
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close_'+stock_symbol],mode='lines',name='Price',
                                          line=dict(color=chart_color)))
            if dma50==True:
                fig_line.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['50_DMA'], mode='lines', name='50 DMA', line=dict(color='yellow')))
            if dma200==True:
                fig_line.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['200_DMA'], mode='lines', name='200 DMA', line=dict(color='blue')))
            max_value = stock_data['Close_'+stock_symbol].max().round(2)
            min_value = stock_data['Close_'+stock_symbol].min().round(2)
            cur_value = stock_data['Close_'+stock_symbol][len(stock_data)-1].round(2)
            max_date = stock_data.loc[stock_data['Close_'+stock_symbol].idxmax(),'Date']
            min_date = stock_data.loc[stock_data['Close_'+stock_symbol].idxmin(),'Date']
            cur_date = stock_data['Date'][len(stock_data)-1]
            fig_line.add_trace(go.Scatter(x=[max_date], y=[max_value], mode='markers+text', name='Max Price', marker=dict(color=chart_color, size=13),
                                          text=[f"Max: ₹{max_value}"], textposition='top center', textfont=dict(size=15)))
            fig_line.add_trace(go.Scatter(x=[min_date], y=[min_value], mode='markers+text', name='Min Price', marker=dict(color=chart_color, size=13),
                                          text=[f"Min: ₹{min_value}"], textposition='bottom center', textfont=dict(size=15)))
            fig_line.add_trace(go.Scatter(x=[cur_date], y=[cur_value], mode='markers+text', name='Curr Price', marker=dict(color='white', size=13),
                                          text=[f"Curr: ₹{cur_value}"], textposition='top center', textfont=dict(size=15)))
            fig_line.update_layout(title=dict(text=stock_name, x=0.5, xanchor='center'), xaxis_title="Date",yaxis_title="Close Price (INR)",
                                   template="plotly_white", xaxis=dict(showgrid=True), yaxis=dict(showgrid=True), width=1350, height=500)
            st.plotly_chart(fig_line, use_container_width=True)
    
        elif chart_type=="Candle":
            fig_candle = go.Figure()
            fig_candle.add_trace(go.Candlestick(x=stock_data['Date'], open=stock_data['Open_'+stock_symbol], high=stock_data['High_'+stock_symbol],
                                                low=stock_data['Low_'+stock_symbol], close=stock_data['Close_'+stock_symbol], name='Candlestick'))
            if dma50==True:
                fig_candle.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['50_DMA'], mode='lines', name='50 DMA', line=dict(color='yellow')))
            if dma200==True:
                fig_candle.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['200_DMA'], mode='lines', name='200 DMA', line=dict(color='blue')))
            fig_candle.update_layout(title=dict(text=stock_name, x=0.5, xanchor='center'), xaxis_title="Date", yaxis_title="Price (INR)", 
                                     template="plotly_white", xaxis=dict(showgrid=True,rangeslider=dict(visible=False)), yaxis=dict(showgrid=True),
                                     height=500)
            st.plotly_chart(fig_candle, use_container_width=True)


    col8, col9, col10, col11 = st.columns([1,1,1,1])
    with col8:
        st.markdown("""<p style="text-align: center; font-size: 14px;">Absolute Returns</p>""", unsafe_allow_html=True)
        st.markdown("""<p style="text-align: center; font-size: 25px; font-weight: bold; color: {fontcolor};">{absolute_returns}</p>""".format(fontcolor=fontcolor, absolute_returns=absolute_returns), unsafe_allow_html=True)
    with col9:
        st.markdown("""<p style="text-align: center; font-size: 14px;">Current status from its All time High</p>""", unsafe_allow_html=True)
        st.markdown("""<p style="text-align: center; font-size: 25px; font-weight: bold; color: {fontcolor2};">{varalltimehigh}</p>""".format(fontcolor2=fontcolor2, varalltimehigh=varalltimehigh), unsafe_allow_html=True)
    with col10:
        st.markdown("""<p style="text-align: center; font-size: 14px;">Current status from its 52 week High</p>""", unsafe_allow_html=True)
        st.markdown("""<p style="text-align: center; font-size: 25px; font-weight: bold; color: {fontcolor3};">{varhigh52week}</p>""".format(fontcolor3=fontcolor3, varhigh52week=varhigh52week), unsafe_allow_html=True)
    with col11:
        st.markdown("""<p style="text-align: center; font-size: 14px;">Total Fall from its 52 week High</p>""", unsafe_allow_html=True)
        st.markdown("""<p style="text-align: center; font-size: 25px; font-weight: bold; color: {fontcolor4};">{varlow52week}</p>""".format(fontcolor4='red', varlow52week=varlow52week), unsafe_allow_html=True)
    
    return stock_name, stock_symbol, industry, alltimehigh, alltimelow, high52week, low52week, high3month, low3month


# ### Get Contents

# In[ ]:


def getcontents(stockurls, stock_name):
    stock_url = stockurls[stock_name]
    response = requests.get(stock_url)
    content = bs(response.content, 'html.parser')
    return content


# ### Basic Info

# In[ ]:


def basicinfo(content, industry, alltimehigh, alltimelow, high52week, low52week, high3month, low3month):
    if industry != "Index":
        capital = webscrap(content,'Market Cap')
        capnum = int(''.join(capital.split(',')))
        stockcmp = webscrap(content,'Current Price')
        stockcmp = ''.join(stockcmp.split(',')) if stockcmp is not None else stockcmp
        stockbv = webscrap(content,'Book Value')
        stockbv = ''.join(stockbv.split(',')) if stockbv is not None else stockbv
        stockpe = webscrap(content,'Stock P/E')
        stockpe = stockpe if stockpe is not None else webscrap(content,'P/E')
        stockpb = webscrap(content,'Price to Book value')
        stockpb = stockpb if stockpb is not None else webscrap(content,'Price to book value')
        stockpb = stockpb if stockpb is not None else (
                            np.round(float(stockcmp)/float(stockbv), 2) if stockbv is not None and stockcmp != '' and stockbv != '' else None)
        stockroe = webscrap(content,'ROE')+"%" if webscrap(content,'ROE') is not None else webscrap(content,'ROE')
        stockroce = webscrap(content,'ROCE')+"%" if webscrap(content,'ROCE') is not None else webscrap(content,'ROCE')
        stockdivy = webscrap(content,'Dividend Yield')+"%" if webscrap(content,'Dividend Yield') is not None else webscrap(content,'Dividend Yield')
        
        scrapcol1, scrapcol2, scrapcol3, scrapcol4, scrapcol5, scrapcol6, scrapcol7, scrapcol8 = st.columns([0.8,1,0.6,0.6,0.6,0.65,0.65,0.1])
        scrapcol9, scrapcol10, scrapcol11, scrapcol12, scrapcol13, scrapcol14 = st.columns([1,1.6,1,1.6,1,1.25])
        with scrapcol1:
            st.metric("Market Segment",
            "Mega Cap" if capnum>1000000 else "Large Cap" if capnum>20000 else "Mid Cap" if capnum>5000 else "Small Cap" if capnum>1000 else "Micro Cap")
        with scrapcol2:
            st.metric("Market Cap", capital+" Cr")
        with scrapcol3:
            st.metric("P/E ratio", stockpe)
        with scrapcol4:
            st.metric("P/B ratio", stockpb)
        with scrapcol5:
            st.metric("ROE (%)", stockroe)
        with scrapcol6:
            st.metric("ROCE (%)", stockroce)
        with scrapcol7:
            st.metric("Div. Yield (%)", stockdivy)
        with scrapcol9:
            st.metric("All time High", "₹"+str(alltimehigh))
        with scrapcol10:
            st.metric("All time Low", "₹"+str(alltimelow))
        with scrapcol11:
            st.metric("52 Week High", "₹"+str(high52week))
        with scrapcol12:
            st.metric("52 Week Low", "₹"+str(low52week))
        with scrapcol13:
            st.metric("3 Month High", "₹"+str(high3month))
        with scrapcol14:
            st.metric("3 Month Low", "₹"+str(low3month))
        st.markdown("<br>", unsafe_allow_html=True)


# ### About, Pros, Cons

# In[ ]:


def proscons(content, industry):
    if industry != "Index":
        about_string = content.find('div', class_='title', string='About')
        if about_string:
            about_parent = about_string.parent
            about_list = about_parent.find_next('p')
            if about_list:
                sup_tag = about_list.find('sup')
                if sup_tag:
                    sup_tag.decompose()
                about_sentence = about_list.text.strip()
        pros_sentence = ''
        pros_string = content.find('p', class_='title', string='Pros')
        if pros_string:
            pros_parent = pros_string.parent
            pros_list = pros_parent.find_next('ul')
            if pros_list:
                pros = [li.text.strip() for li in pros_list.find_all('li')]
                for i in range(len(pros)):
                    pros_sentence += str(i+1)+". "+pros[i]+"<br>"
        cons_sentence = ''
        cons_string = content.find('p', class_='title', string='Cons')
        if cons_string:
            cons_parent = cons_string.parent
            cons_list = cons_parent.find_next('ul')
            if cons_list:
                cons = [li.text.strip() for li in cons_list.find_all('li')]
                for i in range(len(cons)):
                    cons_sentence += str(i+1)+". "+cons[i]+"<br>"
    
        with st.expander("Details", expanded=True):
            containercol1, containercol2, containercol3 = st.columns([1,1,1])
            with containercol1:
                about_container = st.container(border=True)
                with about_container:
                    st.markdown(f"""<div style="background-color: #2e88bf; font-size:16px;">About:</div>""", unsafe_allow_html=True)
                    st.markdown(f"""<div style="background-color: #2e88bf; font-size:13px;">{about_sentence}</div>""", unsafe_allow_html=True)
            with containercol2:
                pros_container = st.container(border=True)
                with pros_container:
                    st.markdown(f"""<div style="background-color: #0ac404; font-size:16px;">Pros:</div>""", unsafe_allow_html=True)
                    st.markdown(f"""<div style="background-color: #0ac404; font-size:14px;">{pros_sentence}</div>""", unsafe_allow_html=True)
            with containercol3:
                cons_container = st.container(border=True)
                with cons_container:
                    st.markdown(f"""<div style="background-color: #d12a3d; font-size:16px;">Cons:</div>""", unsafe_allow_html=True)
                    st.markdown(f"""<div style="background-color: #d12a3d; font-size:14px;">{cons_sentence}</div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)


# ### Profit results

# In[ ]:


def profitloss(content, industry):
    if industry != "Index":
        quarterly_profit_content = content.find('h2', string="Quarterly Results")
        if quarterly_profit_content:
            quaterly_profit_table = quarterly_profit_content.find_next('table', class_='data-table responsive-text-nowrap')
            if quaterly_profit_table:
                quarterly_profit_headers = [th.get_text(strip=True) for th in quaterly_profit_table.find('thead').find_all('th')]
            rows = []
            for row in quaterly_profit_table.find("tbody").find_all("tr"):
                cols = [col.text.strip() for col in row.find_all("td")]
                rows.append(cols)
            df_quarterly_results = pd.DataFrame(rows, columns=quarterly_profit_headers)
            df_quarterly_results = df_quarterly_results.drop(index=len(df_quarterly_results)-1)
            df_quarterly_results = df_quarterly_results.rename(columns={'':'Type'})
            df_quarterly_results['Type'] = df_quarterly_results['Type'].str.rstrip('\xa0+')
            df_quarterly_results = df_quarterly_results.drop(columns=[df_quarterly_results.columns.tolist()[1]])
            df_quarterly_results = df_quarterly_results[(df_quarterly_results['Type']=="Sales")|(df_quarterly_results['Type']=="Revenue")|
                                                        (df_quarterly_results['Type']=="Expenses")|(df_quarterly_results['Type']=="Operating Profit")|
                                                        (df_quarterly_results['Type']=="OPM %")|(df_quarterly_results['Type']=="Financing Profit")|
                                                        (df_quarterly_results['Type']=="Financing Margin %")|
                                                        (df_quarterly_results['Type']=="Other Income")|(df_quarterly_results['Type']=="Depreciation")|
                                                        (df_quarterly_results['Type']=="Tax %")|(df_quarterly_results['Type']=="Profit before tax")|
                                                        (df_quarterly_results['Type']=="Net Profit")].reset_index(drop=True)
            df_quarterly_results_cols = df_quarterly_results.columns.tolist()
            df_quarterly_results_cols.remove('Type')
            for col in df_quarterly_results_cols:
                df_quarterly_results[col] = pd.to_numeric(df_quarterly_results[col].str.replace(r'[,|%]', '', regex=True), errors='coerce')
            
        
        profit_loss_content = content.find('h2', string="Profit & Loss")
        if profit_loss_content:
            profit_loss_table = profit_loss_content.find_next('table', class_='data-table responsive-text-nowrap')
            if profit_loss_table:
                profit_loss_headers = [th.get_text(strip=True) for th in profit_loss_table.find('thead').find_all('th')]
            rows = []
            for row in profit_loss_table.find("tbody").find_all("tr"):
                cols = [col.text.strip() for col in row.find_all("td")]
                rows.append(cols)
            df_yearly_results = pd.DataFrame(rows, columns=profit_loss_headers)
            df_yearly_results = df_yearly_results.drop(index=len(df_yearly_results)-1)
            df_yearly_results = df_yearly_results.rename(columns={'':'Type'})
            df_yearly_results['Type'] = df_yearly_results['Type'].str.rstrip('\xa0+')
            df_yearly_results = df_yearly_results.drop(columns=[df_yearly_results.columns.tolist()[1]])
            if 'TTM' in df_yearly_results.columns:
                df_ttm_results = df_yearly_results[['Type','TTM']]
                df_yearly_results = df_yearly_results.drop(columns=['TTM'])
            else:
                df_ttm_results = df_yearly_results[['Type',df_yearly_results.columns.tolist()[-1]]]
                df_ttm_results = df_ttm_results.rename(columns={df_yearly_results.columns.tolist()[-1]:'TTM'})
            df_yearly_results = df_yearly_results[(df_yearly_results['Type']=="Sales")|(df_yearly_results['Type']=="Revenue")|
                                                  (df_yearly_results['Type']=="Expenses")|(df_yearly_results['Type']=="Operating Profit")|
                                                  (df_yearly_results['Type']=="OPM %")|(df_yearly_results['Type']=="Financing Profit")|
                                                  (df_yearly_results['Type']=="Financing Margin %")|(df_yearly_results['Type']=="Other Income")|
                                                  (df_yearly_results['Type']=="Depreciation")|(df_yearly_results['Type']=="Tax %")|
                                                  (df_yearly_results['Type']=="Profit before tax")|
                                                  (df_yearly_results['Type']=="Net Profit")].reset_index(drop=True)
            df_yearly_results_cols = df_yearly_results.columns.tolist()
            df_yearly_results_cols.remove('Type')
            for col in df_yearly_results_cols:
                df_yearly_results[col] = pd.to_numeric(df_yearly_results[col].str.replace(r'[,|%]', '', regex=True), errors='coerce')
        
    
            df_ttm_results = df_ttm_results[(df_ttm_results['Type']!="Operating Profit")&(df_ttm_results['Type']!="OPM %")&
                                            (df_ttm_results['Type']!="Tax %")&(df_ttm_results['Type']!="EPS in Rs")].reset_index(drop=True)
            df_ttm_results['TTM'] = pd.to_numeric(df_ttm_results['TTM'].str.replace(',', ''), errors='coerce').fillna(0).astype('int')
            index_to_update = df_ttm_results[df_ttm_results['Type']=="Profit before tax"].index[0]
            df_ttm_results.at[index_to_update, 'Type'] = "Tax"
            profit_before_tax = df_ttm_results[df_ttm_results['Type']=="Tax"]['TTM'].values[0]
            net_profit = df_ttm_results[df_ttm_results['Type']=="Net Profit"]['TTM'].values[0]
            df_ttm_results.at[index_to_update,'TTM'] = profit_before_tax - net_profit
            df_ttm_results.loc[df_ttm_results['Type'].isin(["Expenses","Interest","Depreciation","Tax","Net Profit"]),'TTM'] *= -1
            desired_order = ["Revenue","Sales","Other Income","Expenses","Interest","Depreciation","Tax","Net Profit"]
            desired_order = [item for item in desired_order if item in df_ttm_results['Type'].values.tolist()]
            df_ttm_results = df_ttm_results.set_index("Type").loc[desired_order].reset_index()
    
    
        with st.expander("Profit & Loss", expanded=True):
            resultoptioncol, tenureoptioncol, extracol1 = st.columns([1.1,0.4,1])
            with tenureoptioncol:
                prtenure = st.radio("Tenure", ["Quarterly","Yearly"], index=0, horizontal=True, key="profit_tenure")
            if prtenure=="Quarterly":
                dfrequired = df_quarterly_results.copy()
                dfrequiredcols = df_quarterly_results_cols
            else:
                dfrequired = df_yearly_results.copy()
                dfrequiredcols = df_yearly_results_cols
            dfrequiredcolors = color_palette[:len(dfrequiredcols)]
            with resultoptioncol:
                resultoptions = dfrequired['Type'].values.tolist()
                resultoption = st.radio("", resultoptions, index=0, horizontal=True)
            
            barchartcol, waterfallcol = st.columns([1.5,1])
            with barchartcol:
                filtered_data_df = dfrequired[dfrequired["Type"]==resultoption]
                values = filtered_data_df.iloc[0,1:]
                intvalues = [item for item in values if not isinstance(item, str)]
                ymin = min(list(intvalues))*1.35 if min(list(intvalues))<0 else 0
                ymax = max(list(intvalues))*1.35 if max(list(intvalues))>0 else 0
                if (resultoption=="OPM %") or (resultoption=="Financing Margin %") or (resultoption=="Tax %"):
                    yaxistitle = "Percentage (%)"
                else:
                    yaxistitle = "₹ in Cr."
                
                percent_variation = [0]
                for i in range(1, len(values)):
                    variation = (((values[i]-values[i-1])/values[i-1])*100).round(1)
                    percent_variation.append(variation)
                fig_pcbar = go.Figure(data=[go.Bar(x=dfrequiredcols, y=values, name=resultoption, text=values, textposition="outside",
                                                   textfont=dict(size=14), marker=dict(color="#74a5f2"))])
                fig_pcbar.add_trace(go.Scatter(x=dfrequiredcols, y=percent_variation, text=percent_variation, name="% Variation", mode="lines+markers",
                                               marker=dict(color="#d94a8c", size=6), line=dict(width=2, dash="dash"), yaxis="y2"))
                fig_pcbar.update_layout(title=dict(text="Profit Comparison", x=0.5, xanchor="center", font=dict(size=18)), yaxis_title=yaxistitle,
                                        xaxis=dict(categoryorder="array", tickfont=dict(size=9.5), tickangle=0, categoryarray=dfrequiredcols),
                                        yaxis=dict(range=[ymin,ymax], showgrid=True), yaxis2=dict(overlaying="y", showticklabels=False, showgrid=False),
                                        height=320, width=750, margin=dict(t=40,b=10,l=15,r=10))
                st.plotly_chart(fig_pcbar, use_container_width=True)
    
            with waterfallcol:
                text_positions = ['outside']*(len(df_ttm_results)-1) + ['inside']
                clr = 1 if net_profit<0 else 2
                df_ttm_results['Type'] = np.where(((df_ttm_results['Type']=="Net Profit")&(df_ttm_results['TTM']>0)),"Net Loss",df_ttm_results['Type'])
                net_expense = df_ttm_results[(df_ttm_results['Type']!='Revenue')&(df_ttm_results['Type']!='Sales')&
                                             (df_ttm_results['Type']!='Other Income')&(df_ttm_results['Type']!='Net Profit')]['TTM'].sum()
                profit_perc = np.round(net_profit/abs(net_expense),1)
                figwf = go.Figure(go.Waterfall(name="Financial Data", orientation="v", x=df_ttm_results['Type'], y=df_ttm_results['TTM'],
                                               connector={"line":{"color":color_palette[clr]}},
                                               text=df_ttm_results['TTM'].apply(lambda x: f"₹ {abs(int(x))}"), textposition=text_positions,
                                               textfont=dict(size=10), increasing={"marker":{"color":color_palette[clr]}},
                                               decreasing={"marker":{"color":color_palette[clr]}}))
                figwf.add_shape(type="line", x0=-0.5, x1=len(df_ttm_results)-0.5, y0=0, y1=0, line=dict(color="white", width=2))
                figwf.update_layout(title=dict(text=f"Profit-Loss (TTM)  ~   {profit_perc}%", x=0.55, xanchor='center', font=dict(size=18)),
                                    yaxis_title="₹ in Cr.", yaxis=dict(tickfont=dict(size=9)), xaxis=dict(tickfont=dict(size=9.5),tickangle=0),
                                    showlegend=False, template="plotly_white", width=500, height=320, margin=dict(t=40,b=10,l=15,r=10))
                st.plotly_chart(figwf, use_container_width=True)


# ### Balance Sheet

# In[ ]:


def balancesheet(content, industry):
    if industry != "Index":
        assetliab_content = content.find('h2', string="Balance Sheet")
        if assetliab_content:
            assetliab_table = assetliab_content.find_next('table', class_='data-table responsive-text-nowrap')
            if assetliab_table:
                assetliab_headers = [th.get_text(strip=True) for th in assetliab_table.find('thead').find_all('th')]
            rows = []
            for row in assetliab_table.find("tbody").find_all("tr"):
                cols = [col.text.strip() for col in row.find_all("td")]
                rows.append(cols)
            assetliab_df = pd.DataFrame(rows, columns=assetliab_headers)
            assetliab_df = assetliab_df.rename(columns={'':'Type'})
            assetliab_df['Type'] = assetliab_df['Type'].str.rstrip('\xa0+')
            assetliab_df_cols = assetliab_df.columns.tolist()
            assetliab_df_cols.remove('Type')
            for col in assetliab_df_cols:
                assetliab_df[col] = pd.to_numeric(assetliab_df[col].str.replace(',', ''), errors='coerce').astype('int')
            liab_df = assetliab_df[:5].copy().reset_index(drop=True)
            asset_df = assetliab_df[5:].copy().reset_index(drop=True)
    
        with st.expander("Balance Sheet", expanded=True):
            assetcolumn, liabilitycolumn = st.columns([1,1])
            with assetcolumn:
                asset_df_long = asset_df[:-1].melt(id_vars="Type", var_name="Year", value_name="Value")
                figasset = go.Figure()
                for t in asset_df["Type"]:
                    filtered_data = asset_df_long[asset_df_long["Type"]==t]
                    figasset.add_trace(go.Bar(x=filtered_data["Year"], y=filtered_data["Value"], name=t, text=filtered_data["Value"],
                                              textposition="inside", textfont=dict(size=10)))
                figasset.update_layout(title=dict(text=f"Assets", x=0.55, xanchor='center', font=dict(size=18)), barmode='stack', xaxis_title="Year",
                                       yaxis_title="₹ in Cr.", yaxis=dict(tickfont=dict(size=9)), xaxis=dict(tickfont=dict(size=9.5),tickangle=0),
                                       legend=dict(title="", orientation="h", y=-0.2, x=0.5, xanchor="center"), template="plotly_white",
                                       width=500, height=320, margin=dict(t=40,b=10,l=15,r=10))
                st.plotly_chart(figasset, use_container_width=True)
        
            with liabilitycolumn: 
                liab_df_long = liab_df[:-1].melt(id_vars="Type", var_name="Year", value_name="Value")
                figliab = go.Figure()
                for t in liab_df["Type"]:
                    filtered_data = liab_df_long[liab_df_long["Type"]==t]
                    figliab.add_trace(go.Bar(x=filtered_data["Year"], y=filtered_data["Value"], name=t, text=filtered_data["Value"],
                                             textposition="inside", textfont=dict(size=10)))
                figliab.update_layout(title=dict(text=f"Liabilities",x=0.55,xanchor='center',font=dict(size=18)), barmode='stack', xaxis_title="Year",
                                      yaxis_title="₹ in Cr.", yaxis=dict(tickfont=dict(size=9)), xaxis=dict(tickfont=dict(size=9.5),tickangle=0),
                                      legend=dict(title="", orientation="h", y=-0.2, x=0.5, xanchor="center"), template="plotly_white",
                                      width=500, height=320, margin=dict(t=40,b=10,l=15,r=10))
                st.plotly_chart(figliab, use_container_width=True)


# ### Shareholding Pattern

# In[ ]:


def shareholding(content, industry):
    if industry != "Index":
        quarterly_table = content.find("div", {"id": "quarterly-shp"}).find("table", {"class": "data-table"})
        headers_quarterly = [th.text.strip() for th in quarterly_table.find("thead").find_all("th")]
        rows_quarterly = []
        for tr in quarterly_table.find("tbody").find_all("tr"):
            row = [td.text.strip() for td in tr.find_all("td")]
            rows_quarterly.append(row)
        df_quarterly = pd.DataFrame(rows_quarterly, columns=headers_quarterly)
        df_quarterly = df_quarterly.drop(index=len(df_quarterly)-1)
        df_quarterly = df_quarterly.rename(columns={'':'Type'})
        df_quarterly['Type'] = df_quarterly['Type'].str.rstrip(' + ')
        df_quarterly_cols = df_quarterly.columns.tolist()
        df_quarterly_cols.remove('Type')
        for col in df_quarterly_cols:
            df_quarterly[col] = df_quarterly[col].str.rstrip('%').astype('float')
    
        
        yearly_table = content.find("div", {"id": "yearly-shp"}).find("table", {"class": "data-table"})
        headers_yearly = [th.text.strip() for th in yearly_table.find("thead").find_all("th")]
        rows_yearly = []
        for tr in yearly_table.find("tbody").find_all("tr"):
            row = [td.text.strip() for td in tr.find_all("td")]
            rows_yearly.append(row)
        df_yearly = pd.DataFrame(rows_yearly, columns=headers_yearly)
        df_yearly = df_yearly.drop(index=len(df_yearly)-1)
        df_yearly = df_yearly.rename(columns={'':'Type'})
        df_yearly['Type'] = df_yearly['Type'].str.rstrip(' + ')
        df_yearly_cols = df_yearly.columns.tolist()
        df_yearly_cols.remove('Type')
        for col in df_yearly_cols:
            df_yearly[col] = df_yearly[col].str.rstrip('%').astype('float')
    
            
        with st.expander("Share Holders", expanded=True):
            shareholdingcol1, shareholdingcol2 = st.columns([1,2])
            shareholders = df_quarterly['Type'].values.tolist()
            with shareholdingcol1:
                fig_shpie = go.Figure(data=[go.Pie(labels=shareholders, values=df_quarterly[df_quarterly.columns.tolist()[-1]], hole=0.5,
                                      textfont=dict(size=14), textinfo='percent', direction='clockwise', sort=False)])
                fig_shpie.update_layout(title=dict(text="Current Share Holders", x=0.5, xanchor="center", font=dict(size=18)),
                                        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"), showlegend=False, width=320, height=380,
                                        margin=dict(t=150,b=10,l=30,r=30))
                st.plotly_chart(fig_shpie, use_container_width=True)
            with shareholdingcol2:
                shareholderscol, shareholdingtenurecol = st.columns([0.7,0.3])
                with shareholdingtenurecol:
                    shtenure = st.radio("Tenure", ["Quarterly","Yearly"], index=0, horizontal=True, key="shareholding_tenure")
                if shtenure=="Quarterly":
                    dfreq = df_quarterly.copy()
                    dfreqcols = df_quarterly_cols
                else:
                    dfreq = df_yearly.copy()
                    dfreqcols = df_yearly_cols
                dfreqcolors = color_palette[:len(dfreqcols)]
                data = []
                with shareholderscol:
                    shareholder = st.radio("Share Holder", shareholders, index=0, horizontal=True)
                filtered_data = dfreq[dfreq["Type"]==shareholder]
                values = filtered_data.iloc[0,1:]
                fig_shbar = go.Figure(data=[go.Bar(x=dfreqcols, y=values, name=shareholder, text=values, textposition="outside", textfont=dict(size=14),
                                                   marker=dict(color=color_palette[shareholders.index(shareholder)]))])
                fig_shbar.update_layout(title=dict(text="Share Holding Pattern", x=0.5, xanchor="center", font=dict(size=22)),xaxis_title=shtenure[:-2],
                                        yaxis_title="Percentage(%)",xaxis=dict(categoryorder="array",categoryarray=dfreqcols),yaxis=dict(range=[0,100]),
                                        height=340, width=830, margin=dict(t=40,b=10,l=15,r=10))
                st.plotly_chart(fig_shbar, use_container_width=True)


# ### Dividends

# In[ ]:


def dividends(industry, stock_symbol, stock_name):
    if industry!="Index":
        with st.expander("Dividends", expanded=True):
            dividcol1, dividcol2 = st.columns([8,1])
            tickerdata = yf.Ticker(stock_symbol)
            dividdf = pd.DataFrame(tickerdata.dividends).reset_index()
            dividdf['Year'] = dividdf['Date'].dt.year
            dividdf['Date'] = pd.to_datetime(dividdf['Date']).dt.date
            yearwisediv = dividdf.groupby(['Year'])['Dividends'].sum()
            ymax = yearwisediv.max()*1.1
            with dividcol1:
                figdivd = go.Figure()
                figdivd.add_trace(go.Bar(x=dividdf['Year'], y=dividdf['Dividends'], name='Dividends', textposition="outside",
                                         textfont=dict(size=12), marker=dict(color=color_palette[0])))
                figdivd.update_layout(title=dict(text="Dividends", x=0.5, xanchor="center", font=dict(size=18)), yaxis_title="Rupees (₹)", height=310,
                                      width=750, margin=dict(t=40,b=10,l=15,r=10), yaxis=dict(range=[0,ymax]),
                                      xaxis=dict(title="Year", tickmode='linear', dtick=1, tickfont=dict(size=10)))
                st.plotly_chart(figdivd, use_container_width=True)
            with dividcol2:
                st.metric("No. Dividend Payouts:", len(dividdf))
                st.metric("Median Dividend (₹):", "₹"+str(np.round(dividdf[dividdf['Dividends']>0]['Dividends'].median(),2)))
                st.metric("Median Last 5 Payouts:", "₹"+str(np.round(dividdf[dividdf['Dividends']>0]['Dividends'][-5:].median(),2)))


# ### Delivery Quantity

# In[ ]:


def deliveryqty(industry, stock_symbol, stock_name):
    if industry!="Index":
        with st.expander("Delivery Quantity", expanded=True):
            placeholder = st.empty()
            loading_gif_url = "https://i.gifer.com/74pZ.gif"
            with placeholder.container():
                imagecol1, imagecol2, imagecol3 = st.columns([3,1,3])
                with imagecol2:
                    st.image(loading_gif_url, caption="Loading... Please wait.", use_column_width=True)
            delivdf = pd.DataFrame()
            for i in range(15,0,-1):
                try:
                    date = (datetime.today()-timedelta(days=i)).strftime("%d-%m-%Y")
                    deldf = capital_market.bhav_copy_with_delivery(date)
                    deldf['Date'] = pd.to_datetime(date,format="%d-%m-%Y")
                    deldf = deldf[deldf['SYMBOL']==stock_symbol.split(".")[0]][['Date','TTL_TRD_QNTY','DELIV_QTY']].reset_index(drop=True)
                    delivdf = pd.concat([delivdf, deldf], ignore_index=True)
                except:
                    pass
            placeholder.empty()
            figdeliv = go.Figure()
            figdeliv.add_trace(go.Bar(x=delivdf['Date'], y=delivdf['TTL_TRD_QNTY'], marker=dict(color="lightblue"), text=delivdf['TTL_TRD_QNTY'],
                                      name="Total Traded Quantity"))
            figdeliv.add_trace(go.Bar(x=delivdf['Date'], y=delivdf['DELIV_QTY'], marker=dict(color="blue"), text=delivdf['DELIV_QTY'],
                                      name="Delivery Quantity"))
            figdeliv.update_layout(title=dict(text=stock_name+" - Traded & Delivered Quantity (Last 15 days)", x=0.5, xanchor='center'), width=200,
                                   barmode='overlay', yaxis_title="Quantity count", xaxis_title=stock_name, template="plotly_white", height=550,
                                   xaxis=dict(tickmode='linear', dtick=1, tickfont=dict(size=10)))
            st.plotly_chart(figdeliv, use_container_width=True)


# ## Stock Analysis function call

# In[ ]:


def stock_analysis():
    companies = {'Index':{"Nifty 50":"^NSEI", "Sensex":"^BSESN", "Nifty Auto":"^CNXAUTO", "Nifty Bank":"^NSEBANK", "Nifty Commodities":"^CNXCMDT",
                          "Nifty Energy":"^CNXENERGY", "Nifty FMCG":"^CNXFMCG", "Nifty IT":"^CNXIT", "Nifty Infrastructure":"^CNXINFRA",
                          "Nifty Media":"^CNXMEDIA", "Nifty Metal":"^CNXMETAL", "Nifty Realty":"^CNXREALTY"},
                 'Automotive':{"Tata Motors":"TATAMOTORS.NS", "Mahindra & Mahindra":"M&M.NS","Hyundai":"HYUNDAI.NS","Hero Motocorp":"HEROMOTOCO.NS",
                               "Maruti Suzuki":"MARUTI.NS", "TVS Motors":"TVSMOTOR.NS", "Bajaj Auto":"BAJAJ-AUTO.NS", "Ola Electric":"OLAELEC.NS",
                               "Eicher Motors":"EICHERMOT.NS", "Ashok Leyland":"ASHOKLEY.NS", "Force Motors":"FORCEMOT.NS"}, 
                 'Banking':{"HDFC":"HDFCBANK.NS", "ICICI":"ICICIBANK.NS", "Kotak":"KOTAKBANK.NS", "Axis":"AXISBANK.NS", "IOB":"IOB.NS","SBI":"SBIN.NS"},
                 'Energy':{"Tata Power":"TATAPOWER.NS", "JSW Energy":"JSWENERGY.NS", "Adani Energy Solutions":"ADANIENSOL.NS",
                           "Adani Power":"ADANIPOWER.NS", "Adani Green Energy":"ADANIGREEN.NS"},
                 'Electr Equip':{"Exicom Tele-Systems":"EXICOM.NS", "ABB":"ABB.NS", "Tata Elxsi":"TATAELXSI.NS"},
                 'FMCG':{"ITC":"ITC.NS", "Nestle":"NESTLEIND.NS","Dabur":"DABUR.NS","Hindustan Unilever":"HINDUNILVR.NS","Britannia":"BRITANNIA.NS",
                         "Godrej":"GODREJCP.NS", "Hatsun":"HATSUN.NS", "Tata Consumer Products":"TATACONSUM.NS", "Varun Beverages":"VBL.NS",
                         "Bikaji Foods":"BIKAJI.NS"},
                 'IT':{"TCS":"TCS.NS", "Wipro":"WIPRO.NS", "Tech Mahindra":"TECHM.NS", "KPIT":"KPITTECH.NS"},
                 'Pharma':{"Cipla":"CIPLA.NS", "Sun Pharma":"SPARC.NS", "Mankind Pharma":"MANKIND.NS", "Natco Pharma":"NATCOPHARM.NS",
                           "Laurus Labs":"LAURUSLABS.NS"},
                 'HealthCare':{"Apollo Hospitals":"APOLLOHOSP.NS", "Narayana Hrudayalaya":"NH.NS"},
                 'E-Commerce':{"Zomato":"ZOMATO.NS", "Swiggy":"SWIGGY.NS"},
                 'Defence':{"Hindustan Aeronautics":"HAL.NS", "Bharat Electronics":"BEL.NS", "Bharat Dynamics":"BDL.NS",
                            "Mazagon Dock ShipBuilders":"MAZDOCK.NS", "Garden Reach ShipBuilders":"GRSE.NS"},
                 'Waste Recycle':{"Ganesha Ecosphere":"GANECOS.NS", "Antony Waste Handling Cell":"AWHCL.NS", "Eco Recycling":"ECORECO.BO"}}
    web = 'https://www.screener.in/company/'
    con = 'consolidated/'
    stockurls = {'Nifty 50':f'{web}NIFTY/', 'Sensex':f'{web}1001/', 'Nifty Auto':f'{web}CNXAUTO/', 'Nifty Bank':f'{web}BANKNIFTY/',
                 'Nifty Commodities':f'{web}CNXCOMMODI/', 'Nifty Energy':f'{web}CNXENERY/', 'Nifty FMCG':f'{web}CNXFMCG/',
                 'Nifty IT':f'{web}CNXIT/', 'Nifty Infrastructure':f'{web}CNXINFRAST/', 'Nifty Media':f'{web}CNXMEDIA/',
                 'Nifty Metal':f'{web}CNXMETAL/', 'Nifty Pharma':f'{web}CNXPHARMA/', 'Nifty Realty':f'{web}CNXREALTY/',
                 'Tata Motors':f'{web}TATAMOTORS/{con}', 'Mahindra & Mahindra':f'{web}M&M/{con}', 'Hyundai':f'{web}HYUNDAI/',
                 'Hero Motocorp':f'{web}HEROMOTOCO/{con}', 'Maruti Suzuki':f'{web}MARUTI/{con}', 'TVS Motors':f'{web}TVSMOTOR/{con}',
                 'Bajaj Auto':f'{web}BAJAJ-AUTO/{con}', 'Ola Electric':f'{web}OLAELEC/{con}', 'Eicher Motors':f'{web}EICHERMOT/{con}',
                 'Ashok Leyland':f'{web}ASHOKLEY/{con}', 'Force Motors':f'{web}FORCEMOT/{con}', 'HDFC':f'{web}HDFCBANK/{con}',
                 'ICICI':f'{web}ICICIBANK/{con}', 'Kotak':f'{web}/KOTAKBANK/{con}', 'Axis':f'{web}/AXISBANK/{con}', 'IOB':f'{web}IOB/',
                 'SBI':f'{web}SBIN/{con}', 'Tata Power':f'{web}TATAPOWER/{con}', 'JSW Energy':f'{web}JSWENERGY/{con}',
                 'Adani Energy Solutions':f'{web}ADANIENSOL/{con}', 'Adani Power':f'{web}ADANIPOWER/{con}',
                 'Adani Green Energy':f'{web}/ADANIGREEN/{con}', 'Exicom Tele-Systems':f'{web}EXICOM/{con}', 'ABB':f'{web}ABB/',
                 'Tata Elxsi':f'{web}/TATAELXSI/', 'ITC':f'{web}ITC/{con}', 'Nestle':f'{web}NESTLEIND/','Dabur':f'{web}DABUR/{con}',
                 'Hindustan Unilever':f'{web}HINDUNILVR/{con}', 'Britannia':f'{web}BRITANNIA/{con}', 'Godrej':f'{web}GODREJCP/{con}',
                 'Hatsun':f'{web}HATSUN/', 'Tata Consumer Products':f'{web}TATACONSUM/{con}', 'Varun Beverages':f'{web}/VBL/{con}',
                 'Bikaji Foods':f'{web}/BIKAJI/', 'TCS':f'{web}/TCS/{con}', 'Wipro':f'{web}/WIPRO/{con}', 'Tech Mahindra':f'{web}/TECHM/{con}',
                 'KPIT':f'{web}KPITTECH/{con}', 'Cipla':f'{web}/CIPLA/{con}', 'Sun Pharma':f'{web}/SUNPHARMA/{con}',
                 'Mankind Pharma':f'{web}/MANKIND/', 'Natco Pharma':f'{web}/NATCOPHARM/{con}', 'Laurus Labs':f'{web}/LAURUSLABS/{con}',
                 'Apollo Hospitals':f'{web}/APOLLOHOSP/{con}', 'Narayana Hrudayalaya':f'{web}/NH/{con}',
                 'Zomato':f'{web}/ZOMATO/{con}', 'Swiggy':f'{web}/SWIGGY/{con}', 'Hindustan Aeronautics':f'{web}/HAL/',
                 'Bharat Electronics':f'{web}/BEL/{con}', 'Bharat Dynamics':f'{web}/BDL/', 'Mazagon Dock ShipBuilders':f'{web}/MAZDOCK/{con}',
                 'Garden Reach ShipBuilders':f'{web}/GRSE/{con}', 'Ganesha Ecosphere':f'{web}/GANECOS/{con}',
                 'Antony Waste Handling Cell':f'{web}/AWHCL/{con}', 'Eco Recycling':f'{web}/530643/{con}'}

    stock_name, stock_symbol, industry, alltimehigh, alltimelow, high52week, low52week, high3month, low3month = prologue(companies)
    content = getcontents(stockurls, stock_name)
    basicinfo(content, industry, alltimehigh, alltimelow, high52week, low52week, high3month, low3month)
    proscons(content, industry)
    profitloss(content, industry)
    balancesheet(content, industry)
    shareholding(content, industry)
    dividends(industry, stock_symbol, stock_name)
    # deliveryqty(industry, stock_symbol, stock_name)


# # Index Performance page

# ### Sector data extraction

# In[ ]:


def sector_extract(sector_url):
    response = requests.get(sector_url)
    soup = bs(response.content, "html.parser")
    pagination = soup.find("div", class_="pagination")
    if not pagination:
        total_pages = 1
    else:
        page_links = pagination.find_all("a", class_="ink-900")
        total_pages = int(page_links[-1].text.strip()) if page_links else 1
    
    stock_names = []
    latest_quarter_sales = []
    
    for page in range(1, total_pages+1):
        url = f"{sector_url}?page={page}"
        response = requests.get(url)
        if response.status_code != 200:
            continue
        soup = bs(response.content, "html.parser")
        table = soup.find("table", class_="data-table text-nowrap striped mark-visited")
        table = soup.select_one("table.data-table")
        if not table:
            continue
        rows = table.find_all("tr")[1:]
        for row in rows:
            cells = row.find_all("td")
            if len(cells) >= 9:
                stock_name = cells[1].text.strip()
                sales_qtr = cells[8].text.strip()
                stock_names.append(stock_name)
                latest_quarter_sales.append(sales_qtr)
    
    df = pd.DataFrame({"Stock Name": stock_names, "Sales": latest_quarter_sales})
    df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
    df = df.dropna(subset=['Sales'], axis=0)
    df['Sales'] = df['Sales'].astype('float')
    df['Percentage'] = 100*df['Sales']/df['Sales'].sum()
    df['Percentage'] = df['Percentage'].round(1)
    df = df.sort_values(by=['Percentage'], ascending=False).reset_index(drop=True)
    df['Cumulative_Percentage'] = df['Percentage'].cumsum()
    return df    


# ### Industry data extraction

# In[ ]:


def industry_extract(sector_url, url_range):
    inddf = []
    indname = []
    for i in range(url_range[0], url_range[1]+1):
        industry_url = f"{sector_url}{i:08d}/"
        response = requests.get(industry_url)
        soup = bs(response.content, "html.parser")
        pagination = soup.find("div", class_="pagination")
        if not pagination:
            total_pages = 1
        else:
            page_links = pagination.find_all("a", class_="ink-900")
            total_pages = int(page_links[-1].text.strip()) if page_links else 1
            
        stock_names = []
        latest_quarter_sales = []
    
        for page in range(1, total_pages+1):
            url = f"{industry_url}?page={page}"
            response = requests.get(url)
            if response.status_code != 200:
                continue
            soup = bs(response.content, "html.parser")
            table = soup.find("table", class_="data-table text-nowrap striped mark-visited")
            table = soup.select_one("table.data-table")
            if not table:
                continue
            rows = table.find_all("tr")[1:]
            for row in rows:
                cells = row.find_all("td")
                if len(cells) >= 9:
                    stock_name = cells[1].text.strip()
                    sales_qtr = cells[8].text.strip()
                    stock_names.append(stock_name)
                    latest_quarter_sales.append(sales_qtr)
        
        df = pd.DataFrame({"Stock Name": stock_names, "Sales": latest_quarter_sales})
        df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
        df = df.dropna(subset=['Sales'], axis=0)
        df['Sales'] = df['Sales'].astype('float')
        df['Percentage'] = 100*df['Sales']/df['Sales'].sum()
        df['Percentage'] = df['Percentage'].round(1)
        df = df.sort_values(by=['Percentage'], ascending=False).reset_index(drop=True)
        df['Cumulative_Percentage'] = df['Percentage'].cumsum()
        
        inddf.append(df)
        indname.append(soup.title.string)
    return inddf, indname


# ### Market Share Graphs

# In[ ]:


def market_share(sector_urls, industry_urls):
    st.markdown("<br>", unsafe_allow_html=True)
    sectorcol1, sectorcol2, sectorcol3 = st.columns([3,1,3])
    with sectorcol1:
        sectorname = st.selectbox("Select the Sector:", list(sector_urls.keys()), index=0)
    sector_url = sector_urls[sectorname]

    sector_market_share_df = sector_extract(sector_url)
    
    desired_binsize = 100
    whileloopcondition = True
    while whileloopcondition:
        if len(sector_market_share_df[sector_market_share_df['Cumulative_Percentage']<desired_binsize])<10:
            whileloopcondition = False
        else:
            desired_binsize-=5
    sector_market_share_df['Stock Name'] = np.where(sector_market_share_df['Cumulative_Percentage']>desired_binsize, "Others",
                                                    sector_market_share_df['Stock Name'])
    
    sectorcol4, sectorcol5, sectorcol6 = st.columns([3,1,3])
    with sectorcol4:
        fig_market_share = go.Figure(data=[go.Pie(labels=sector_market_share_df['Stock Name'], values=sector_market_share_df['Percentage'], hole=0.5,
                                                  textfont=dict(size=14), textinfo='percent', direction='clockwise', sort=False)])
        fig_market_share.update_layout(title=dict(text="Current Market Share", x=0.5, xanchor="center", font=dict(size=18)), height=450,showlegend=True,
                                       legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"), margin=dict(t=40,b=10,l=30,r=30))
        st.plotly_chart(fig_market_share, use_container_width=True)

    with sectorcol2:
        st.write("Sector Break down")
        sector_breakdown = st.toggle("On", value=False)

    if sector_breakdown:
        industry_url_range = industry_urls[sectorname]
        industry_market_share_df_list, industry_names_list = industry_extract(sector_url, industry_url_range)
        industry_names_list = [i.split(' - Screener')[0] for i in industry_names_list if ' - Screener' in i]
        industry_names_list = [i.split(' Companies')[0] for i in industry_names_list if ' Companies' in i]
        industry_names_list = [i.split(' - ')[1] for i in industry_names_list if ('Automobiles - ' in i)|('Banks - ' in i)]
        
        with sectorcol3:
            industryname = st.selectbox("Select the Industry:", industry_names_list, index=0)
            indindex = industry_names_list.index(industryname)
        
        desired_binsize = 100
        whileloopcondition = True
        while whileloopcondition:
            if len(industry_market_share_df_list[indindex][industry_market_share_df_list[indindex]['Cumulative_Percentage']<desired_binsize])<10:
                whileloopcondition = False
            else:
                desired_binsize-=5
        industry_market_share_df_list[indindex]['Stock Name'] = np.where(
                                                        industry_market_share_df_list[indindex]['Cumulative_Percentage']>desired_binsize, "Others",
                                                        industry_market_share_df_list[indindex]['Stock Name'])
        
        with sectorcol6:
            fig_market_share2 = go.Figure(data=[go.Pie(labels=industry_market_share_df_list[indindex]['Stock Name'],
                                                       values=industry_market_share_df_list[indindex]['Percentage'], hole=0.5,
                                                       textfont=dict(size=14), textinfo='percent', direction='clockwise', sort=False)])
            fig_market_share2.update_layout(title=dict(text="Current Market Share", x=0.5, xanchor="center", font=dict(size=18)), showlegend=True,
                                           legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"), height=450, margin=dict(t=40,b=10,l=30,r=30))
            st.plotly_chart(fig_market_share2, use_container_width=True)


# ### Index Performance Graph

# In[ ]:


def index_performance():
    indices = {"Nifty 50":"^NSEI", "Sensex":"^BSESN", "Nifty Auto":"^CNXAUTO", "Nifty Bank":"^NSEBANK", "Nifty Commodities":"^CNXCMDT",
               "Nifty Energy":"^CNXENERGY", "Nifty FMCG":"^CNXFMCG", "Nifty IT":"^CNXIT", "Nifty Infrastructure":"^CNXINFRA",
               "Nifty Media":"^CNXMEDIA", "Nifty Metal":"^CNXMETAL", "Nifty Realty":"^CNXREALTY"}
    web = 'https://www.screener.in/company/'
    indexurls = {'Nifty 50':f'{web}NIFTY/', 'Sensex':f'{web}1001/', 'Nifty Auto':f'{web}CNXAUTO/', 'Nifty Bank':f'{web}BANKNIFTY/',
                 'Nifty Commodities':f'{web}CNXCOMMODI/', 'Nifty Energy':f'{web}CNXENERY/', 'Nifty FMCG':f'{web}CNXFMCG/',
                 'Nifty IT':f'{web}CNXIT/', 'Nifty Infrastructure':f'{web}CNXINFRAST/', 'Nifty Media':f'{web}CNXMEDIA/',
                 'Nifty Metal':f'{web}CNXMETAL/', 'Nifty Pharma':f'{web}CNXPHARMA/', 'Nifty Realty':f'{web}CNXREALTY/'}
    index_names = list(indices.keys())
    date_ranges = ["1D", "3D", "1W", "2W", "1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "Max"]
    
    incol1, incol2, incol3 = st.columns([1,3,1])
    with incol2:
        date_range = st.radio("Select Date Range:", date_ranges, index=7, horizontal=True)
    
    default_start_date = datetime.today() - timedelta(days=1)
    
    if date_range == "1D":
        start_date = default_start_date
        term = "1 Day"
    elif date_range == "3D":
        start_date = default_start_date - timedelta(days=3)
        term = "3 Days"
    elif date_range == "1W":
        start_date = default_start_date - timedelta(weeks=1)
        term = "1 Week"
    elif date_range == "2W":
        start_date = default_start_date - timedelta(weeks=2)
        term = "2 Weeks"
    elif date_range == "1M":
        start_date = default_start_date - timedelta(days=30)
        term = "1 Month"
    elif date_range == "3M":
        start_date = default_start_date - timedelta(days=30 * 3)
        term = "3 Months"
    elif date_range == "6M":
        start_date = default_start_date - timedelta(days=30 * 6)
        term = "6 Months"
    elif date_range == "1Y":
        start_date = default_start_date - timedelta(days=365)
        term = "1 Year"
    elif date_range == "2Y":
        start_date = default_start_date - timedelta(days=2 * 365)
        term = "2 Years"
    elif date_range == "3Y":
        start_date = default_start_date - timedelta(days=3 * 365)
        term = "3 Years"
    elif date_range == "5Y":
        start_date = default_start_date - timedelta(days=5 * 365)
        term = "5 Years"
    else:
        term = "Overall"

    indexdf_list = []
    for indexname in index_names:
        try:
            indexdf = yf.download(indices[indexname])
            if isinstance(indexdf.columns, pd.MultiIndex):
                indexdf.columns = ['_'.join(col).strip() for col in indexdf.columns.values]
            indexdf['Date'] = indexdf.index
            if date_range!="Max":
                indexdf = indexdf[indexdf['Date']>=start_date]
            indexdf = indexdf.reset_index(drop=True)
        except:
            indexdf = pd.DataFrame()
        indexdf_list.append(indexdf)
    index_prfm = []
    colors = []
    for i in range(len(index_names)):
        df = indexdf_list[i]
        indexname = indices[index_names[i]]
        if len(df)>0:
            index_returns = np.round(100*((df['Close_'+indexname].iloc[-1])-(df['Close_'+indexname].iloc[0]))/(df['Close_'+indexname].iloc[0]),1)
        else:
            index_returns = 0
        color = 'green' if index_returns>0 else 'red'
        index_prfm.append(index_returns)
        colors.append(color)
    
    st.markdown("<br>", unsafe_allow_html=True)
    fig_indret = go.Figure(go.Bar(x=index_names, y=index_prfm, orientation='v', text=index_prfm, textposition="outside", marker=dict(color=colors)))
    fig_indret.update_layout(title=dict(text=f"Index Performance - {term}", x=0.55,xanchor="center",font=dict(size=25)), yaxis_title="Performance (%)",
                            xaxis_title="Index", height=500, width=650, margin=dict(t=40,b=10,l=10,r=10))
    st.plotly_chart(fig_indret, use_container_width=True)

    web2 = web+"compare/"
    secotr_urls = {'Automobile':f'{web2}00000005/', 'Banking':f'{web2}00000006/', 'Energy':f'{web2}00000049/', 'FMCG':f'{web2}00000027/',
                   'IT':f'{web2}00000034/', 'Infrastructure':f'{web2}00000032/', 'Media':f'{web2}00000024/', 'Mining & Mineral':f'{web2}00000038/',
                   'Metal':f'{web2}00000057/', 'Pharmaceutical':f'{web2}00000046/', 'Realty':f'{web2}00000051/'}
    industry_urls = {'Automobile':(5,9), 'Banking':(11,12), 'Energy':(76,76), 'FMCG':(53,54), 'IT':(26,30), 'Infrastructure':(45,45), 'Media':(47,47),
                     'Mining & Mineral':(59,59), 'Metal':(84,86), 'Pharmaceutical':(70,73), 'Realty':(31,31)}
    market_share(secotr_urls, industry_urls)


# # Strategy Indices

# In[ ]:


def nifty_strategy_indices():
    indices = {"Sectoral Index": {"Nifty 50":"^NSEI", "Sensex":"^BSESN", "Nifty Auto":"^CNXAUTO", "Nifty Bank":"^NSEBANK",
                                  "Nifty Commodities":"^CNXCMDT", "Nifty Energy":"^CNXENERGY", "Nifty FMCG":"^CNXFMCG", "Nifty IT":"^CNXIT",
                                  "Nifty Infrastructure":"^CNXINFRA", "Nifty Media":"^CNXMEDIA", "Nifty Metal":"^CNXMETAL","Nifty Realty":"^CNXREALTY"},
               "Strategy Index": {"Nifty Growth Sectors 15":"NI15.NS", "Nifty Alpha 50":"NFTALPHA50.NS", "Nifty 100 Quality 30":"NIFTYQLY30.NS",
                                  "Nifty 100 Low Volatality 30":"NFT100LV30.NS", "Nifty 200 Momentum 30":"NI200MOM30.NS",
                                  "Nifty Midcap 150 Quality 50":"NFMIDQLT50.NS", "Nifty 50 Value 20":"NV5020.NS",
                                  "Nifty Alpha Low Volatality 30":"ALPHALOWVO.NS"}}
    web = 'https://www.screener.in/company/'
    indexurls = {'Nifty 50':f'{web}NIFTY/', 'Sensex':f'{web}1001/', 'Nifty Auto':f'{web}CNXAUTO/', 'Nifty Bank':f'{web}BANKNIFTY/',
                 'Nifty Commodities':f'{web}CNXCOMMODI/', 'Nifty Energy':f'{web}CNXENERY/', 'Nifty FMCG':f'{web}CNXFMCG/',
                 'Nifty IT':f'{web}CNXIT/', 'Nifty Infrastructure':f'{web}CNXINFRAST/', 'Nifty Media':f'{web}CNXMEDIA/',
                 'Nifty Metal':f'{web}CNXMETAL/', 'Nifty Pharma':f'{web}CNXPHARMA/', 'Nifty Realty':f'{web}CNXREALTY/',
                 'Nifty Growth Sectors 15':f'{web}NI15/', 'Nifty Alpha 50':f'{web}NFTALPHA50/', 'Nifty 100 Quality 30':f'{web}/NIFTYQLTY30',
                 'Nifty 100 Low Volatality 30':f'{web}/NFT100LV30/', 'Nifty 200 Momentum 30':f'{web}/NI200MOM30/',
                 'Nifty Midcap 150 Quality 50':f'{web}/NFMIDQLT50/', 'Nifty 50 Value 20':f'{web}/NV5020/',
                 'Nifty Alpha Low Volatality 30':f'{web}/ALPHALOWVO/'}
    index_groups = list(indices.keys())
    indcol1, indcol2, indcol3, indcol4 = st.columns([1,0.75,1.25,1])
    with indcol2:
        index_group = st.selectbox("Index group", index_groups, index=0)
    index_dict = indices[index_group]
    with indcol3:
        index = st.selectbox("Index", list(index_dict.keys()), index=0)
    
    index_url = indexurls[index]
    response = requests.get(index_url)
    content = bs(response.content, 'html.parser')
    pagination = content.find("div", class_="pagination")
    if not pagination:
        total_pages = 1
    else:
        page_links = pagination.find_all("a", class_="ink-900")
        total_pages = int(page_links[-1].text.strip()) if page_links else 1

    stock_name = []
    cmp = []
    pe = []
    divyld = []
    qtrsalesgr = []
    qtrprofitgr = []
    roce = []
    
    for page in range(1, total_pages+1):
        url = f"{index_url}?page={page}"
        response = requests.get(url)
        if response.status_code != 200:
            continue
        soup = bs(response.content, "html.parser")
        table = soup.find("table", class_="data-table text-nowrap striped mark-visited")
        table = soup.select_one("table.data-table")
        if not table:
            continue
        rows = table.find_all("tr")[1:]
        for row in rows:
            cells = row.find_all("td")
            if len(cells) >= 9:
                stock_name.append(cells[1].text.strip())
                cmp.append(float(cells[2].text.strip()))
                pe.append(float(cells[3].text.strip()))
                divyld.append(float(cells[5].text.strip()))
                qtrsalesgr.append(float(cells[9].text.strip()))
                qtrprofitgr.append(float(cells[7].text.strip()))
                roce.append(float(cells[10].text.strip()))
    
    indexdf = pd.DataFrame({"Stock Name": stock_name, "Stock Price": cmp, "Stock PE": pe, "ROCE": roce, "Dividend Yield": divyld,
                       "Quarterly Sales Growth": qtrsalesgr, "Quarterly Profit Growth": qtrprofitgr})
    indexdfstocknames = indexdf['Stock Name'].values.tolist()
    indexdfstocknames = ["Median" if re.match(r"Median: \d+ Co\.", i) else i for i in indexdfstocknames]
    if "Median" in indexdfstocknames:
        indexdfmedianindex = indexdfstocknames.index("Median")
        indexdf = indexdf.drop(index=indexdfmedianindex).reset_index(drop=True)
    
    indexcol5, indexcol6, indexcol7, indexcol8 = st.columns([2,1,1,2])

    with indexcol6:
        selected_col = st.selectbox("Sort by", indexdf.columns.tolist()[2:], index=0)
    with indexcol7:
        sort_order = st.selectbox("Sort Order", ["High to Low", "Low to High"], index=0)
    
    if sort_order=="High to Low":
        indexdfsorted = indexdf.sort_values(by=[selected_col], ascending=False).reset_index(drop=True)
    else:
        indexdfsorted = indexdf.sort_values(by=[selected_col], ascending=True).reset_index(drop=True)
    
    indexdfsorted.index = range(1, len(indexdfsorted) + 1)
    st.dataframe(indexdfsorted.head(), use_container_width=True)

    st.write("Total List of Companies in this Index:")
    st.write(indexdf['Stock Name'].values.tolist())


# # FII DII Activity

# In[ ]:


def fii_dii_activity():
    st.markdown("<br>", unsafe_allow_html=True)
    url = 'https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/index.php'
    response = requests.get(url)
    if response.status_code == 200:
        page_content = response.content
        soup = bs(page_content, 'html.parser')
        table = soup.find("table", class_="mctable1 tble1")
        thismonth = [th.get_text(strip=True) for th in table.find_all("th") if th.get_text(strip=True)]
        thismonth = thismonth[-6:]
        data = []
        for row in table.find_all("tr")[2:]:
            cols = row.find_all("td")
            cols = [col.get_text(strip=True) for col in cols]
            if cols:
                data.append(cols)
        fii_data = [[row[0][:11], row[1], row[2], row[3]] for row in data]
        dii_data = [[row[0][:11], row[4], row[5], row[6]] for row in data]
        
        headers = ['Date', 'Gross Purchase', 'Gross Sales', 'Net Cash Flow']
        fii_df = pd.DataFrame(fii_data, columns=headers)
        fii_df['Date'] = pd.to_datetime(fii_df['Date'], format="%d-%b-%Y").dt.date
        fii_df[['Gross Purchase', 'Gross Sales', 'Net Cash Flow']] = fii_df[['Gross Purchase', 'Gross Sales', 'Net Cash Flow']
                                                                            ].replace(',', '', regex=True).astype(float)
        fii_df = fii_df.sort_index(ascending=False).reset_index(drop=True)
        
        dii_df = pd.DataFrame(dii_data, columns=headers)
        dii_df['Date'] = pd.to_datetime(dii_df['Date'], format="%d-%b-%Y").dt.date
        dii_df[['Gross Purchase', 'Gross Sales', 'Net Cash Flow']] = dii_df[['Gross Purchase', 'Gross Sales', 'Net Cash Flow']
                                                                            ].replace(',', '', regex=True).astype(float)
        dii_df = dii_df.sort_index(ascending=False).reset_index(drop=True)

        fiicol, diicol = st.columns([1,1])
        curmon = datetime.today().strftime('%b %Y')
        with fiicol:
            st.markdown("""<style>.title {{text-align: center; font-size: 28px; font-weight: bold;}}</style><div class="title">FII Activity - in {}</div>""".format(curmon), unsafe_allow_html=True)
            desiredcol1 = st.radio("", headers[1:], horizontal=True, index=2, key='fii_col')
            if desiredcol1=="Net Cash Flow":
                fii_colors = ["red" if val < 0 else "green" for val in fii_df[desiredcol1]]
            else:
                fii_colors = "#74a5f2"
            values = fii_df[desiredcol1].values.tolist()
            ymin = min(list(values))*1.35 if min(list(values))<0 else 0
            ymax = max(list(values))*1.35 if max(list(values))>0 else 0
            fig_fii = go.Figure(data=[go.Bar(x=fii_df['Date'], y=fii_df[desiredcol1], text=fii_df[desiredcol1], textposition="outside",
                                             textfont=dict(size=14), marker=dict(color=fii_colors))])
            fig_fii.update_layout(yaxis_title="₹ in Cr.", yaxis=dict(range=[ymin,ymax]), xaxis=dict(tickfont=dict(size=9.5), tickangle=0),
                                  height=320, width=750, margin=dict(t=40,b=10,l=10,r=25))
            st.plotly_chart(fig_fii, use_container_width=True)

        with diicol:
            st.markdown("""<style>.title {{text-align: center; font-size: 28px; font-weight: bold;}}</style><div class="title">DII Activity - in {}</div>""".format(curmon), unsafe_allow_html=True)
            desiredcol2 = st.radio("", headers[1:], horizontal=True, index=2, key='dii_col')
            if desiredcol2=="Net Cash Flow":
                dii_colors = ["red" if val < 0 else "green" for val in dii_df[desiredcol2]]
            else:
                dii_colors = "#74a5f2"
            values = dii_df[desiredcol2].values.tolist()
            ymin = min(list(values))*1.35 if min(list(values))<0 else 0
            ymax = max(list(values))*1.35 if max(list(values))>0 else 0
            fig_dii = go.Figure(data=[go.Bar(x=dii_df['Date'], y=dii_df[desiredcol2], text=dii_df[desiredcol2], textposition="outside",
                                             textfont=dict(size=14), marker=dict(color=dii_colors))])
            fig_dii.update_layout(yaxis_title="₹ in Cr.", yaxis=dict(range=[ymin,ymax]),xaxis=dict(tickfont=dict(size=9.5), tickangle=0),
                                  height=320, width=750, margin=dict(t=40,b=10,l=25,r=10))
            st.plotly_chart(fig_dii, use_container_width=True)


        st.markdown("<br>", unsafe_allow_html=True)
        section = soup.find("section", class_="prevfidi_sec")
        tables = section.find_all("table")
        table = tables[1]
        rows = table.find("tbody").find_all("tr")
        fii_data, dii_data = [], []
        for row in rows:
            cols = row.find_all("td")
            month = cols[0].get_text(strip=True).split(" ")[0] + " " + cols[0].get_text(strip=True).split(" ")[-1]
            year = cols[0].get_text(strip=True).split(" ")[-1]
            fii_data.append([month, year, cols[1].get_text(strip=True), cols[2].get_text(strip=True), cols[3].get_text(strip=True)])
            dii_data.append([month, year, cols[4].get_text(strip=True), cols[5].get_text(strip=True), cols[6].get_text(strip=True)])
            
        headers = ['Month', 'Year', 'Gross Purchase', 'Gross Sales', 'Net Cash Flow']
        
        fii_dff = pd.DataFrame(fii_data, columns=headers)
        fii_dff[['Gross Purchase', 'Gross Sales', 'Net Cash Flow']] = fii_dff[['Gross Purchase', 'Gross Sales', 'Net Cash Flow']
                                                                            ].replace(',', '', regex=True).astype(float)
        
        dii_dff = pd.DataFrame(dii_data, columns=headers)
        dii_dff[['Gross Purchase', 'Gross Sales', 'Net Cash Flow']] = dii_dff[['Gross Purchase', 'Gross Sales', 'Net Cash Flow']
                                                                            ].replace(',', '', regex=True).astype(float)


        selected_duration = st.slider("Last few months:", min_value=0, max_value=len(fii_dff), value=12)
        fii_dff2 = fii_dff[:selected_duration]
        fii_dff2 = fii_dff2.sort_index(ascending=False).reset_index(drop=True)
        dii_dff2 = dii_dff[:selected_duration]
        dii_dff2 = dii_dff2.sort_index(ascending=False).reset_index(drop=True)

        
        fiicol2, diicol2 = st.columns([1,1])
        with fiicol2:
            st.markdown("""<style>.title {{text-align: center; font-size: 28px; font-weight: bold;}}</style><div class="title">FII Activity - for the Last {} months</div>""".format(selected_duration), unsafe_allow_html=True)
            desiredcol3 = st.radio("", headers[2:], horizontal=True, index=2, key='fii_col2')
            if desiredcol3=="Net Cash Flow":
                fii_colors = ["red" if val < 0 else "green" for val in fii_dff2[desiredcol3]]
            else:
                fii_colors = "#74a5f2"
            values = fii_dff2[desiredcol3].values.tolist()
            ymin = min(list(values))*1.35 if min(list(values))<0 else 0
            ymax = max(list(values))*1.35 if max(list(values))>0 else 0
            if selected_duration<16:
                fii_dff2['Month'] = fii_dff2['Month'].replace(' ', '<br>', regex=True)
            fig_fii2 = go.Figure(data=[go.Bar(x=fii_dff2['Month'], y=fii_dff2[desiredcol3], text=fii_dff2[desiredcol3], textposition="outside",
                                             textfont=dict(size=14), marker=dict(color=fii_colors))])
            fig_fii2.update_layout(yaxis_title="₹ in Cr.", yaxis=dict(range=[ymin,ymax]), xaxis=dict(tickfont=dict(size=9.5), tickangle=-90),
                                  height=320, width=750, margin=dict(t=40,b=10,l=10,r=25))
            st.plotly_chart(fig_fii2, use_container_width=True)

        with diicol2:
            st.markdown("""<style>.title {{text-align: center; font-size: 28px; font-weight: bold;}}</style><div class="title">DII Activity - for the Last {} months</div>""".format(selected_duration), unsafe_allow_html=True)
            desiredcol4 = st.radio("", headers[2:], horizontal=True, index=2, key='dii_col2')
            if desiredcol4=="Net Cash Flow":
                dii_colors = ["red" if val < 0 else "green" for val in dii_dff2[desiredcol4]]
            else:
                dii_colors = "#74a5f2"
            values = dii_dff2[desiredcol4].values.tolist()
            ymin = min(list(values))*1.35 if min(list(values))<0 else 0
            ymax = max(list(values))*1.35 if max(list(values))>0 else 0
            if selected_duration<16:
                dii_dff2['Month'] = dii_dff2['Month'].replace(' ', '<br>', regex=True)
            fig_dii2 = go.Figure(data=[go.Bar(x=dii_dff2['Month'], y=dii_dff2[desiredcol4], text=dii_dff2[desiredcol4], textposition="outside",
                                             textfont=dict(size=14), marker=dict(color=dii_colors))])
            fig_dii2.update_layout(yaxis_title="₹ in Cr.", yaxis=dict(range=[ymin,ymax]),xaxis=dict(tickfont=dict(size=9.5), tickangle=-90),
                                  height=320, width=750, margin=dict(t=40,b=10,l=25,r=10))
            st.plotly_chart(fig_dii2, use_container_width=True)


# # Union Budget

# In[ ]:


def inttostr(x):
    if len(x)==5:
        x = x[:2]+","+x[-3:]
    elif len(x)==6:
        x = x[:1]+","+x[1:3]+","+x[-3:]
    elif len(x)==7:
        x = x[:2]+","+x[2:4]+","+x[-3:]
    return "₹"+x


# In[ ]:


def union_budget():
    data = {"Expenditure" : ["Interest", "Transport", "Defence", "Major<br>Subsidies", "Pension", "Rural<br>Development", "Home<br>Affairs",
                             "Tax<br>Administration", "Agriculture", "Education", "Health", "Urban<br>Development", "IT and<br>Telecom", "Energy",
                             "Commerce<br>& Industry", "Finance", "Social<br>Welfare", "Scientific<br>Departments", "External<br>Affairs",
                             "North East<br>Development", "Others"],
            "Amount" : [1276338, 548649, 491732, 383407, 276618, 266817, 233211, 186632, 171437, 128650, 98311, 96777, 95298, 81174, 65553, 62924,
                        60052, 55679, 20517, 5915, 482653]}
    df = pd.DataFrame(data)
    df['Amount_str'] = df['Amount'].astype(str).apply(inttostr)
    df['Percentage'] = (100*df['Amount']/df['Amount'].sum()).round(2)
    st.markdown("<br>", unsafe_allow_html=True)
    fig_bar = go.Figure(data=[go.Bar(x=df["Expenditure"], y=df["Amount"], marker=dict(color=(color_palette*2)+['#d9ccde']), text=df["Amount_str"],
                                     hovertemplate="Category: %{x}<br>Amount: ₹%{text} Cr.<br>Percentage: %{customdata}", customdata=df["Percentage"],
                                     textposition="outside", textfont=dict(size=12))])
    fig_bar.update_layout(title=dict(text="Government's Budget Allocation for 2025-26", x=0.5, xanchor="center", font=dict(size=22)),
                          xaxis=dict(title="Category",tickangle=-90,tickfont=dict(size=13)), yaxis=dict(title="₹ in Cr.",tickfont=dict(size=1)),
                          width=650, height=500, margin=dict(l=5,r=5,t=40,b=5))
    st.plotly_chart(fig_bar, use_container_width=True)

    indata = {"Category" : ["Borrowings &<br>Other Liabilities", "Income Tax", "GST &<br>Other Tax", "Corporation<br>Tax", "Non Tax<br>Receipts",
                            "Excise", "Customs", "Non debt<br>Capital"],
              "Amount" : [24, 22, 18, 17, 9, 5, 4, 1]}
    outdata = {"Category" : ["State<br>Taxes<br>Duties", "Interest", "Central<br>Sector<br>Scheme", "Defence", "Finance<br>Commission",
                             "Centrally<br>Sponsored<br>Scheme", "Other<br>Expenditure", "Subsidies", "Pension"],
               "Amount" : [22, 20, 16, 8, 8, 8, 8, 6, 4]}
    rupcfdf = pd.DataFrame(indata)
    rupgtdf = pd.DataFrame(outdata)
    rupeecol1, rupeecol2 = st.columns([1,1])
    with rupeecol1:
        fig_rupcf = go.Figure(data=[go.Pie(labels=rupcfdf['Category'], values=rupcfdf['Amount'], hole=0.5, direction='clockwise', sort=False,
                              textfont=dict(size=12.5), textinfo='label+percent', rotation=-45)])
        fig_rupcf.update_layout(title=dict(text="Rupee comes from", x=0.5, xanchor="center", font=dict(size=24)),
                                legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"), showlegend=False, width=320, height=420,
                                margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig_rupcf, use_container_width=True)
    with rupeecol2:
        fig_rupgt = go.Figure(data=[go.Pie(labels=rupgtdf['Category'], values=rupgtdf['Amount'], hole=0.5, direction='clockwise', sort=False,
                              textfont=dict(size=12.5), textinfo='label+percent', rotation=-45)])
        fig_rupgt.update_layout(title=dict(text="Rupee goes to", x=0.5, xanchor="center", font=dict(size=24)),
                                legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"), showlegend=False, width=320, height=420,
                                margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig_rupgt, use_container_width=True)


    inflation_df = pd.read_excel("India_Inflation_rate.xlsx")
    inflation_df['Period'] = inflation_df['Month'].str[:3] + '-' + inflation_df['Year'].astype('str')
    inflation_df['Inflation rate'] = inflation_df['Inflation rate']*100
    inflcol1, inflcol2, inflcol3, inflcol4 = st.columns([2,1,1,2])
    with inflcol2:
        infl_year = st.selectbox("Select the Year", ["All"]+inflation_df['Year'].unique().tolist(), index=0)
    if infl_year!="All":
        inflation_df_sub = inflation_df[inflation_df['Year']==infl_year].copy().reset_index(drop=True)
    else:
        inflation_df_sub = inflation_df.copy()
    avginfl = str(inflation_df_sub['Inflation rate'].mean().round(2))+"%"
    with inflcol3:
        st.markdown("""<p style="text-align: center; font-size: 25px; font-weight: bold;">{avginfl}</p>""".format(avginfl=avginfl),
                    unsafe_allow_html=True)
    fig_infl = go.Figure()
    fig_infl.add_trace(go.Scatter(x=inflation_df_sub['Period'], y=inflation_df_sub['Inflation rate'], mode='lines', name='Inflation rate (%)',
                                  line=dict(color='#f5af47')))
    fig_infl.update_layout(title=dict(text='Inflation rate in India', x=0.5, xanchor='center', font=dict(size=24)), xaxis_title="Period",
                           yaxis_title="Inflation rate (%)", template="plotly_white", xaxis=dict(tickformat="%Y-%b", tickangle=-90, showgrid=True),
                           yaxis=dict(showgrid=True), width=1350, height=500)
    st.plotly_chart(fig_infl, use_container_width=True)


# # Main Function

# In[ ]:


def main_dashboard():
    st.markdown("""<style>.title {text-align: center; font-size: 34px; font-weight: bold;}</style><div class="title">Stock Analysis</div>""",
            unsafe_allow_html=True)
    page_navigation()

    if st.session_state.page == "Stock Analysis":
        stock_analysis()

    elif st.session_state.page == "Index Performance":
        index_performance()

    elif st.session_state.page == "Strategy Indices":
        nifty_strategy_indices()

    elif st.session_state.page == "FII DII Activity":
        fii_dii_activity()

    elif st.session_state.page == "Union Budget":
        union_budget()


# In[ ]:


def main():
    valid_emails = load_valid_emails()
    if "page" not in st.session_state:
        st.session_state.page = "Union Budget"
    
    if st.session_state.page == "Login":
        login_page(valid_emails)
    else:
        main_dashboard()


# In[ ]:


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# # Testing Codes

# In[ ]:


# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.common.keys import Keys
# import pandas as pd
# from bs4 import BeautifulSoup
# import time


# In[ ]:


# start = time.time()
# options = webdriver.ChromeOptions()
# options.add_argument("--headless")
# driver = webdriver.Chrome(options=options)

# driver.get("https://www.screener.in/login/")

# email_input = driver.find_element(By.NAME, "username")
# password_input = driver.find_element(By.NAME, "password")

# email_input.send_keys("stockanalysiswithesh@gmail.com")
# password_input.send_keys("eswaraprasath@22M")
# password_input.send_keys(Keys.RETURN)

# time.sleep(3)

# driver.get("https://www.screener.in/company/NI15/")
# time.sleep(3)  # Wait for data to load

# # Extract page source
# soup = BeautifulSoup(driver.page_source, "html.parser")

# # Find table
# table = soup.find("table", class_="data-table text-nowrap striped mark-visited")
# table = soup.select_one("table.data-table")
# # Convert to DataFrame
# df = pd.read_html(str(table))[0]

# # Close the browser
# driver.quit()
# end = time.time()


# In[ ]:




