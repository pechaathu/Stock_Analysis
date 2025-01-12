#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[5]:


import io
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


# # Functions

# In[2]:


def calculate_holdings(group):
    buy_units = group.loc[group["Action"] == "Buy", "Units"].sum()
    sell_units = group.loc[group["Action"] == "Sell", "Units"].sum()
    remaining_units = buy_units - sell_units

    buy_amount = group.loc[group["Action"] == "Buy", "Total Amount"].sum()
    sell_amount = (group.loc[group["Action"] == "Sell", "Units"] * 
                   group.loc[group["Action"] == "Sell", "Price per Unit"]).sum()

    total_investment = buy_amount - sell_amount

    return pd.Series({"Total Units": remaining_units, "Total Investment": total_investment})


# In[25]:


def fetch_current_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        return stock.history(period="1d")["Close"].iloc[-1]
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None


# In[24]:


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


# In[9]:


def webscrap(webcontent, label):
    label_tag = webcontent.find(string=lambda text: text and label in text)
    if label_tag:
        parent = label_tag.parent
        value_tag = parent.find_next(class_='number')
        if value_tag:
            value_string = value_tag.text.strip()
            return value_string


# # Stock Watchlist Page

# ## Stock Filters & Price Chart

# In[ ]:


def prologue(companies):
    comparison = False
    headcol1, headcol2 = st.columns([8.5,1.5])
    industries = list(companies.keys())
    with headcol1:
        industry = st.radio("Select Sector:", industries, index=0, horizontal=True)
    stocks = list(companies[industry].keys())
    with headcol2:
        st.markdown('<p style="font-size:14px;">Benchmark Comparison:</p>',unsafe_allow_html=True)
        comparison = st.toggle("NIFTY_50", value=False)
    
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
    
    stock_data = stock_data[stock_data['Date']>=start_date].reset_index(drop=True)
    
    if comparison==True:
        bm_data = yf.download("^NSEI")
        bm_data.columns = ['_'.join(col).strip() for col in bm_data.columns.values]
        bm_data['Date'] = bm_data.index
        bm_data = bm_data[bm_data['Date']>=start_date].reset_index(drop=True)
        bm_data = bm_data.reset_index(drop=True)
        stock_data = pd.merge(stock_data, bm_data, how="left", on='Date')
        stock_data['Variation_^NSEI'] = np.round(100*(stock_data['Close_^NSEI']-stock_data['Close_^NSEI'][0])/stock_data['Close_^NSEI'][0],2)
        if stock_symbol!="^NSEI":
            stock_data['Variation_'+stock_symbol] = np.round(100*(
                                        stock_data['Close_'+stock_symbol]-stock_data['Close_'+stock_symbol][0])/stock_data['Close_'+stock_symbol][0],2)
        
        fig_compline = go.Figure()
        fig_compline.add_trace(go.Scatter(x=stock_data['Date'],y=stock_data['Variation_'+stock_symbol],mode='lines',name=stock_name,
                                          line=dict(color='blue')))
        if stock_symbol!="^NSEI":
            fig_compline.add_trace(go.Scatter(x=stock_data['Date'],y=stock_data['Variation_^NSEI'],mode='lines',name='Nifty 50',
                                              line=dict(color='yellow')))
        fig_compline.update_layout(title=dict(text="Nifty-50  v/s  "+stock_name, x=0.5, xanchor='center'), xaxis_title="Date",yaxis_title="% variation",
                               template="plotly_white", xaxis=dict(showgrid=True), yaxis=dict(showgrid=True), width=1350, height=500)
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
    return stock_name, industry, alltimehigh, alltimelow, high52week, low52week, high3month, low3month


# ### Get Contents

# In[ ]:


def getcontents(stockurls, stock_name):
    stock_url = stockurls[stock_name]
    response = requests.get(stock_url)
    content = bs(response.content, 'html.parser')
    return content


# ### Basic Info

# In[ ]:


def basicinfo(content, alltimehigh, alltimelow, high52week, low52week, high3month, low3month):
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
    
        with st.expander("Details"):
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
            df_quarterly_results = df_quarterly_results[(df_quarterly_results['Type']=="Operating Profit")|
                                                        (df_quarterly_results['Type']=="Profit before tax")|
                                                        (df_quarterly_results['Type']=="Net Profit")].reset_index(drop=True)
            df_quarterly_results_cols = df_quarterly_results.columns.tolist()
            df_quarterly_results_cols.remove('Type')
            for col in df_quarterly_results_cols:
                df_quarterly_results[col] = pd.to_numeric(df_quarterly_results[col].str.replace(',', ''), errors='coerce')
            
        
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
            df_yearly_results = df_yearly_results[(df_yearly_results['Type']=="Operating Profit")|(df_yearly_results['Type']=="Profit before tax")|
                                                  (df_yearly_results['Type']=="Net Profit")].reset_index(drop=True)
            df_yearly_results_cols = df_yearly_results.columns.tolist()
            df_yearly_results_cols.remove('Type')
            for col in df_yearly_results_cols:
                df_yearly_results[col] = pd.to_numeric(df_yearly_results[col].str.replace(',', ''), errors='coerce')
        
    
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
    
    
        with st.expander("Profit & Loss"):
            resultoptioncol, tenureoptioncol, extracol1 = st.columns([1,0.5,1])
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
                resultoption = st.radio("Profit", resultoptions, index=0, horizontal=True)
            
            barchartcol, waterfallcol = st.columns([1.5,1])
            with barchartcol:
                filtered_data_df = dfrequired[dfrequired["Type"]==resultoption]
                values = filtered_data_df.iloc[0,:-1]
                intvalues = [item for item in values if not isinstance(item, str)]
                ymax = max(list(intvalues))*1.25
                if ymax>0:
                    ymin=0
                else:
                    ymin=ymax
                    ymax=0
                fig_pcbar = go.Figure(data=[go.Bar(x=dfrequiredcols, y=values, name=resultoption, text=values, textposition="outside",
                                                   textfont=dict(size=14), marker=dict(color="#74a5f2"))])
                fig_pcbar.update_layout(title=dict(text="Profit Comparison", x=0.5, xanchor="center", font=dict(size=18)), yaxis_title="₹ in Cr.",
                                        xaxis=dict(categoryorder="array", tickfont=dict(size=9.5), tickangle=0, categoryarray=dfrequiredcols),
                                        yaxis=dict(range=[0,ymax]), height=320, width=750, margin=dict(t=40,b=10,l=15,r=10))
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
    
        with st.expander("Balance Sheet"):
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
    
            
        with st.expander("Share Holders"):
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
                values = filtered_data.iloc[0,:-1]
                fig_shbar = go.Figure(data=[go.Bar(x=dfreqcols, y=values, name=shareholder, text=values, textposition="outside", textfont=dict(size=14),
                                                   marker=dict(color=color_palette[shareholders.index(shareholder)]))])
                fig_shbar.update_layout(title=dict(text="Share Holding Pattern", x=0.5, xanchor="center", font=dict(size=22)),xaxis_title=shtenure[:-2],
                                        yaxis_title="Percentage(%)",xaxis=dict(categoryorder="array",categoryarray=dfreqcols),yaxis=dict(range=[0,100]),
                                        height=340, width=830, margin=dict(t=40,b=10,l=15,r=10))
                st.plotly_chart(fig_shbar, use_container_width=True)


# # Main Dashboard Function

# In[ ]:


def main_dashboard():
    st.markdown("""<style>.title {text-align: center; font-size: 34px; font-weight: bold;}</style><div class="title">Stock Analysis dashboard</div>""",
            unsafe_allow_html=True)
    
    companies = {'Index':{"Nifty50":"^NSEI", "Sensex":"^BSESN", "NiftyAuto":"^CNXAUTO"}, 
                 'Automotive':{"Tata Motors":"TATAMOTORS.NS", "Mahindra & Mahindra":"M&M.NS", "Hero Motocorp":"HEROMOTOCO.NS"}, 
                 'Banking':{"HDFC":"HDFCBANK.NS", "ICICI":"ICICIBANK.NS", "IOB":"IOB.NS", "SBI":"SBIN.NS"},
                 'Energy':{"Tata Power":"TATAPOWER.NS", "JSW Energy":"JSWENERGY.NS", "Adani Energy Solutions":"ADANIENSOL.NS"},
                 'Electr Equip':{"Exicom Tele-Systems":"EXICOM.NS","ABB":"ABB.NS"},
                 'FMCG':{"ITC":"ITC.NS", "Nestle":"NESTLEIND.NS", "Varun Beverages":"VBL.NS", "Bikaji Foods":"BIKAJI.NS"},
                 'IT':{"TCS":"TCS.NS", "Wipro":"WIPRO.NS", "Tech Mahindra":"TECHM.NS"},
                 'Pharma':{"Cipla":"CIPLA.NS", "Sun Pharma":"SPARC.NS", "Mankind Pharma":"MANKIND.NS", "Natco Pharma":"NATCOPHARM.NS"},
                 'E-Commerce':{"Zomato":"ZOMATO.NS", "Swiggy":"SWIGGY.NS"}}
    web = 'https://www.screener.in/company/'
    con = 'consolidated/'
    stockurls = {'Nifty50':f'{web}NIFTY/', 'Sensex':f'{web}1001/', 'NiftyAuto':f'{web}CNXAUTO/', 'Tata Motors':f'{web}TATAMOTORS/{con}',
                 'Mahindra & Mahindra':f'{web}M&M/{con}', 'Hero Motocorp':f'{web}HEROMOTOCO/{con}', 'HDFC':f'{web}HDFCBANK/{con}',
                 'ICICI':f'{web}ICICIBANK/{con}', 'IOB':f'{web}IOB/', 'SBI':f'{web}SBIN/{con}', 'Tata Power':f'{web}TATAPOWER/{con}',
                 'JSW Energy':f'{web}JSWENERGY/{con}', 'Adani Energy Solutions':f'{web}ADANIENSOL/{con}','Exicom Tele-Systems':f'{web}EXICOM/{con}',
                 'ABB':f'{web}ABB/', 'ITC':f'{web}ITC/{con}', 'Nestle':f'{web}/NESTLEIND/', 'Varun Beverages':f'{web}/VBL/{con}',
                 'Bikaji Foods':f'{web}/BIKAJI/', 'TCS':f'{web}/TCS/{con}', 'Wipro':f'{web}/WIPRO/{con}', 'Tech Mahindra':f'{web}/TECHM/{con}',
                 'Cipla':f'{web}/CIPLA/{con}', 'Sun Pharma':f'{web}/SUNPHARMA/{con}', 'Mankind Pharma':f'{web}/MANKIND/',
                 'Natco Pharma':f'{web}/NATCOPHARM/{con}', 'Zomato':f'{web}/ZOMATO/{con}', 'Swiggy':f'{web}/SWIGGY/{con}'}

    stock_name, industry, alltimehigh, alltimelow, high52week, low52week, high3month, low3month = prologue(companies)
    content = getcontents(stockurls, stock_name)
    basicinfo(content, alltimehigh, alltimelow, high52week, low52week, high3month, low3month)
    proscons(content, industry)
    profitloss(content, industry)
    balancesheet(content, industry)
    shareholding(content, industry)


# ## Login Page

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
    msg['From'] = "eswaraprasath.m@gmail.com"
    msg['To'] = recipient_email
    msg['Subject'] = "OTP for Logging into Stock Analysis dashboard"

    msg.attach(MIMEText(f'OTP: {otpnum}', 'plain'))
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login("eswaraprasath.m@gmail.com", "dwjs oben nmnp zfas")
            text = msg.as_string()
            server.sendmail("eswaraprasath.m@gmail.com", recipient_email, text)
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
                    st.session_state.page = "Dashboard"
                    st.experimental_rerun()
                else:
                    st.error("Invalid OTP!...")


# In[ ]:


def main():
    valid_emails = load_valid_emails()
    if "page" not in st.session_state:
        st.session_state.page = "Login"
    
    if st.session_state.page == "Login":
        login_page(valid_emails)
    elif st.session_state.page == "Dashboard":
        main_dashboard()


# In[ ]:


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




