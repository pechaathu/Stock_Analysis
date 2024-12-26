#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[1]:


import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from contextlib import redirect_stderr
import io
import sys
import warnings
warnings.filterwarnings('ignore')


# ## Page Settings

# In[ ]:


st.set_page_config(layout='wide')


# In[ ]:


if "page" not in st.session_state:
    st.session_state.page = "Stock Portfolio"

st.markdown("""<style>.menu-container {display: flex; justify-content: center; align-items: center; margin-top: 1px;
            }.menu-button {padding: 0.2px 0.60x; font-size: 24px; font-weight: bold; background-color: #007BFF; color: white; border: none; 
            border-radius: 0.1px; margin: 0 0.2px; cursor: pointer; text-align: center; text-decoration: none; transition: background-color 0.3s ease;
            }.menu-button.selected {background-color: #0056b3;}.menu-button:hover {background-color: #0056b3;}</style>""", unsafe_allow_html=True)

st.markdown('<div class="menu-container">', unsafe_allow_html=True)
colmenu1, colmenu2, colmenu3, colmenu4, colmenu5 = st.columns([2,1,1,1,2])

with colmenu2:
    if st.button("Stock Portfolio", key="st_portfolio", use_container_width=True):
        st.session_state.page = "Stock Portfolio"

with colmenu3:
    if st.button("Mutual Fund Portfolio", key="mf_portfolio", use_container_width=True):
        st.session_state.page = "Mutual Fund Portfolio"

with colmenu4:
    if st.button("Stock Watchlist", key="st_watchlist", use_container_width=True):
        st.session_state.page = "Stock Watchlist"

st.markdown('</div>', unsafe_allow_html=True)


# ## Functions

# In[26]:


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


# ## Stock Portfolio Page

# In[ ]:


if st.session_state.page == "Stock Portfolio":
    st.markdown("""<style>.title {text-align: center; font-size: 34px; font-weight: bold;}</style><div class="title">My Stock Portfolio</div><br>""",
            unsafe_allow_html=True)
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
    
    metriccol1, metriccol2, metriccol3, metriccol4, metriccol5, metriccol6, metriccol7 = st.columns([0.75,1,1,1,1,1,0.75])
    with metriccol2:
        st.metric(label="Total Investment", value="₹"+str(holdings['Total Investment'].sum().round(2)))
    with metriccol3:
        st.metric(label="Current Value", value="₹"+str(holdings['Current Value'].sum().round(2)),
                  delta="₹"+str((holdings['Current Value'].sum()-holdings['Total Investment'].sum()).round(2)))
    with metriccol4:
        st.metric("Total Stocks", len(holdings['Stock Name'].unique().tolist()))
    with metriccol5:
        st.metric("Total Buys", df[df['Action']=="Buy"]['Action'].count())
    with metriccol6:
        st.metric("Total Sells", df[df['Action']=="Sell"]['Action'].count())
    
    colpie, colbar = st.columns([1, 1.25])
    with colpie:
        fig_pie = go.Figure(data=[go.Pie(labels=holdings["Stock Name"], values=holdings["Total Investment"], hole=0.5, textfont=dict(size=14))])
        fig_pie.update_layout(title=dict(text="Current Holdings", x=0.5, xanchor="center", font=dict(size=22)),
                              legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"), showlegend=True, width=550, height=450)
        st.plotly_chart(fig_pie)

    with colbar:
        fig_bar = go.Figure(data=[go.Bar(x=holdings["Stock Name"], y=holdings["Returns"], marker=dict(color="lightblue"),
                                  text=holdings["Returns"].round(2), textposition="inside", textfont=dict(size=14))])
        fig_bar.update_layout(title=dict(text="Returns (%)", x=0.5, xanchor="center", font=dict(size=22)), xaxis=dict(title="Stock Name",tickangle=-10),
                              yaxis_title="Returns (%)", width=650, height=450)
        st.plotly_chart(fig_bar)

    coltree, colline = st.columns([1, 1.25])
    with coltree:
        fig_tree = px.treemap(df, path=["Sector"], values="Total Amount", color="Sector", color_discrete_sequence=px.colors.qualitative.Set1)
        fig_tree.update_traces(textinfo="label+value+percent entry", textfont_size=14)
        fig_tree.update_layout(title=dict(text="Stock Diversification", x=0.5, xanchor='center', font=dict(size=22)), template="plotly_white",
                               width=550, height=500)
        st.plotly_chart(fig_tree)

    with colline:
        fig_inv = go.Figure()
        fig_inv.add_trace(go.Scatter(x=investment_journey_df['Date'], y=investment_journey_df['Cumulative Investment'], mode='lines', name='Investment',
                                     line=dict(color='blue')))
        fig_inv.add_trace(go.Scatter(x=investment_journey_df['Date'], y=investment_journey_df['Net Value'], mode='lines', name='Performance',
                                     line=dict(color='orange')))
        fig_inv.update_layout(title=dict(text="Stock Investment Journey", x=0.5, xanchor='center', font=dict(size=22)), xaxis_title="Date",
                              yaxis_title="Rupees (₹)", template="plotly_white", xaxis=dict(showgrid=True), yaxis=dict(showgrid=True),
                              width=650, height=500, xaxis_range=[pd.Timestamp("2024-11-01"), pd.Timestamp(datetime.today().strftime("%Y-%m-%d"))])
        st.plotly_chart(fig_inv)


# ## Mutual Fund Portfolio Page

# In[ ]:


if st.session_state.page == "Mutual Fund Portfolio":
    st.markdown("""<style>.title {text-align: center; font-size: 34px; font-weight: bold;}</style><div class="title">Mutual Fund Portfolio</div>""",
                unsafe_allow_html=True)
    df = pd.read_excel("MutualFund_Database.xlsx")
    df = df.drop(columns=['S.No.'])
    fundhouses = df['Mutual Fund House'].unique().tolist()
    fundhousedf_list = []
    returns_list = []
    current_value = 0
    for i in range(len(fundhouses)):
        fundhousedf = df[df['Mutual Fund House']==fundhouses[i]].reset_index(drop=True)
        start_date = fundhousedf['Date'].min().date()
        end_date = datetime.today()
        date_range = pd.date_range(start=start_date, end=end_date)
        dff = pd.DataFrame({'Date': date_range})
        fundhousedf = pd.merge(fundhousedf, dff, how="outer", on='Date')
        fundhousedf = fundhousedf.sort_values(by='Date')
        fundhousedf[['Mutual Fund House','Fund House Symbol']] = fundhousedf[['Mutual Fund House','Fund House Symbol']].fillna(method='ffill')
        historical_prices = fetch_historical_prices_batch(fundhousedf['Fund House Symbol'].unique().tolist()[0], fundhousedf['Date'])
        fundhousedf["NAV"] = fundhousedf["Date"].map(historical_prices)
        fundhousedf = fundhousedf[fundhousedf['NAV'].notna()].reset_index(drop=True)
        fundhousedf_list.append(fundhousedf)
        fundhousedf['Units'] = fundhousedf['Amount']/fundhousedf['NAV']
        fundhousedf['Total Investment'] = fundhousedf['Amount'].cumsum()
        fundhousedf['Cumulative Units'] = fundhousedf['Units'].cumsum()
        fundhousedf[['Total Investment','Cumulative Units']] = fundhousedf[['Total Investment','Cumulative Units']].fillna(method='ffill')
        fundhousedf['Current Value'] = fundhousedf['NAV']*fundhousedf['Cumulative Units']
        # current_value += fundhousedf[fundhousedf['Date']==fundhousedf['Date'].max()]['Current Value'].max()
        current_value += fundhousedf['Current Value'][len(fundhousedf)-1]
        returns_list.append(np.round(100*(fundhousedf[fundhousedf['Date']==fundhousedf['Date'].max()
                                ]['Current Value'].max()-fundhousedf[fundhousedf['Date']==fundhousedf['Date'].max()
                                ]['Total Investment'].max())/fundhousedf[fundhousedf['Date']==fundhousedf['Date'].max()]['Total Investment'].max(),2))
    df = pd.concat(fundhousedf_list,ignore_index=True)

    metriccol1, metriccol2, metriccol3, metriccol4, metriccol5, metriccol6, metriccol7 = st.columns([0.75,1,1,1,1,1,0.75])
    with metriccol2:
        st.metric(label="Total Investment", value="₹"+str(df['Amount'].sum().round(2)))
    with metriccol3:
        st.metric(label="Current Value", value="₹"+str(np.round(current_value,2)),delta="₹"+str((current_value-df['Amount'].sum()).round(2)))
    with metriccol4:
        st.metric("Total Funds", len(fundhouses))
    with metriccol5:
        st.metric("Total SIPs", len(list(set([i.strftime("%b") for i in df[df['Action']=="SIP"]['Date'].unique().tolist()]))))
    with metriccol6:
        st.metric("Total Withdrawls", df[df['Action']=="Withdrawl"]['Action'].count())

    colpie, colbar = st.columns([1, 1.25])
    with colpie:
        fig_pie = go.Figure(data=[go.Pie(labels=df["Mutual Fund House"], values=df["Amount"], hole=0.5, textfont=dict(size=14))])
        fig_pie.update_layout(title=dict(text="Fund wise Investment breakup", x=0.5, xanchor="center", font=dict(size=22)),
                              legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"), showlegend=True, width=550, height=450)
        st.plotly_chart(fig_pie)

    with colbar:
        fig_bar = go.Figure(data=[go.Bar(x=fundhouses, y=returns_list, marker=dict(color="lightblue"), text=returns_list, textposition="inside",
                                         textfont=dict(size=14))])
        fig_bar.update_layout(title=dict(text="Returns (%)", x=0.5, xanchor="center", font=dict(size=22)), xaxis=dict(title="Fund House",tickangle=-10),
                              yaxis_title="Returns (%)", width=650, height=450)
        st.plotly_chart(fig_bar)

    dfinv = df.groupby('Date', as_index=False)['Total Investment'].sum()
    dfcur = df.groupby('Date', as_index=False)['Current Value'].sum()
    fig_inv = go.Figure()
    fig_inv.add_trace(go.Scatter(x=dfinv['Date'], y=dfinv['Total Investment'], mode='lines', name='Investment', line=dict(color='blue')))
    fig_inv.add_trace(go.Scatter(x=dfcur['Date'], y=dfcur['Current Value'], mode='lines', name='Performance', line=dict(color='orange')))
    fig_inv.update_layout(title=dict(text="Mutual Fund Investment Journey", x=0.5, xanchor='center', font=dict(size=22)), xaxis_title="Date",
                          yaxis_title="Rupees (₹)", template="plotly_white", xaxis=dict(showgrid=True), yaxis=dict(showgrid=True),
                          width=1350, height=500)
    st.plotly_chart(fig_inv)


# ## Stock Watchlist Page

# In[ ]:


if st.session_state.page == "Stock Watchlist":
    comparison = False
    companies = {'Index':{"Nifty50":"^NSEI", "Sensex":"^BSESN", "NiftyAuto":"^CNXAUTO"}, 
                 'Automotive':{"Tata Motors":"TATAMOTORS.NS", "Mahindra & Mahindra":"M&M.NS", "Hero Motocorp":"HEROMOTOCO.NS"}, 
                 'Banking':{"HDFC":"HDFCBANK.NS", "ICICI":"ICICIBANK.NS", "IOB":"IOB.NS", "SBI":"SBIN.NS"},
                 'Energy':{"Tata Power":"TATAPOWER.NS", "JSW Energy":"JSWENERGY.NS", "Adani Energy Solutions":"ADANIENSOL.NS"},
                 'Electr Equip':{"Exicom Tele-Systems":"EXICOM.NS","ABB":"ABB.NS"},
                 'FMCG':{"ITC":"ITC.NS", "Nestle":"NESTLEIND.NS", "Varun Beverages":"VBL.NS", "Bikaji Foods":"BIKAJI.NS"},
                 'IT':{"TCS":"TCS.NS", "Wipro":"WIPRO.NS", "Tech Mahindra":"TECHM.NS", "Sonata Softwares":"SONATASOFTW.NS"},
                 'Pharma':{"Cipla":"CIPLA.NS", "Sun Pharma":"SPARC.NS", "Mankind Pharma":"MANKIND.NS", "Natco Pharma":"NATCOPHARM.NS"}}
    if comparison==True:
        companies.pop('Index')
    st.markdown("""<style>.title {text-align: center; font-size: 34px; font-weight: bold;}</style><div class="title">Stock Watchlist</div>""",
            unsafe_allow_html=True)
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

    try:
        stock_data = yf.download(stock_symbol)
    
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = ['_'.join(col).strip() for col in stock_data.columns.values]
        stock_data['Date'] = stock_data.index
        stock_data = stock_data.reset_index(drop=True)
        stock_data['50_DMA'] = stock_data['Close_'+stock_symbol].rolling(window=50).mean()
        stock_data['200_DMA'] = stock_data['Close_'+stock_symbol].rolling(window=200).mean()
    
    except Exception as e:
        st.error(f"Error fetching data: {e}")

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
        bm_data = yf.download("^NSEI",start=start_date,end=datetime.today())
        bm_data.columns = ['_'.join(col).strip() for col in bm_data.columns.values]
        bm_data['Date'] = bm_data.index
        bm_data = bm_data.reset_index(drop=True)
        bm_data['Variation_^NSEI'] = np.round(100*(bm_data['Close_^NSEI']-bm_data['Close_^NSEI'][0])/bm_data['Close_^NSEI'],2)
        if stock_symbol!="^NSEI":
            stock_data['Variation_'+stock_symbol] = np.round(100*(
                                            stock_data['Close_'+stock_symbol]-stock_data['Close_'+stock_symbol][0])/stock_data['Close_'+stock_symbol],2)
        stock_data = pd.merge(stock_data, bm_data, how="left", on='Date')
        
        fig_compline = go.Figure()
        fig_compline.add_trace(go.Scatter(x=stock_data['Date'],y=stock_data['Variation_'+stock_symbol],mode='lines',name=stock_name,
                                          line=dict(color='blue')))
        if stock_symbol!="^NSEI":
            fig_compline.add_trace(go.Scatter(x=stock_data['Date'],y=stock_data['Variation_^NSEI'],mode='lines',name='Nifty 50',
                                              line=dict(color='yellow')))
        fig_compline.update_layout(title=dict(text="Nifty 50 v/s "+stock_name, x=0.5, xanchor='center'), xaxis_title="Date",yaxis_title="% variation",
                               template="plotly_white", xaxis=dict(showgrid=True), yaxis=dict(showgrid=True), width=1350, height=500)
        st.plotly_chart(fig_compline)

    else:
        with col6:
            st.markdown('<p style="font-size:14px;">Add on:</p>',unsafe_allow_html=True)
            placeholder = st.empty()
            with placeholder.container():
                cb1, cb2, cb3= st.columns([1,1,1.5])
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
            fig_line.update_layout(title=dict(text=stock_name, x=0.5, xanchor='center'), xaxis_title="Date",yaxis_title="Close Price (INR)",
                                   template="plotly_white", xaxis=dict(showgrid=True), yaxis=dict(showgrid=True), width=1350, height=500)
            st.plotly_chart(fig_line)
    
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
            st.plotly_chart(fig_candle)


# ## Testing Codes

# In[ ]:




