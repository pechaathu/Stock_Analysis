#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import io
import sys
import requests
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup as bs
from contextlib import redirect_stderr
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')
pio.templates["custom_template"] = pio.templates["plotly"]
pio.templates["custom_template"]["layout"]["colorway"] = px.colors.qualitative.Plotly
pio.templates.default = "custom_template"
color_palette = px.colors.qualitative.Plotly


# # Page Settings

# In[2]:


st.set_page_config(layout='wide')


# In[3]:


if "page" not in st.session_state:
    st.session_state.page = "Salary Portfolio"

st.markdown("""<style>.menu-container {display: flex; justify-content: center; align-items: center; margin-top: 1px;
            }.menu-button {padding: 0.2px 0.60x; font-size: 24px; font-weight: bold; background-color: #007BFF; color: white; border: none; 
            border-radius: 0.1px; margin: 0 0.2px; cursor: pointer; text-align: center; text-decoration: none; transition: background-color 0.3s ease;
            }.menu-button.selected {background-color: #0056b3;}.menu-button:hover {background-color: #0056b3;}</style>""", unsafe_allow_html=True)

st.markdown('<div class="menu-container">', unsafe_allow_html=True)
colmenu1, colmenu2, colmenu3, colmenu4, colmenu5 = st.columns([1,1,1,1,1])

with colmenu2:
    if st.button("Salary Portfolio", key="slry_portfolio", use_container_width=True):
        st.session_state.page = "Salary Portfolio"

with colmenu3:
    if st.button("Stock Portfolio", key="st_portfolio", use_container_width=True):
        st.session_state.page = "Stock Portfolio"

with colmenu4:
    if st.button("Mutual Fund Portfolio", key="mf_portfolio", use_container_width=True):
        st.session_state.page = "Mutual Fund Portfolio"

st.markdown('</div>', unsafe_allow_html=True)


# # Functions

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


def fetch_current_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        return stock.history(period="1d")["Close"].iloc[-1]
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None


# In[6]:


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


# In[8]:


def webscrap(webcontent, label):
    label_tag = webcontent.find(string=lambda text: text and label in text)
    if label_tag:
        parent = label_tag.parent
        value_tag = parent.find_next(class_='number')
        if value_tag:
            value_string = value_tag.text.strip()
            return value_string


# # Dashboard Pages

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
                  delta=f"₹{abs((holdings['Current Value'].sum()-holdings['Total Investment'].sum())):.2f}" if (holdings['Current Value'].sum()-
                        holdings['Total Investment'].sum())>0 else f"-₹{abs((holdings['Current Value'].sum()-holdings['Total Investment'].sum())):.2f}")
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
        st.plotly_chart(fig_pie, use_container_width=True)

    with colbar:
        fig_bar = go.Figure(data=[go.Bar(x=holdings["Stock Name"], y=holdings["Returns"], marker=dict(color="lightblue"),
                                  text=holdings["Returns"].round(2), textposition="inside", textfont=dict(size=14))])
        fig_bar.update_layout(title=dict(text="Returns (%)", x=0.5, xanchor="center", font=dict(size=22)), xaxis=dict(title="Stock Name",tickangle=-10),
                              yaxis_title="Returns (%)", width=650, height=450)
        st.plotly_chart(fig_bar, use_container_width=True)

    coltree, colline = st.columns([1, 1.25])
    with coltree:
        fig_tree = px.treemap(holdings, path=["Sector"], values="Total Amount", color="Sector", color_discrete_sequence=px.colors.qualitative.Set1)
        fig_tree.update_traces(textinfo="label+value+percent entry", textfont_size=14)
        fig_tree.update_layout(title=dict(text="Stock Diversification", x=0.5, xanchor='center', font=dict(size=22)), template="plotly_white",
                               width=550, height=500)
        st.plotly_chart(fig_tree, use_container_width=True)

    with colline:
        fig_inv = go.Figure()
        fig_inv.add_trace(go.Scatter(x=investment_journey_df['Date'], y=investment_journey_df['Cumulative Investment'], mode='lines', name='Investment',
                                     line=dict(color='blue')))
        fig_inv.add_trace(go.Scatter(x=investment_journey_df['Date'], y=investment_journey_df['Net Value'], mode='lines', name='Performance',
                                     line=dict(color='orange')))
        fig_inv.update_layout(title=dict(text="Stock Investment Journey", x=0.5, xanchor='center', font=dict(size=22)), xaxis_title="Date",
                              yaxis_title="Rupees (₹)", template="plotly_white", xaxis=dict(showgrid=True), yaxis=dict(showgrid=True),
                              width=650, height=500, xaxis_range=[pd.Timestamp("2024-11-01"), pd.Timestamp(datetime.today().strftime("%Y-%m-%d"))])
        st.plotly_chart(fig_inv, use_container_width=True)

    colperfm1, colperfm2, colperfm3 = st.columns([2,1,2])
    with colperfm2:
        selected_stock = st.selectbox("Select the Stock", mystocks, index=0)
        st.markdown("<br>", unsafe_allow_html=True)
    stockdf_sub = stockdffull[stockdffull['Stock Name']==selected_stock].reset_index(drop=True)
    invested_amount = stockdf_sub[stockdf_sub['Action']=="Buy"]['Total Amount'].sum() + stockdf_sub[stockdf_sub['Action']=="Sell"]['Total Amount'].sum()
    current_value = stockdf_sub['Net Value'].iloc[-1]
    avg_stock_price = stockdf_sub[stockdf_sub['Action']=="Buy"]['Total Amount'].sum()/stockdf_sub[stockdf_sub['Action']=="Buy"]['Units'].sum()
    invested_amount_str = "₹"+str(np.round(invested_amount,2))
    current_value_str = "₹"+str(np.round(current_value,2))
    avg_stock_price_str = "₹"+str(np.round(avg_stock_price,2))
    bought_units = stockdf_sub[stockdf_sub['Action']=="Buy"]['Units'].sum()
    sold_units = -1*stockdf_sub[stockdf_sub['Action']=="Sell"]['Units'].sum()
    holding_units = stockdf_sub['Units'].sum()
    chart_color = 'green' if current_value>invested_amount else 'red'
    fill_color = "rgba(0, 255, 0, 0.15)" if current_value>invested_amount else "rgba(255, 0, 0, 0.15)"
    colperfm4, colperfm5, colperfm6, colperfm7 = st.columns([0.5,3,0.15,0.35])
    with colperfm4:
        st.metric("Invested Amount:", invested_amount_str)
        st.metric("Current Value:", current_value_str)
        st.metric("Avg. Stock Price:", avg_stock_price_str)
    with colperfm5:
        fig_stprfm = go.Figure()
        fig_stprfm.add_trace(go.Scatter(x=stockdf_sub['Date'], y=stockdf_sub['Closing Price'], mode='lines', name=selected_stock,
                                        line=dict(color=chart_color)))
        fig_stprfm.add_trace(go.Scatter(x=stockdf_sub['Date'], y=stockdf_sub['Closing Price'], mode='lines', name="", line=dict(color=chart_color),
                                        fill='tozeroy', fillcolor=fill_color))
        fig_stprfm.add_trace(go.Scatter(x=stockdf_sub[stockdf_sub['Action']=="Buy"]['Date'], mode='markers+text', name='Buy', textfont=dict(size=15),
                                        y=stockdf_sub[stockdf_sub['Action']=="Buy"]['Closing Price'], marker=dict(color="white", size=13),
                                        text=stockdf_sub[stockdf_sub['Action']=="Buy"]['Closing Price'].round(2), textposition='top center'))
        fig_stprfm.update_layout(title=dict(text=selected_stock+" - performance", x=0.5, xanchor='center', font=dict(size=22)), xaxis_title="Date",
                                 yaxis_title="Rupees (₹)", template="plotly_white", xaxis=dict(showgrid=True), yaxis=dict(showgrid=True),
                                 width=650, height=500, showlegend=False,
                                 yaxis_range=[stockdf_sub['Closing Price'].min()-(0.1*stockdf_sub['Closing Price'].min()),
                                              stockdf_sub['Closing Price'].max()+(0.1*stockdf_sub['Closing Price'].max())])
        st.plotly_chart(fig_stprfm, use_container_width=True)
    with colperfm7:
        st.write("Units")
        st.metric("Bought:", int(bought_units))
        st.metric("Sold:", int(sold_units))
        st.metric("Holdings:", int(holding_units))


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
        current_value += fundhousedf['Current Value'][len(fundhousedf)-1]
        returns_list.append(np.round(100*(fundhousedf[fundhousedf['Date']==fundhousedf['Date'].max()
                                ]['Current Value'].max()-fundhousedf[fundhousedf['Date']==fundhousedf['Date'].max()
                                ]['Total Investment'].max())/fundhousedf[fundhousedf['Date']==fundhousedf['Date'].max()]['Total Investment'].max(),2))
    df = pd.concat(fundhousedf_list,ignore_index=True)

    metriccol1, metriccol2, metriccol3, metriccol4, metriccol5, metriccol6, metriccol7 = st.columns([0.75,1,1,1,1,1,0.75])
    with metriccol2:
        st.metric(label="Total Investment", value="₹"+str(df['Amount'].sum().round(2)))
    with metriccol3:
        st.metric(label="Current Value", value="₹"+str(np.round(current_value,2)),
                  delta=f"₹{abs((current_value-df['Amount'].sum())):.2f}" if (current_value-
                        df['Amount'].sum())>0 else f"-₹{abs(current_value-df['Amount'].sum()):.2f}")
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
        st.plotly_chart(fig_pie, use_container_width=True)

    with colbar:
        fig_bar = go.Figure(data=[go.Bar(x=fundhouses, y=returns_list, marker=dict(color="lightblue"), text=returns_list, textposition="inside",
                                         textfont=dict(size=14))])
        fig_bar.update_layout(title=dict(text="Returns (%)", x=0.5, xanchor="center", font=dict(size=22)), xaxis=dict(title="Fund House",tickangle=-10),
                              yaxis_title="Returns (%)", width=650, height=450)
        st.plotly_chart(fig_bar, use_container_width=True)

    dfinv = df.groupby('Date', as_index=False)['Total Investment'].sum()
    dfcur = df.groupby('Date', as_index=False)['Current Value'].sum()
    fig_inv = go.Figure()
    fig_inv.add_trace(go.Scatter(x=dfinv['Date'], y=dfinv['Total Investment'], mode='lines', name='Investment', line=dict(color='blue')))
    fig_inv.add_trace(go.Scatter(x=dfcur['Date'], y=dfcur['Current Value'], mode='lines', name='Performance', line=dict(color='orange')))
    fig_inv.update_layout(title=dict(text="Mutual Fund Investment Journey", x=0.5, xanchor='center', font=dict(size=22)), xaxis_title="Date",
                          yaxis_title="Rupees (₹)", template="plotly_white", xaxis=dict(showgrid=True), yaxis=dict(showgrid=True),
                          width=1350, height=500)
    st.plotly_chart(fig_inv, use_container_width=True)


# ## Salary Portfolio

# In[19]:


def inttostr(x):
    if len(x)==5:
        x = x[:2]+","+x[-3:]
    elif len(x)==6:
        x = x[:1]+","+x[1:3]+","+x[-3:]
    elif len(x)==7:
        x = x[:2]+","+x[2:4]+","+x[-3:]
    return "₹"+x+"/-"


# In[ ]:


if st.session_state.page == "Salary Portfolio":
    st.markdown("""<style>.title {text-align: center; font-size: 34px; font-weight: bold;}</style><div class="title">Salary Portfolio</div>""",
                unsafe_allow_html=True)
    df = pd.read_excel('Salary.xlsx')
    salary_year_list = df['Salary Year'].unique().tolist()
    drpdcol1, drpdcol2, drpdcol3 = st.columns([2,1,2])
    with drpdcol2:
        salary_year = st.selectbox("Salary Year", salary_year_list+["Overall"], index=len(salary_year_list)-1)
    if salary_year=="Overall":
        dff = df.copy()
        dff['Month_Year'] = dff['Month']+" "+dff['Year'].astype(str)
        tf = 10
        ta = -90
        metric_title = " so far"
    else:
        dff = df[df['Salary Year']==salary_year].reset_index(drop=True)
        dff['Month_Year'] = dff['Month']+"<br>"+dff['Year'].astype(str)
        tf = 14
        ta = 0
        metric_title = " this year"
    dff['Amount_str'] = dff['Amount'].astype(str).apply(inttostr)
    tot_salary = str(dff[dff['Income Type']=="Salary"]['Amount'].sum())
    tot_pp = str(dff[dff['Income Type']=="Performance Pay"]['Amount'].sum())
    tot_sprew = str(dff[dff['Income Type']=="Special Reward"]['Amount'].sum())
    experienceyears = str(int(len(df[(df['Income Type']=="Salary")&(df['Amount']>0)])/12))
    experiencemonths = str(int(len(df[(df['Income Type']=="Salary")&(df['Amount']>0)])%12))
    spcol1, spcol2 = st.columns([3.35,0.65])
    with spcol1:
        dfff = dff[dff['Income Type']=="Salary"].reset_index(drop=True)
        if salary_year!="Overall":
            colors = [color_palette[0]]*len(dfff)
        else:
            colors = []
            for i in range(len(salary_year_list)):
                colors += ([color_palette[i]]*len(dfff[dfff['Salary Year']==salary_year_list[i]]))
        fig_bar = go.Figure(data=[go.Bar(x=dfff["Month_Year"], y=dfff["Amount"], marker=dict(color=colors), text=dfff["Amount_str"],
                                         textposition="inside", textfont=dict(size=14))])
        fig_bar.update_layout(title=dict(text="This year Salary (₹)", x=0.5, xanchor="center", font=dict(size=22)), width=650, height=400,
                              xaxis=dict(title="Month",tickangle=ta,tickfont=dict(size=tf)), yaxis=dict(title="Salary (₹)",tickfont=dict(size=15)),
                              margin=dict(l=10,r=30,t=40,b=5))
        st.plotly_chart(fig_bar, use_container_width=True)
    with spcol2:
        st.metric("Overall Experience", " "+experienceyears+" yr "+experiencemonths+" mon")
        st.metric("Total Salary"+metric_title, " "+inttostr(tot_salary))
        st.metric("Performance Pay"+metric_title, " "+inttostr(tot_pp))
        st.metric("Special Rewards"+metric_title, " "+inttostr(tot_sprew))


# # Testing Codes

# In[ ]:




