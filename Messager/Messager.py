import pandas as pd
import numpy as np
import yfinance as yf
import smtplib
import warnings
from io import BytesIO
from datetime import datetime
import plotly.graph_objects as go
warnings.filterwarnings('ignore')
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage


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


def fetch_current_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        return stock.history(period="1d")["Close"].iloc[-1]
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None

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


def piechart(dff):
    fig = go.Figure(data=[go.Pie(labels=dff["Stock Name"], values=dff["Total Investment"], hole=0.5, textfont=dict(size=14))])
    fig.update_layout(title=dict(text="Current Holdings", x=0.5, xanchor="center", font=dict(size=22)),
                      legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"), showlegend=True, width=550, height=450)
    return fig


def stockperformance(dff, stocklist):
    string = ""
    for stock in stocklist:
        string += stock+": "
        minindex = dff[dff['Stock Name']==stock].index.max()-1
        maxindex = dff[dff['Stock Name']==stock].index.max()
        returns = np.round(dff['Net Value'][maxindex]-dff['Net Value'][minindex],2)
        returnsperc = np.round((dff['Net Value'][maxindex]-dff['Net Value'][minindex])/dff['Net Value'][minindex],1)
        string += f"+ ₹{str(returns)} (+{returnsperc}%)<br>" if returns>0 else f"- ₹{str(abs(returns))} (-{returnsperc}%)<br>"
    return string


def send_email_with_chart(sender_email, recipient_email, subject, body_text, app_password, dataframe1, dataframe2, mystocks):
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    stockprfm = stockperformance(dataframe2, mystocks)
    
    buf = BytesIO()
    fig_pie = piechart(dataframe1)
    fig_pie.write_image(buf, format="png")
    buf.seek(0)
    
    html_body = f"""<html><body><h1 style="font-size:20px;">{body_text}</h1><h3>{stockprfm}</h3><img src="cid:pie_chart" alt="Pie Chart"></body>
                    </html>"""
    msg.attach(MIMEText(html_body, 'html'))
    img = MIMEImage(buf.getvalue())
    img.add_header('Content-ID', '<pie_chart>')
    msg.attach(img)
    
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, app_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
            print(f"Email sent to {recipient_email} successfully.")
    except Exception as e:
        print(f"Error: {e}")


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


portfolio_performance = np.round(investment_journey_df['Net Value'][len(investment_journey_df)-1
                                 ]-investment_journey_df['Net Value'][len(investment_journey_df)-2],2)

sender_email = "eswaraprasath.m@gmail.com"
recipient_email = "muruganeswaraprasath@gmail.com"
subject = "Stock Portfolio performance"
body = f"Your Stock Portfolio performance for {datetime.today().strftime("%d-%b-%Y")} : ₹{portfolio_performance}/-"
app_password = "dwjs oben nmnp zfas"

send_email_with_chart(sender_email, recipient_email, subject, body, app_password, holdings, stockdffull, mystocks)
