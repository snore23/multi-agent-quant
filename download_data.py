# download_data.py
import os
import requests
import pandas as pd
import akshare as ak


def fetch_from_yahoo_finance(symbol, start_date, end_date):
    """优先通道: Yahoo Finance 官方全球直连接口 (历史数据最全、无截断)"""
    start_ts = int(pd.to_datetime(start_date).timestamp())
    end_ts = int((pd.to_datetime(end_date) + pd.Timedelta(days=1)).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={start_ts}&period2={end_ts}&interval=1d"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code == 200:
            data = res.json()
            result = data.get("chart", {}).get("result", [])
            if result:
                timestamps = result[0].get("timestamp", [])
                indicators = result[0].get("indicators", {})
                quote = indicators.get("quote", [{}])[0]
                adjclose = indicators.get("adjclose", [{}])[0].get("adjclose", quote.get("close", []))

                df = pd.DataFrame({
                    "datetime": pd.to_datetime(timestamps, unit="s"),
                    "Open": quote.get("open", []),
                    "High": quote.get("high", []),
                    "Low": quote.get("low", []),
                    "Close": adjclose if adjclose else quote.get("close", []),
                    "Volume": quote.get("volume", [])
                })
                df.dropna(inplace=True)
                df = df.sort_values(by="datetime").reset_index(drop=True)
                return df[["datetime", "Open", "Close", "High", "Low", "Volume"]]
    except Exception as e:
        print(f"[WARN] Yahoo Finance 接口异常: {e}")
    return None


def fetch_from_eastmoney(ticker, start_date, end_date):
    """备用通道 1: 东方财富美股历史行情"""
    s_date = start_date.replace("-", "")
    e_date = end_date.replace("-", "")
    try:
        df = ak.stock_us_hist(symbol=ticker, period="daily", start_date=s_date, end_date=e_date, adjust="qfq")
        if df is not None and not df.empty:
            df.rename(columns={
                "日期": "datetime",
                "开盘": "Open",
                "收盘": "Close",
                "最高": "High",
                "最低": "Low",
                "成交量": "Volume"
            }, inplace=True)
            df["datetime"] = pd.to_datetime(df["datetime"])
            return df[["datetime", "Open", "Close", "High", "Low", "Volume"]]
    except Exception as e:
        print(f"[WARN] 东方财富接口异常: {e}")
    return None


def fetch_from_sina(symbol, start_date, end_date):
    """备用通道 2: 新浪财经美股前复权接口 (滑动窗口)"""
    try:
        df = ak.stock_us_daily(symbol=symbol, adjust="qfq")
        if df is not None and not df.empty:
            df.rename(columns={
                "date": "datetime",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            }, inplace=True)
            df["datetime"] = pd.to_datetime(df["datetime"])
            mask = (df["datetime"] >= pd.to_datetime(start_date)) & (df["datetime"] <= pd.to_datetime(end_date))
            sliced = df.loc[mask].copy().sort_values(by="datetime").reset_index(drop=True)
            return sliced[["datetime", "Open", "Close", "High", "Low", "Volume"]]
    except Exception as e:
        print(f"[WARN] 新浪财经接口异常: {e}")
    return None


def is_data_complete(df, start_date):
    """校验数据头部是否真正覆盖到了期望的起始日期 (允许 7 天假期公休)"""
    if df is None or df.empty:
        return False
    earliest_dt = df["datetime"].iloc[0]
    expected_dt = pd.to_datetime(start_date)
    return earliest_dt <= expected_dt + pd.Timedelta(days=7)


def download_us_stock_data(ticker="105.META", start_date="2022-01-01", end_date="2023-12-31"):
    symbol = ticker.split(".")[-1] if "." in ticker else ticker
    print("=" * 50)
    print(f"[START] 正在下载美股 [{symbol}] 日线数据 [{start_date} 至 {end_date}]")
    print("=" * 50)

    # 1. 优先通道: Yahoo Finance
    print(f"[INFO] 尝试通道 1: Yahoo Finance 全球直连接口 ({symbol})...")
    df = fetch_from_yahoo_finance(symbol, start_date, end_date)

    # 2. 备用通道 1: 东方财富
    if not is_data_complete(df, start_date):
        print(f"[INFO] 切换至通道 2: 东方财富接口 ({ticker})...")
        df = fetch_from_eastmoney(ticker, start_date, end_date)

    # 3. 备用通道 2: 新浪财经
    if not is_data_complete(df, start_date):
        print(f"[INFO] 切换至通道 3: 新浪财经接口 ({symbol})...")
        df = fetch_from_sina(symbol, start_date, end_date)

    # 4. 保存与结果展示
    if df is not None and not df.empty:
        os.makedirs("data", exist_ok=True)
        filename = f"data/{ticker}_daily.csv"
        df.to_csv(filename, index=False)
        first_date = df["datetime"].iloc[0].strftime("%Y-%m-%d")
        last_date = df["datetime"].iloc[-1].strftime("%Y-%m-%d")
        print(f"[SUCCESS] 数据已保存至 {filename}")
        print(f"[INFO] 数据有效区间: [{first_date} 至 {last_date}]，共 {len(df)} 个交易日！")
    else:
        print(f"[ERROR] 所有通道均无法下载标的 [{symbol}] 的完整行情数据。")


if __name__ == "__main__":
    download_us_stock_data(ticker="105.META", start_date="2022-01-01", end_date="2023-12-31")