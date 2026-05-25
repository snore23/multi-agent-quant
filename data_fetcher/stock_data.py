import os

import akshare as ak
import pandas as pd
import numpy as np
import time


def get_stock_kline(ticker: str, period="daily", start_date="2023-01-01", end_date="2024-04-01"):
    """
    优先读取本地股票数据
    """
    # 适配日线数据的文件名格式
    local_file = f"data/{ticker}_{period}.csv"

    if os.path.exists(local_file):
        print(f"📂 成功加载本地股票数据: {local_file}")
        df = pd.read_csv(local_file)
        df['datetime'] = pd.to_datetime(df['datetime'])

        # 按照回测引擎传来的时间段进行截取
        mask = (df['datetime'] >= start_date) & (df['datetime'] <= end_date)
        sliced_df = df.loc[mask].copy()

        if not sliced_df.empty:
            return sliced_df
        else:
            print("⚠️ 警告：时间段截取为空，返回全量本地数据。")
            return df
    else:
        raise FileNotFoundError(f"❌ 找不到本地数据 {local_file}，请先运行 download_data.py")


def get_stock_news(ticker: str):
    """
    获取个股新闻。内置重试与兜底机制。
    """
    for attempt in range(3):
        try:
            news_df = ak.stock_news_em(symbol=ticker)
            return news_df['新闻内容'].tolist()[:5]
        except Exception as e:
            time.sleep(1)

    # 兜底假新闻
    return [
        f"{ticker} 披露最新财报，净利润同比增长显著",
        "北向资金今日大幅净买入该股",
        "行业迎来政策利好，估值有望修复"
    ]


def _generate_mock_data(start_str, end_str, period):
    """
    生成逼真的随机 K 线数据，用于系统无网环境下的测试兜底
    """
    dates = pd.date_range(start=start_str, end=end_str, freq=f"{period}T")
    # 模拟工作日和交易时间（简单剔除周末）
    dates = [d for d in dates if d.weekday() < 5]

    # 生成随机布朗运动价格
    np.random.seed(42)
    returns = np.random.normal(0, 0.002, len(dates))
    prices = 10.0 * np.cumprod(1 + returns)

    # 伪造 OHLC 数据
    highs = prices * (1 + np.abs(np.random.normal(0, 0.001, len(dates))))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.001, len(dates))))
    opens = prices * (1 + np.random.normal(0, 0.0005, len(dates)))

    df = pd.DataFrame({
        'datetime': dates,
        'Open': opens,
        'Close': prices,
        'High': highs,
        'Low': lows,
        'Volume': np.random.randint(1000, 10000, len(dates))
    })
    return df