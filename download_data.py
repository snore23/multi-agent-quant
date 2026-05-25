# download_data.py
import akshare as ak
import pandas as pd
import os


def download_us_stock_data(ticker="105.NVDA", start_date="20230101", end_date="20240401"):
    print(f"正在通过 AKShare 下载美股 {ticker} 的真实日线数据...")
    try:
        # 使用 akshare 获取美股历史数据（前复权 qfq）
        df = ak.stock_us_hist(symbol=ticker, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")

        # akshare 返回的是中文列名，重命名为 mplfinance 所需的标准英文列名
        df.rename(columns={
            '日期': 'datetime',
            '开盘': 'Open',
            '收盘': 'Close',
            '最高': 'High',
            '最低': 'Low',
            '成交量': 'Volume'
        }, inplace=True)

        # 转换时间格式
        df['datetime'] = pd.to_datetime(df['datetime'])

        # 保存到本地
        os.makedirs("data", exist_ok=True)
        filename = f"data/{ticker}_daily.csv"
        df.to_csv(filename, index=False)
        print(f"✅ 美股数据已成功下载并保存至 {filename}，共 {len(df)} 个交易日！")

    except Exception as e:
        print(f"❌ 下载失败，请检查网络或 akshare 接口: {e}")


if __name__ == "__main__":
    download_us_stock_data()