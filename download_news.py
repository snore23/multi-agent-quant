# download_news.py
import os
import time
import random
import datetime
import urllib.parse
import requests
from bs4 import BeautifulSoup
import pandas as pd


def download_news_from_baidu(ticker="105.NVDA", start_year=2023, end_year=2024):
    """
    通过百度资讯获取任意股票的历史真实新闻。
    将时间明文作为核心搜索词，并进行全局标题去重，彻底解决数据重复问题。
    """
    symbol = ticker.split('.')[-1] if '.' in ticker else ticker
    print(f"⏳ 开始获取 [{symbol}] 的真实历史新闻 ({start_year} - {end_year})...")

    news_records = []

    # 建立一个全局去重集合，确保整张 CSV 里的新闻headline绝不重复
    seen_headlines = set()

    months = [
        (1, "1月"), (2, "2月"), (3, "3月"), (4, "4月"), (5, "5月"), (6, "6月"),
        (7, "7月"), (8, "8月"), (9, "9月"), (10, "10月"), (11, "11月"), (12, "12月")
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Referer": "https://news.baidu.com/",
        "Connection": "keep-alive"
    }

    session = requests.Session()

    for year in range(start_year, end_year + 1):
        for month_num, month_cn in months:
            if year == end_year and month_num > 4:
                break

            # --- 核心改进：把时间戳作为硬性搜索条件明文写入查询词（wd），用双引号强制精确匹配 ---
            # 例如: NVDA "2023年1月"
            query = f'{symbol} "{year}年{month_num}月"'
            encoded_query = urllib.parse.quote(query)

            url = f"https://www.baidu.com/s?tn=news&rtt=4&bsst=1&cl=2&wd={encoded_query}"

            print(f"🔍 正在检索: {query} ...")

            try:
                response = session.get(url, headers=headers, timeout=10)

                if response.status_code != 200:
                    print(f"  ⚠️ 百度接口异常，状态码: {response.status_code}")
                    continue

                if "安全验证" in response.text or "网络状况不佳" in response.text:
                    print("  ⚠️ 触发了百度的反爬验证码，本次检索跳过。")
                    continue

                soup = BeautifulSoup(response.text, 'html.parser')

                content_left = soup.find('div', id='content_left')
                if not content_left:
                    print("  ⚠️ 未能定位到新闻内容区。")
                    continue

                news_items = content_left.find_all('div', class_=lambda x: x and ('result-op' in x or 'result' in x))

                month_records_count = 0
                simulated_date = f"{year}-{month_num:02d}-15"

                for item in news_items:
                    a_tag = (
                            item.find('a', class_=lambda x: x and 'title' in x.lower())
                            or (item.find('h3').find('a') if item.find('h3') else None)
                            or item.find('a')
                    )
                    if not a_tag:
                        continue

                    headline = a_tag.get_text().strip()

                    # 过滤导航条干扰
                    ignore_keywords = ["百度", "登录", "注册", "百度一下", "百度首页", "百度热搜", "安全验证",
                                       "百度资讯"]
                    if any(kw in headline for kw in ignore_keywords):
                        continue

                    # --- 核心改进：全局标题去重，防止同月转载新闻和跨月重复推荐的新闻被记录 ---
                    if headline in seen_headlines:
                        continue

                    summary_tag = (
                            item.find('span', class_=lambda x: x and 'c-color-text' in x)
                            or item.find('div', class_=lambda x: x and 'c-span-last' in x)
                    )
                    summary = summary_tag.get_text().strip() if summary_tag else ""

                    if not summary:
                        summary = item.get_text().replace(headline, "").strip()[:150]

                    summary = summary.replace("\xa0", " ").strip()

                    # 时间穿梭校验
                    has_lookahead_bias = False
                    for future_y in range(year + 1, 2027):
                        if str(future_y) in headline or str(future_y) in summary:
                            has_lookahead_bias = True
                            break

                    if has_lookahead_bias:
                        continue

                    if symbol.lower() in headline.lower() or symbol.lower() in summary.lower():
                        # 通过所有校验，写入去重集合
                        seen_headlines.add(headline)

                        news_records.append({
                            "datetime": simulated_date,
                            "headline": headline,
                            "summary": summary
                        })
                        month_records_count += 1

                        # 每个月保留 2~3 条高价值新闻已足够 RAG 运作
                        if month_records_count >= 3:
                            break

                print(f"  📊 成功获取并过滤该月真实新闻 {month_records_count} 条。")
                time.sleep(random.uniform(2.5, 4.5))

            except Exception as e:
                print(f"  ⚠️ 检索发生异常: {e}，跳过该月...")
                time.sleep(3)
                continue

    if not news_records:
        print("\n❌ 未提取到有效新闻。")
        return

    # 导出
    df = pd.DataFrame(news_records)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(by='datetime')

    os.makedirs("data", exist_ok=True)
    save_path = f"data/{symbol}_news.csv"
    df.to_csv(save_path, index=False)
    print(f"\n✅ 成功使用【明文过滤 + 全局去重】重新下载 [{symbol}] 的历史新闻！共 {len(df)} 条报道已保存至: {save_path}")


if __name__ == "__main__":
    download_news_from_baidu(ticker="105.NVDA", start_year=2023, end_year=2024)