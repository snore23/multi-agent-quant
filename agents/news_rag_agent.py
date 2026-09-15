# agents/news_rag_agent.py
import os
import hashlib
import requests
import pandas as pd
from datetime import timedelta
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class FundamentalNewsAgent:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("DeepSeek API Key 缺失！")

        self.finnhub_key = os.getenv("FINNHUB_API_KEY")
        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            temperature=0.0,
            max_tokens=200,
            timeout=15
        )

        self.prompt = PromptTemplate.from_template(
            """你是一个资深的金融宏观与基本面分析师。
            请根据以下过去几天关于标的【{ticker}】的真实历史新闻，
            判断截至【{current_date}】时该股票的基本面情绪倾向。

            要求：
            1. 明确指出基本面情绪是：【利好】、【利空】还是【震荡/中性】。
            2. 用一句话概括理由。

            近期新闻：
            {news_content}

            请直接输出分析结论："""
        )
        self.chain = self.prompt | self.llm | StrOutputParser()
        self.news_sentiment_cache = {}

    def fetch_historical_news(self, ticker, current_date):
        symbol = ticker.split('.')[-1] if '.' in ticker else ticker

        if not hasattr(self, 'local_news_df'):
            self.local_news_df = None
            local_file = f"data/{symbol}_news.csv"
            if os.path.exists(local_file):
                try:
                    df = pd.read_csv(local_file)
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    self.local_news_df = df
                    print(f"[News Agent] 成功挂载 [{symbol}] 本地历史新闻库。")
                except Exception as e:
                    print(f"[News Agent] 挂载本地 CSV 失败: {e}")

        if self.local_news_df is not None:
            start_date = current_date - timedelta(days=30)
            mask = (self.local_news_df['datetime'] >= start_date) & (self.local_news_df['datetime'] <= current_date)
            day_news = self.local_news_df.loc[mask]

            if not day_news.empty:
                news_list = []
                for _, row in day_news.head(5).iterrows():
                    headline = row.get('headline', '')
                    summary = row.get('summary', '')
                    news_list.append(f"- 标题: {headline} | 摘要: {summary}")
                return "\n".join(news_list)

        if not self.finnhub_key:
            return "近期无有效新闻"

        try:
            end_date_str = current_date.strftime('%Y-%m-%d')
            start_date_str = (current_date - timedelta(days=3)).strftime('%Y-%m-%d')
            url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={start_date_str}&to={end_date_str}&token={self.finnhub_key}"
            response = requests.get(url, timeout=10)
            data = response.json()

            if not data or len(data) == 0:
                return "近期无有效新闻"

            news_list = []
            for item in data[:5]:
                news_list.append(f"- 标题: {item.get('headline', '')} | 摘要: {item.get('summary', '')}")
            return "\n".join(news_list)

        except Exception:
            return "近期无有效新闻"

    def analyze(self, ticker, current_date):
        news_content = self.fetch_historical_news(ticker, current_date)

        if "近期无有效新闻" in news_content:
            return f"基本面新闻分析师报告：截至 {current_date.strftime('%Y-%m-%d')}，近期无有效基本面催化剂，视为中性。"

        # 计算新闻内容哈希，若内容未发生变化直接复用之前的大模型分析结果
        content_hash = hashlib.md5(news_content.encode('utf-8')).hexdigest()
        if content_hash in self.news_sentiment_cache:
            cached_report = self.news_sentiment_cache[content_hash]
            return f"基本面新闻分析师报告：{cached_report}"

        try:
            report = self.chain.invoke({
                "ticker": ticker,
                "current_date": current_date.strftime('%Y-%m-%d'),
                "news_content": news_content
            })
            self.news_sentiment_cache[content_hash] = report
            return f"基本面新闻分析师报告：{report}"
        except Exception as e:
            return f"基本面新闻分析师报告：大模型分析异常，默认视为中性。({e})"