# agents/news_rag_agent.py
import os
import hashlib
import pandas as pd
from datetime import timedelta
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


class FundamentalNewsAgent:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("DeepSeek API Key 缺失！")

        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            temperature=0.0,
            max_tokens=200,
            timeout=15
        )

        self.prompt = PromptTemplate.from_template(
            """你是一个资深的宏观与基本面量化分析师。
            请根据以下截至【{current_date}】关于标的【{ticker}】的真实官方披露与新闻事件，
            客观评估该股票的基本面情绪倾向。

            要求：
            1. 明确指出基本面情绪倾向是：【利好】、【利空】还是【震荡/中性】。
            2. 用一句话概括核心事实理由。

            近期披露与事件：
            {news_content}

            请直接输出分析结论："""
        )
        self.chain = self.prompt | self.llm | StrOutputParser()
        self.news_sentiment_cache = {}

    def fetch_historical_news(self, ticker, current_date):
        symbol = ticker.split('.')[-1] if '.' in ticker else ticker

        if not hasattr(self, 'local_news_df') or self.local_news_df is None:
            self.local_news_df = None
            local_file = f"data/{symbol}_news.csv"
            if os.path.exists(local_file):
                try:
                    df = pd.read_csv(local_file)
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    self.local_news_df = df
                    print(f"[News Agent] 挂载 [{symbol}] 动态历史情报库成功。")
                except Exception as e:
                    print(f"[News Agent] 挂载本地库失败: {e}")

        # 检索当前日期前 30 天内发生的真实重大披露
        if self.local_news_df is not None and not self.local_news_df.empty:
            start_date = current_date - timedelta(days=30)
            mask = (self.local_news_df['datetime'] >= start_date) & (self.local_news_df['datetime'] <= current_date)
            day_news = self.local_news_df.loc[mask]

            if not day_news.empty:
                news_list = []
                for _, row in day_news.tail(5).iterrows():
                    headline = row.get('headline', '')
                    summary = row.get('summary', '')
                    date_str = pd.to_datetime(row.get('datetime')).strftime('%Y-%m-%d')
                    news_list.append(f"[{date_str}] {headline} | 摘要: {summary}")
                return "\n".join(news_list)

        return "近期无重大基本面披露与事件"

    def analyze(self, ticker, current_date):
        news_content = self.fetch_historical_news(ticker, current_date)

        if "近期无重大基本面披露与事件" in news_content:
            return f"基本面新闻分析师报告：截至 {current_date.strftime('%Y-%m-%d')}，近期无重大官方披露与事件，基本面平稳视为中性。"

        content_hash = hashlib.md5(news_content.encode('utf-8')).hexdigest()
        if content_hash in self.news_sentiment_cache:
            return f"基本面新闻分析师报告：{self.news_sentiment_cache[content_hash]}"

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