# agents/trend_temperature_agent.py
import pandas as pd
import numpy as np

class TrendTemperatureAgent:
    def __init__(self, ema_fast=5, ema_mid=10, ema_slow=20, channel_period=20):
        """
        :param ema_fast: 快速指数移动平均线周期
        :param ema_mid: 中速指数移动平均线周期
        :param ema_slow: 慢速指数移动平均线周期
        :param channel_period: 唐奇安通道（自适应创高低）周期
        """
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_slow = ema_slow
        self.channel_period = channel_period

    def analyze(self, df_window):
        """
        基于趋势动物的“趋势温度”思想：
        输入过去时间步的 DataFrame，计算趋势温度评分（0 - 100 分）。
        输出分析报告文本，作为技术面决策依据。
        """
        # 确保传入的窗口长度足够计算均线和通道（我们把窗口改到 30 天）
        if len(df_window) < self.channel_period:
            return "趋势温度分析师报告：当前滑动窗口数据不足，建议观望。"

        close_series = df_window['Close']
        high_series = df_window['High']
        low_series = df_window['Low']
        current_close = float(close_series.iloc[-1])

        # ==========================================
        # 维度一：均线排列度得分 (满分 50 分)
        # ==========================================
        # 计算近期 EMA 均线
        ema_f = close_series.ewm(span=self.ema_fast, adjust=False).mean().iloc[-1]
        ema_m = close_series.ewm(span=self.ema_mid, adjust=False).mean().iloc[-1]
        ema_s = close_series.ewm(span=self.ema_slow, adjust=False).mean().iloc[-1]

        ma_score = 0
        if current_close > ema_f: ma_score += 10
        if current_close > ema_m: ma_score += 10
        if current_close > ema_s: ma_score += 10
        if ema_f > ema_m: ma_score += 10
        if ema_m > ema_s: ma_score += 10

        # ==========================================
        # 维度二：创近期新高强度（唐奇安通道位置，满分 50 分）
        # ==========================================
        recent_high = float(high_series.iloc[-self.channel_period:].max())
        recent_low = float(low_series.iloc[-self.channel_period:].min())

        if recent_high == recent_low:
            channel_score = 25.0
        else:
            # 计算当前收盘价在近期价格通道中的百分比位置
            channel_score = ((current_close - recent_low) / (recent_high - recent_low)) * 50.0

        # ==========================================
        # 汇总：趋势温度评分 (0 - 100)
        # ==========================================
        temp_score = ma_score + channel_score

        # 趋势温度映射：[冻, 寒, 凉, 平, 温, 热, 沸]
        if temp_score >= 70:
            temp_level = f"热/沸 (Hot/Boiling) [分值: {temp_score:.1f}]"
            signal = "看涨 (Buy)"
        elif temp_score <= 30:
            temp_level = f"冻/寒 (Freezing/Cold) [分值: {temp_score:.1f}]"
            signal = "看跌 (Sell)"
        else:
            temp_level = f"温/平/凉 (Neutral) [分值: {temp_score:.1f}]"
            signal = "震荡/观望 (Wait)"

        return (
            f"趋势温度分析师报告：当前标的近期趋势温度评分为 {temp_score:.1f}，"
            f"处于 【{temp_level}】 状态。均线多头排列与通道创高共振，呈现 {signal} 模式。"
        )