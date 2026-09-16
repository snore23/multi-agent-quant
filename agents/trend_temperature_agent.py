# agents/trend_temperature_agent.py
import pandas as pd
import numpy as np


class TrendTemperatureAgent:
    def __init__(self, ema_fast=10, ema_mid=20, ema_slow=40, channel_period=20):
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_slow = ema_slow
        self.channel_period = channel_period

    def analyze(self, df_window):
        # 兼容当前窗口长度
        if len(df_window) < self.channel_period:
            return "趋势温度报告：预热数据不足，建议观望。"

        close_series = df_window['Close']
        high_series = df_window['High']
        low_series = df_window['Low']
        current_close = float(close_series.iloc[-1])

        # 1. 均线多头排列得分 (满分 50 分)
        ema_f = close_series.ewm(span=self.ema_fast, adjust=False).mean().iloc[-1]
        ema_m = close_series.ewm(span=self.ema_mid, adjust=False).mean().iloc[-1]
        ema_s = close_series.ewm(span=self.ema_slow, adjust=False).mean().iloc[-1]

        ma_score = 0
        if current_close > ema_f: ma_score += 10
        if current_close > ema_m: ma_score += 10
        if current_close > ema_s: ma_score += 10
        if ema_f > ema_m: ma_score += 10
        if ema_m > ema_s: ma_score += 10

        # 2. 通道强度得分 (满分 50 分)
        recent_high = float(high_series.iloc[-self.channel_period:].max())
        recent_low = float(low_series.iloc[-self.channel_period:].min())

        if recent_high == recent_low:
            channel_score = 25.0
        else:
            channel_score = ((current_close - recent_low) / (recent_high - recent_low)) * 50.0

        temp_score = ma_score + channel_score
        is_above_slow_ma = current_close > ema_s

        # 3. 输出细分的市场形态
        if temp_score >= 70:
            pattern = "强烈看涨 (BULLISH)"
            desc = "高热上升通道，均线与通道共振创新高"
        elif temp_score >= 45 and is_above_slow_ma:
            pattern = "多头良性回调 (BULLISH_HOLD)"
            desc = "处于多头中继支撑区，主趋势未破，建议持仓者拿住、空仓者观望"
        elif temp_score <= 25 and not is_above_slow_ma:
            pattern = "破位看跌 (BEARISH)"
            desc = "跌破长期均线支撑，空头主导"
        else:
            pattern = "中性震荡 (NEUTRAL)"
            desc = "方向不明或宽幅震荡，建议防御"

        return f"趋势温度报告：当前评分 {temp_score:.1f}，均线长期支撑处于{'上方' if is_above_slow_ma else '下方'}。形态判定为【{pattern}】，理由：{desc}。"