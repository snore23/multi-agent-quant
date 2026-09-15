class RiskControlAgent:
    def __init__(self, max_drawdown=0.12, max_rebound=0.12):
        self.max_drawdown = max_drawdown
        self.max_rebound = max_rebound

    def analyze(self, df_window, position=0, entry_price=0.0):
        current_price = float(df_window['Close'].iloc[-1])
        recent_high = float(df_window['High'].max())
        recent_low = float(df_window['Low'].min())

        long_drawdown = (recent_high - current_price) / recent_high
        short_rebound = (current_price - recent_low) / recent_low

        if position == 1:
            if long_drawdown > self.max_drawdown:
                return f"[RISK ALERT] 多头持仓下价格自近期高点回撤 {long_drawdown:.2%}，超过阈值 {self.max_drawdown:.2%}，强制平仓避险。"
            if entry_price > 0 and current_price < entry_price * 0.90:
                return f"[STOP LOSS] 多头持仓亏损达 10%，触发硬性止损指令。"

        elif position == -1:
            if short_rebound > self.max_rebound:
                return f"[RISK ALERT] 空头持仓下价格自近期低点反弹 {short_rebound:.2%}，超过阈值 {self.max_rebound:.2%}，强制平仓避险。"
            if entry_price > 0 and current_price > entry_price * 1.10:
                return f"[STOP LOSS] 空头持仓亏损达 10%，触发硬性止损指令。"

        return f"[RISK OK] 波动处于安全阈值内 (多头回撤: {long_drawdown:.2%}, 空头反弹: {short_rebound:.2%})，允许正常交易。"