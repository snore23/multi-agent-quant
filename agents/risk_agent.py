# agents/risk_agent.py

class RiskControlAgent:
    def __init__(self, max_drawdown=0.15):
        """
        :param max_drawdown: 允许的最大回撤比例。默认 15%（破位判定）
        """
        self.max_drawdown = max_drawdown

    def analyze(self, df_window):
        # 计算当前收盘价距离 10 日内最高价的回撤幅度
        peak_price = df_window['High'].max()
        current_price = df_window['Close'].iloc[-1]
        drawdown = (peak_price - current_price) / peak_price

        if drawdown > self.max_drawdown:
            return (
                f"强制风控指令：当前价格距离10日高点回撤达到 {drawdown:.2%}，"
                f"超过安全阈值 {self.max_drawdown:.2%}，判定为技术面破位下行，强行要求空仓避险。"
            )

        return f"风控允许：当前最大回撤为 {drawdown:.2%}，处于安全阈值内，允许正常交易。"