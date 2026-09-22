# backtest/engine.py
import json
import numpy as np
import pandas as pd
from data_fetcher.stock_data import get_stock_kline


class BacktestEngine:
    def __init__(self, tech_agent=None, news_agent=None, risk_agent=None, meta_agent=None,
                 commission_rate=0.001, slippage=0.0005):
        self.tech_agent = tech_agent
        self.news_agent = news_agent
        self.risk_agent = risk_agent
        self.meta_agent = meta_agent
        self.commission_rate = commission_rate
        self.slippage = slippage

    def run_backtest(self, ticker="105.NVDA", period="daily", start_date="2023-01-01", end_date="2024-04-01"):
        print(f"\n[INFO] 开始回测标的: {ticker} [{start_date} 至 {end_date}]")
        df = get_stock_kline(ticker, period=period, start_date=start_date, end_date=end_date)
        df.reset_index(drop=True, inplace=True)

        lookback = 60
        total_steps = len(df)
        if total_steps <= lookback:
            print("[ERROR] 数据量不足以支持 60 天预热窗口。")
            return

        initial_capital = 100000.0
        capital = initial_capital
        position = 0  # 1: 多头, -1: 空头, 0: 空仓
        entry_price = 0.0
        max_price_since_entry = 0.0
        min_price_since_entry = 999999.0
        trailing_stop_pct = 0.12

        trade_records = []
        equity_curve = [initial_capital] * lookback

        for i in range(lookback, total_steps):
            window_df = df.iloc[i - lookback:i]
            today_bar = df.iloc[i]
            current_date = today_bar['datetime']
            close_price = float(today_bar['Close'])
            step_progress = f"[{i - lookback + 1}/{total_steps - lookback}]"

            # 1. 检查持仓的主动硬止损与移动追踪止盈
            stop_triggered = False
            if position == 1:
                max_price_since_entry = max(max_price_since_entry, close_price)
                trailing_line = max_price_since_entry * (1 - trailing_stop_pct)
                hard_stop_line = entry_price * 0.90

                if close_price < trailing_line or close_price < hard_stop_line:
                    exec_price = close_price * (1 - self.slippage)
                    pnl = (exec_price - entry_price) / entry_price * capital
                    fee = capital * self.commission_rate
                    capital += (pnl - fee)
                    reason = "多头移动止盈" if close_price < trailing_line else "多头硬性止损"
                    trade_records.append({
                        "type": "CLOSE_LONG", "price": exec_price, "pnl": pnl, 
                        "fee": fee, "date": current_date, "reason": reason
                    })
                    print(f"{step_progress} {current_date.strftime('%Y-%m-%d')} | [止盈/止损] {reason} | 价格: {exec_price:.2f} | 盈亏: {pnl:+.2f} | 资产: {capital:.2f}")
                    position = 0
                    entry_price = 0.0
                    stop_triggered = True

            elif position == -1:
                min_price_since_entry = min(min_price_since_entry, close_price)
                trailing_line = min_price_since_entry * (1 + trailing_stop_pct)
                hard_stop_line = entry_price * 1.10

                if close_price > trailing_line or close_price > hard_stop_line:
                    exec_price = close_price * (1 + self.slippage)
                    pnl = (entry_price - exec_price) / entry_price * capital
                    fee = capital * self.commission_rate
                    capital += (pnl - fee)
                    reason = "空头移动止盈" if close_price > trailing_line else "空头硬性止损"
                    trade_records.append({
                        "type": "CLOSE_SHORT", "price": exec_price, "pnl": pnl, 
                        "fee": fee, "date": current_date, "reason": reason
                    })
                    print(f"{step_progress} {current_date.strftime('%Y-%m-%d')} | [止盈/止损] {reason} | 价格: {exec_price:.2f} | 盈亏: {pnl:+.2f} | 资产: {capital:.2f}")
                    position = 0
                    entry_price = 0.0
                    stop_triggered = True

            if stop_triggered:
                equity_curve.append(capital)
                continue

            # 2. 运行多智能体研判 (修复风控参数传递)
            tech_report = self.tech_agent.analyze(window_df) if self.tech_agent else "技术面缺失"
            news_report = self.news_agent.analyze(ticker, current_date) if self.news_agent else "基本面缺失"
            risk_report = self.risk_agent.analyze(window_df, position=position, entry_price=entry_price) if self.risk_agent else "[RISK OK]"

            decision_json_str = self.meta_agent.make_decision(tech_report, news_report, risk_report)

            try:
                decision = json.loads(decision_json_str)
                stance = decision.get("stance", "NEUTRAL")
                reason = decision.get("reason", "无")
            except Exception:
                stance = "NEUTRAL"
                reason = "解析兜底"

            # 3. 状态机决策映射
            if stance == "BULLISH":
                target_position = 1
            elif stance == "BULLISH_HOLD":
                target_position = 1 if position == 1 else 0
            elif stance == "BEARISH":
                target_position = -1
            else:
                target_position = 0

            # 4. 执行调仓
            if target_position != position:
                # 平旧仓
                if position == 1:
                    exec_price = close_price * (1 - self.slippage)
                    pnl = (exec_price - entry_price) / entry_price * capital
                    fee = capital * self.commission_rate
                    capital += (pnl - fee)
                    trade_records.append({"type": "CLOSE_LONG", "price": exec_price, "pnl": pnl, "fee": fee, "date": current_date, "reason": f"立场转变 -> {stance}"})
                    print(f"{step_progress} {current_date.strftime('%Y-%m-%d')} | [多头平仓] 价格: {exec_price:.2f} | 盈亏: {pnl:+.2f} | 净资产: {capital:.2f}")
                    position = 0

                elif position == -1:
                    exec_price = close_price * (1 + self.slippage)
                    pnl = (entry_price - exec_price) / entry_price * capital
                    fee = capital * self.commission_rate
                    capital += (pnl - fee)
                    trade_records.append({"type": "CLOSE_SHORT", "price": exec_price, "pnl": pnl, "fee": fee, "date": current_date, "reason": f"立场转变 -> {stance}"})
                    print(f"{step_progress} {current_date.strftime('%Y-%m-%d')} | [空头平仓] 价格: {exec_price:.2f} | 盈亏: {pnl:+.2f} | 净资产: {capital:.2f}")
                    position = 0

                # 开新仓
                if target_position == 1:
                    exec_price = close_price * (1 + self.slippage)
                    fee = capital * self.commission_rate
                    capital -= fee
                    entry_price = exec_price
                    max_price_since_entry = exec_price
                    position = 1
                    trade_records.append({"type": "OPEN_LONG", "price": exec_price, "fee": fee, "date": current_date, "reason": reason})
                    print(f"{step_progress} {current_date.strftime('%Y-%m-%d')} | [建立多头] 价格: {exec_price:.2f} | 理由: {reason}")

                elif target_position == -1:
                    exec_price = close_price * (1 - self.slippage)
                    fee = capital * self.commission_rate
                    capital -= fee
                    entry_price = exec_price
                    min_price_since_entry = exec_price
                    position = -1
                    trade_records.append({"type": "OPEN_SHORT", "price": exec_price, "fee": fee, "date": current_date, "reason": reason})
                    print(f"{step_progress} {current_date.strftime('%Y-%m-%d')} | [建立空头] 价格: {exec_price:.2f} | 理由: {reason}")
            else:
                pos_str = "多头持仓中" if position == 1 else "空头持仓中" if position == -1 else "空仓观望"
                print(f"{step_progress} {current_date.strftime('%Y-%m-%d')} | [立场: {stance}] 状态: {pos_str} | 收盘价: {close_price:.2f}")

            # 记录当前净值
            if position == 1:
                cur_val = capital * (close_price / entry_price)
            elif position == -1:
                cur_val = capital * (1 + (entry_price - close_price) / entry_price)
            else:
                cur_val = capital
            equity_curve.append(cur_val)

        # 回测结束平仓与净值同步
        if position != 0:
            final_price = float(df['Close'].iloc[-1])
            pnl = (final_price - entry_price) / entry_price * capital if position == 1 else (entry_price - final_price) / entry_price * capital
            capital += pnl * (1 - self.commission_rate)
            equity_curve[-1] = capital

        self._print_summary(equity_curve, initial_capital, trade_records, df, lookback)

    def _print_summary(self, equity_curve, initial_capital, trade_records, df, lookback):
        equity = pd.Series(equity_curve[lookback:])
        returns = equity.pct_change().dropna()

        total_return = (equity.iloc[-1] - initial_capital) / initial_capital
        benchmark_return = (df['Close'].iloc[-1] - df['Close'].iloc[lookback]) / df['Close'].iloc[lookback]
        max_dd = ((equity.cummax() - equity) / equity.cummax()).max()

        # 计算年化与夏普比率 (按美股 252 交易日计算)
        trading_days = len(equity)
        cagr = (equity.iloc[-1] / initial_capital) ** (252.0 / max(trading_days, 1)) - 1
        sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252) if len(returns) > 1 else 0.0

        winning = [t for t in trade_records if t.get("pnl", 0) > 0]
        losing = [t for t in trade_records if t.get("pnl", 0) < 0]
        win_rate = len(winning) / max(len(winning) + len(losing), 1)

        total_profit = sum(t.get("pnl", 0) for t in winning)
        total_loss = abs(sum(t.get("pnl", 0) for t in losing))
        profit_factor = total_profit / total_loss if total_loss > 0 else (total_profit if total_profit > 0 else 0.0)

        print("\n" + "=" * 55)
        print("                 量化回测绩效综合评估报告                 ")
        print("=" * 55)
        print(f"初始资产:               {initial_capital:>15,.2f}")
        print(f"期末总资产:             {equity.iloc[-1]:>15,.2f}")
        print(f"策略累计收益率:         {total_return:>15.2%}")
        print(f"基准收益率 (买入持有):   {benchmark_return:>15.2%}")
        print(f"年化复合收益率 (CAGR):   {cagr:>15.2%}")
        print(f"夏普比率 (Sharpe Ratio): {sharpe:>15.2f}")
        print(f"最大回撤 (Max Drawdown): {max_dd:>15.2%}")
        print(f"盈亏比 (Profit Factor):  {profit_factor:>15.2f}")
        print(f"平仓交易总笔数:         {len(winning) + len(losing):>15} 笔")
        print(f"胜率 (Win Rate):        {win_rate:>15.2%}")
        print("=" * 55)