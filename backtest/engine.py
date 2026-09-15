# backtest/engine.py
import json
import pandas as pd
from data_fetcher.stock_data import get_stock_kline


class BacktestEngine:
    def __init__(self, tech_agent=None, news_agent=None, risk_agent=None, meta_agent=None,
                 cnn_agent=None, commission_rate=0.001, slippage=0.0005):
        self.tech_agent = tech_agent or cnn_agent
        self.news_agent = news_agent
        self.risk_agent = risk_agent
        self.meta_agent = meta_agent
        self.commission_rate = commission_rate
        self.slippage = slippage

    def run_backtest(self, ticker="105.NVDA", period="daily", start_date="2023-01-01", end_date="2024-04-01"):
        print(f"\n[INFO] 开始回测标的: {ticker} [{start_date} 至 {end_date}]")
        df = get_stock_kline(ticker, period=period, start_date=start_date, end_date=end_date)
        df.reset_index(drop=True, inplace=True)

        total_steps = len(df)
        if total_steps < 40:
            print("[ERROR] 数据量不足，无法运行。")
            return

        initial_capital = 100000.0
        capital = initial_capital
        position = 0  # 0: 现金, 1: 多头, -1: 空头
        entry_price = 0.0

        max_price_since_entry = 0.0
        min_price_since_entry = 99999.0
        trailing_stop_pct = 0.12

        trade_records = []
        equity_curve = [initial_capital] * 30

        print(f"[INFO] 历史数据加载完成，有效交易日共 {total_steps} 天，开始滑动回测...\n")

        for i in range(30, total_steps):
            window_df = df.iloc[i - 30:i]
            today_bar = df.iloc[i]
            current_date = today_bar['datetime']
            close_price = float(today_bar['Close'])
            step_progress = f"[{i - 29}/{total_steps - 30}]"

            # 1. 硬性止损与移动追踪止盈检查
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
                    trade_records.append({"type": "CLOSE_LONG", "price": exec_price, "pnl": pnl, "fee": fee, "date": current_date, "reason": reason})
                    print(f"{step_progress} {current_date.strftime('%Y-%m-%d')} | [平仓触发] {reason} | 价格: {exec_price:.2f} | 盈亏: {pnl:+.2f} | 总资产: {capital:.2f}")
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
                    trade_records.append({"type": "CLOSE_SHORT", "price": exec_price, "pnl": pnl, "fee": fee, "date": current_date, "reason": reason})
                    print(f"{step_progress} {current_date.strftime('%Y-%m-%d')} | [平仓触发] {reason} | 价格: {exec_price:.2f} | 盈亏: {pnl:+.2f} | 总资产: {capital:.2f}")
                    position = 0
                    entry_price = 0.0
                    stop_triggered = True

            if stop_triggered:
                equity_curve.append(capital)
                continue

            # 2. 智能体分析与决策
            tech_report = self.tech_agent.analyze(window_df)
            news_report = self.news_agent.analyze(ticker, current_date)
            risk_report = self.risk_agent.analyze(window_df)

            decision_json_str = self.meta_agent.make_decision(tech_report, news_report, risk_report)

            try:
                decision = json.loads(decision_json_str)
                stance = decision.get("stance", "NEUTRAL")
                reason = decision.get("reason", "无")
            except Exception:
                stance = "NEUTRAL"
                reason = "解析兜底"

            target_position = 1 if stance == "BULLISH" else -1 if stance == "BEARISH" else 0

            # 3. 状态机执行
            if target_position != position:
                # 平掉旧仓
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
                # 仓位保持不变时打印简要状态
                pos_str = "多头持仓中" if position == 1 else "空头持仓中" if position == -1 else "空仓观望"
                print(f"{step_progress} {current_date.strftime('%Y-%m-%d')} | [维持立场: {stance}] 当前状态: {pos_str} | 收盘价: {close_price:.2f}")

            # 动态资产估值
            if position == 1:
                cur_val = capital * (close_price / entry_price)
            elif position == -1:
                cur_val = capital * (1 + (entry_price - close_price) / entry_price)
            else:
                cur_val = capital
            equity_curve.append(cur_val)

        # 回测结束强制清算
        if position != 0:
            final_price = float(df['Close'].iloc[-1])
            pnl = (final_price - entry_price) / entry_price * capital if position == 1 else (entry_price - final_price) / entry_price * capital
            capital += pnl * (1 - self.commission_rate)

        self._print_summary(equity_curve, initial_capital, trade_records, df)

    def _print_summary(self, equity_curve, initial_capital, trade_records, df):
        equity = pd.Series(equity_curve)
        total_return = (equity.iloc[-1] - initial_capital) / initial_capital
        benchmark_return = (df['Close'].iloc[-1] - df['Close'].iloc[30]) / df['Close'].iloc[30]
        max_dd = ((equity.cummax() - equity) / equity.cummax()).max()

        winning = [t for t in trade_records if t.get("pnl", 0) > 0]
        losing = [t for t in trade_records if t.get("pnl", 0) < 0]
        win_rate = len(winning) / max(len(winning) + len(losing), 1)

        print("\n" + "=" * 50)
        print("                 回测绩效评估报告                 ")
        print("=" * 50)
        print(f"初始本金:             {initial_capital:,.2f}")
        print(f"期末总资产:           {equity.iloc[-1]:,.2f}")
        print(f"策略累计收益率:       {total_return:+.2%}")
        print(f"基准收益率 (买入持有): {benchmark_return:+.2%}")
        print(f"最大回撤 (Max DD):    {max_dd:.2%}")
        print(f"交易总笔数:           {len(trade_records)} 笔")
        print(f"平仓胜率 (Win Rate):  {win_rate:.2%}")
        print("=" * 50)