# backtest/engine.py
import json
import time
from data_fetcher.stock_data import get_stock_kline


class BacktestEngine:
    def __init__(self, cnn_agent, news_agent, risk_agent, meta_agent, commission_rate=0.001):
        """
        :param commission_rate: 单边交易费率（默认 0.1%）
        """
        self.cnn_agent = cnn_agent
        self.news_agent = news_agent
        self.risk_agent = risk_agent
        self.meta_agent = meta_agent
        self.commission_rate = commission_rate

    def run_backtest(self, ticker="000001", period="5", start_date="2023-01-01 00:00:00",
                     end_date="2024-04-01 00:00:00"):
        print(f"\n--- 开始回测 {ticker} [{start_date} 至 {end_date}] ---")

        # 1. 获取 K 线数据
        df = get_stock_kline(ticker, period=period, start_date=start_date, end_date=end_date)

        print(f">>> 成功获取到 K 线数据: {len(df)} 条")
        if len(df) <= 30:
            print("❌ 错误：获取到的数据量不足 30 条（不够均线预热窗口），回测终止！")
            return

        # 2. 初始化账户与状态
        initial_capital = 100000.0  # 初始资金 10 万
        capital = initial_capital
        position = 0  # 0: 空仓, 1: 多头持仓, -1: 空头持仓
        entry_price = 0.0  # 开仓价格
        trade_count = 0  # 统计交易次数

        # 双向移动追踪止盈核心变量
        max_price_since_entry = 0.0  # 多头持仓期间的最高价
        min_price_since_entry = 99999.0  # 空头持仓期间的最低价
        trailing_stop_pct = 0.12  # 从极值回撤/反弹 12% 强制止盈

        # 从第 30 个时间步开始滑动窗口
        for i in range(30, len(df)):
            window_df = df.iloc[i - 30:i]
            current_date = window_df['datetime'].iloc[-1]
            current_price = float(window_df['Close'].iloc[-1])

            # ==========================================
            # 1. 双向 移动追踪止盈 与 硬性止损 检查
            # ==========================================
            stop_loss_triggered = False

            if position == 1:
                # 更新多头最高价
                max_price_since_entry = max(max_price_since_entry, current_price)
                trailing_stop_line = max_price_since_entry * (1 - trailing_stop_pct)

                # 追踪止盈（自最高点回撤 12%）
                if current_price < trailing_stop_line:
                    profit = (current_price - entry_price) / entry_price * capital
                    capital += profit
                    fee = capital * self.commission_rate
                    capital -= fee
                    position = 0
                    trade_count += 1
                    stop_loss_triggered = True
                    print(
                        f"🚨 [多头追踪止盈触发] 价格自最高点 {max_price_since_entry:.2f} 回撤 {trailing_stop_pct * 100:.0f}%！"
                        f"平仓价: {current_price:.2f} | 盈亏: {profit:+.2f} | 交易手续费: {fee:.2f} | 账户总资产: {capital:.2f}")

                # 硬性止损（跌破买入价的 10%）
                elif current_price < entry_price * 0.90:
                    profit = (current_price - entry_price) / entry_price * capital
                    capital += profit
                    fee = capital * self.commission_rate
                    capital -= fee
                    position = 0
                    trade_count += 1
                    stop_loss_triggered = True
                    print(f"🚨 [多头硬性止损触发] 价格跌破买入价 10%！"
                          f"平仓价: {current_price:.2f} | 盈亏: {profit:+.2f} | 交易手续费: {fee:.2f} | 账户总资产: {capital:.2f}")

            elif position == -1:
                # 更新空头最低价
                min_price_since_entry = min(min_price_since_entry, current_price)
                trailing_stop_line = min_price_since_entry * (1 + trailing_stop_pct)

                # 追踪止盈（自最低点反弹 12%）
                if current_price > trailing_stop_line:
                    profit = (entry_price - current_price) / entry_price * capital
                    capital += profit
                    fee = capital * self.commission_rate
                    capital -= fee
                    position = 0
                    trade_count += 1
                    stop_loss_triggered = True
                    print(
                        f"🚨 [空头追踪止盈触发] 价格自最低点 {min_price_since_entry:.2f} 反弹 {trailing_stop_pct * 100:.0f}%！"
                        f"平仓价: {current_price:.2f} | 盈亏: {profit:+.2f} | 交易手续费: {fee:.2f} | 账户总资产: {capital:.2f}")

                # 硬性止损（冲高超过做空价的 10%）
                elif current_price > entry_price * 1.10:
                    profit = (entry_price - current_price) / entry_price * capital
                    capital += profit
                    fee = capital * self.commission_rate
                    capital -= fee
                    position = 0
                    trade_count += 1
                    stop_loss_triggered = True
                    print(f"🚨 [空头硬性止损触发] 价格上涨超过做空价 10%！"
                          f"平仓价: {current_price:.2f} | 盈亏: {profit:+.2f} | 交易手续费: {fee:.2f} | 账户总资产: {capital:.2f}")

            # 若本日已触发止盈止损平仓，直接跳过后续大模型决策，节约 API 额度
            if stop_loss_triggered:
                time.sleep(1)
                continue

            # ==========================================
            # 2. 智能体协同决策环节
            # ==========================================
            cnn_report = self.cnn_agent.analyze(window_df)
            news_report = self.news_agent.analyze(ticker, current_date)
            risk_report = self.risk_agent.analyze(window_df)

            decision_json_str = self.meta_agent.make_decision(cnn_report, news_report, risk_report)

            try:
                # 提取纯 JSON 字典
                clean_str = decision_json_str.replace('```json', '').replace('```', '').strip()
                decision = json.loads(clean_str)
                action = decision.get("action", "WAIT")
                reason = decision.get("reason", "解析失败")
            except Exception as e:
                action = "WAIT"
                reason = f"JSON 解析错误兜底: {e}"

            # ==========================================
            # 3. 交易执行模块 (持有在 WAIT，仅在反转或风控时平仓)
            # ==========================================
            print(
                f"\n[{current_date.strftime('%Y-%m-%d %H:%M:%S')}] 决策信号: {action} | 当前价格: {current_price:.2f} | 理由: {reason}")

            if action == "BUY":
                if position == 1:
                    print(f"  ➔ [继续持股] 当前已全仓持有多头头寸，中线持股待涨中。(持仓买入价: {entry_price:.2f})")
                else:
                    if position == -1:
                        # 翻转：先平空仓
                        profit = (entry_price - current_price) / entry_price * capital
                        capital += profit
                        fee = capital * self.commission_rate
                        capital -= fee
                        print(
                            f"  🔄 [反转平空仓] 结算价: {current_price:.2f} | 盈亏: {profit:+.2f} | 交易手续费: {fee:.2f} | 账户总资产: {capital:.2f}")

                    # 开多仓
                    fee = capital * self.commission_rate
                    capital -= fee
                    entry_price = current_price
                    max_price_since_entry = current_price  # 重置多头最高价追踪器
                    position = 1
                    trade_count += 1
                    print(
                        f"  🟢 [建立多头头寸] 开仓价: {entry_price:.2f} | 交易手续费: {fee:.2f} | 账户总资产(已全仓持股): {capital:.2f}")

            elif action == "SELL":
                if position == -1:
                    print(f"  ➔ [继续持空] 当前已全仓持有空头头寸，中线持空待跌中。(持仓卖出价: {entry_price:.2f})")
                else:
                    if position == 1:
                        # 翻转：先平多仓
                        profit = (current_price - entry_price) / entry_price * capital
                        capital += profit
                        fee = capital * self.commission_rate
                        capital -= fee
                        print(
                            f"  🔄 [反转平多仓] 结算价: {current_price:.2f} | 盈亏: {profit:+.2f} | 交易手续费: {fee:.2f} | 账户总资产: {capital:.2f}")

                    # 开空仓
                    fee = capital * self.commission_rate
                    capital -= fee
                    entry_price = current_price
                    min_price_since_entry = current_price  # 重置空头最低价追踪器
                    position = -1
                    trade_count += 1
                    print(
                        f"  🔴 [建立空头头寸] 开仓价: {entry_price:.2f} | 交易手续费: {fee:.2f} | 账户总资产(已全仓开空): {capital:.2f}")

            else:  # WAIT 信号
                if position == 1:
                    print(
                        f"  ➔ [趋势平稳] 决策官保持观望，我们继续中长线【满仓持股】(持仓买入价: {entry_price:.2f}，当前资产估值: {capital * (current_price / entry_price):.2f})")
                elif position == -1:
                    print(
                        f"  ➔ [趋势平稳] 决策官保持观望，我们继续中长线【持空等待】(持仓卖出价: {entry_price:.2f}，当前资产估值: {capital * (entry_price / current_price):.2f})")
                else:
                    print(f"  ➔ [空仓等待] 当前无趋势信号，保持现金观望状态。(总资产: {capital:.2f})")

            time.sleep(1)

        # 4. 回测结束后强制清仓结算
        last_price = float(df['Close'].iloc[-1])
        if position == 1:
            profit = (last_price - entry_price) / entry_price * capital
            capital += profit
            capital *= (1 - self.commission_rate)
            trade_count += 1
            print(f"\n🏁 [最终强制平仓] 多头以收盘价 {last_price:.2f} 结算 | 盈亏: {profit:+.2f}")
        elif position == -1:
            profit = (entry_price - last_price) / entry_price * capital
            capital += profit
            capital *= (1 - self.commission_rate)
            trade_count += 1
            print(f"\n🏁 [最终强制平仓] 空头以收盘价 {last_price:.2f} 结算 | 盈亏: {profit:+.2f}")

        total_return = (capital - initial_capital) / initial_capital * 100
        print("\n==================================================")
        print(f"📊 回测结束 | 初始资金: {initial_capital:.2f} | 最终资金: {capital:.2f}")
        print(f"📈 累计收益率: {total_return:+.2f}%")
        print(f"🔄 总交易次数 (包含平仓): {trade_count} 次")
        print("==================================================")