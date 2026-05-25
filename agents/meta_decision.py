import json
from openai import Client


class MetaDecisionAgent:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("API Key 缺失！请检查 .env 文件是否正确配置。")

        # 初始化大模型客户端 (使用 OpenAI SDK 调用 DeepSeek 接口)
        self.client = Client(api_key=api_key, base_url="https://api.deepseek.com/v1",timeout=15.0)

    def make_decision(self, cnn_report, news_report, risk_report):
        """
        首席决策官逻辑：综合评估，输出最终动作。
        """
        prompt = f"""
        你是量化交易系统的首席决策官 (Meta-Agent)，请根据以下三份下属报告做出最终的交易决策。

        【技术面趋势温度分析】: {cnn_report}
        【基本面消息分析】: {news_report}
        【风控管理员指令】: {risk_report}

        执行逻辑（严格遵守）：
        1. 必须优先服从风控指令！如果风控提示回撤过大要求强行空仓避险，无视其他信号，立刻输出 WAIT。
        2. 顺势而为：当技术面强烈看涨 (Buy) 且基本面没有明确利空（即基本面为利好或中性）时，允许输出 BUY。
        3. 顺势做空：当技术面强烈看跌 (Sell) 且基本面没有明确利好（即基本面为利空或中性）时，允许输出 SELL。
        4. 如果技术面和基本面方向产生严重冲突（例如技术看涨但基本面利空），或者技术面提示震荡/观望 (Wait)，输出 WAIT。

        请严格以 JSON 格式输出，不要包含任何 Markdown 语法（不要使用 ```json 标签）。
        JSON 格式必须完全匹配如下结构：
        {{"action": "BUY", "reason": "技术面多头且基本面无利空，顺势建仓。"}}
        （注：action 的值只能是 "BUY", "SELL", 或 "WAIT"）
        """

        try:
            # 调用大模型接口
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个只输出标准 JSON 格式的无情量化交易机器，不包含任何废话。"},
                    {"role": "user", "content": prompt}
                ],
                # 开启 JSON Mode，确保大模型强制返回可解析的 JSON
                response_format={"type": "json_object"},
                # 交易决策需要绝对的确定性，将温度降到最低，减少随机性
                temperature=0.1,
            )

            # 获取大模型返回的文本
            json_result = response.choices[0].message.content

            # 简单验证一下是否真的是合法 JSON
            parsed_json = json.loads(json_result)
            if "action" not in parsed_json or "reason" not in parsed_json:
                raise ValueError("大模型返回的 JSON 缺少必要的字段。")

            return json_result

        except Exception as e:
            # 【核心兜底机制】
            # 在高频次的回测中，API 极有可能因为网络抖动、限流等原因报错。
            # 作为严肃的量化系统，一旦 API 异常，不能让程序崩溃，也不能盲目开仓。
            # 正确的做法是：行使一票否决权，默认输出 WAIT，保住本金。
            fallback_decision = {
                "action": "WAIT",
                "reason": f"系统异常兜底触发：决策引擎大模型调用失败，为规避风险自动暂停交易。错误信息：{str(e)}"
            }
            return json.dumps(fallback_decision, ensure_ascii=False)