import json
import hashlib
from openai import OpenAI


class MetaDecisionAgent:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("API Key 缺失")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            timeout=15.0
        )
        self.decision_cache = {}

    def make_decision(self, tech_report, news_report, risk_report):
        """
        纯无状态决策：仅根据技术面、基本面和风控报告研判当前市场的客观多空属性。
        不引入任何账户持仓、成本价或浮亏数据，消除处置效应与沉没成本心理偏差。
        """
        cache_key = hashlib.md5(
            f"{tech_report}_{news_report}_{risk_report}".encode('utf-8')
        ).hexdigest()

        if cache_key in self.decision_cache:
            return self.decision_cache[cache_key]

        prompt = f"""
你是量化投资系统的客观市场研判官。请根据以下三份专业报告，评估当前标的的综合市场立场 (Market Stance)。

【技术面趋势报告】: {tech_report}
【基本面情报报告】: {news_report}
【风控状态报告】: {risk_report}

判断逻辑（严格执行）：
1. 优先遵守风控：若风控提示破位或异常波动，市场立场必须判定为 NEUTRAL（现金防御）。
2. 技术面与基本面共振看涨，且无重大利空时，判定为 BULLISH（多头优势）。
3. 技术面与基本面共振看跌，且无重大利好时，判定为 BEARISH（空头优势）。
4. 若技术面与基本面冲突、处于宽幅震荡整理、或信息不足时，判定为 NEUTRAL（中性观望）。

请严格以 JSON 格式输出：
{{"stance": "BULLISH", "confidence": 0.85, "reason": "技术面突破且基本面无利空"}}

可选 stance 仅限: "BULLISH" (看多), "BEARISH" (看空), "NEUTRAL" (中性/现金)
"""

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个只输出标准 JSON 格式的无情量化分析机器，不包含主观情绪。"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            json_result = response.choices[0].message.content
            parsed = json.loads(json_result)
            if "stance" not in parsed or "reason" not in parsed:
                raise ValueError("JSON 缺少必要字段")

            self.decision_cache[cache_key] = json_result
            return json_result

        except Exception as e:
            fallback = {
                "stance": "NEUTRAL",
                "confidence": 0.0,
                "reason": f"决策接口异常兜底，默认保持中性现金状态: {str(e)}"
            }
            return json.dumps(fallback, ensure_ascii=False)