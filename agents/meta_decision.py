# agents/meta_decision.py
import re
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

    def _generate_cache_key(self, tech_report, news_report, risk_report):
        """
        提取结构化语义特征生成哈希，避免技术面连续浮点数微小扰动击穿缓存。
        """
        tech_pattern_match = re.search(r"【(.*?)】", tech_report)
        tech_pattern = tech_pattern_match.group(1) if tech_pattern_match else tech_report[:30]

        risk_state = "RISK_ALERT" if "ALERT" in risk_report or "STOP" in risk_report else "RISK_OK"
        normalized_str = f"{tech_pattern}_{news_report.strip()}_{risk_state}"
        return hashlib.md5(normalized_str.encode('utf-8')).hexdigest()

    def make_decision(self, tech_report, news_report, risk_report):
        cache_key = self._generate_cache_key(tech_report, news_report, risk_report)

        if cache_key in self.decision_cache:
            return self.decision_cache[cache_key]

        prompt = f"""
你是量化投资系统的客观市场研判官。请根据以下三份专业报告，评估当前标的的综合市场立场 (Market Stance)。

【技术面趋势报告】: {tech_report}
【基本面情报报告】: {news_report}
【风控状态报告】: {risk_report}

判断规则（严格执行）：
1. 优先遵守风控：若风控提示破位或异常波动，必须判定为 NEUTRAL（现金防御）。
2. 当技术面强烈看涨且基本面无利空时，输出 BULLISH（主升浪开多/加仓）。
3. 当技术面提示“多头良性回调 (BULLISH_HOLD)”且基本面无利空时，输出 BULLISH_HOLD（多头防守中继，持股者拿住，空仓者不追高）。
4. 当技术面破位看跌且基本面无利好时，输出 BEARISH（看跌）。
5. 信号严重冲突、震荡无序或技术转弱跌破支撑时，输出 NEUTRAL（中性观望/落袋结利）。

请严格以 JSON 格式输出：
{{"stance": "BULLISH", "reason": "分析理由"}}

可选 stance 仅限: "BULLISH", "BULLISH_HOLD", "BEARISH", "NEUTRAL"
"""

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个只输出标准 JSON 格式的无情量化分析机器。"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            json_result = response.choices[0].message.content
            parsed = json.loads(json_result)
            if "stance" not in parsed or "reason" not in parsed:
                raise ValueError("JSON 缺少必要字段")

            if parsed["stance"] not in ["BULLISH", "BULLISH_HOLD", "BEARISH", "NEUTRAL"]:
                parsed["stance"] = "NEUTRAL"
                json_result = json.dumps(parsed, ensure_ascii=False)

            self.decision_cache[cache_key] = json_result
            return json_result

        except Exception as e:
            fallback = {
                "stance": "NEUTRAL",
                "reason": f"决策接口异常兜底: {str(e)}"
            }
            return json.dumps(fallback, ensure_ascii=False)