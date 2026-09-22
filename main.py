# main.py
import os
import warnings

# 屏蔽底层 C++ 日志与警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
from agents.trend_temperature_agent import TrendTemperatureAgent
# from agents.cnn_vision_agent import CNNVisionAgent
from agents.news_rag_agent import FundamentalNewsAgent
from agents.risk_agent import RiskControlAgent
from agents.meta_decision import MetaDecisionAgent
from backtest.engine import BacktestEngine


def main():
    print("=" * 50)
    print("[START] 启动 Multi-Agent 混合量化交易系统")
    print("=" * 50)

    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("[ERROR] 未找到 DEEPSEEK_API_KEY，请检查 .env 文件！")
        return

    print(">>> 正在初始化智能体团队...")
    try:
        # 技术面分析师 (支持在趋势温度与 CNN 视觉之间灵活切换)
        tech_agent = TrendTemperatureAgent()
        # tech_agent = CNNVisionAgent(model_path='models/cnn_model.h5')
        print("  技术面分析师 (Technical Agent) 加载完成")

        news_agent = FundamentalNewsAgent(api_key=api_key)
        print("  基本面分析师 (News RAG Agent) 加载完成")

        risk_agent = RiskControlAgent()
        print("  风控管理员 (Risk Control Agent) 加载完成")

        meta_agent = MetaDecisionAgent(api_key=api_key)
        print("  首席决策官 (Meta-Decision Agent) 加载完成")
    except Exception as e:
        print(f"[ERROR] 智能体初始化失败: {e}")
        return

    print(">>> 组装回测引擎...")
    engine = BacktestEngine(
        tech_agent=tech_agent,
        news_agent=news_agent,
        risk_agent=risk_agent,
        meta_agent=meta_agent,
        commission_rate=0.001,
        slippage=0.0005
    )

    print(">>> 开始执行历史回测...")
    engine.run_backtest(
        ticker="105.NVDA",
        period="daily",
        start_date="2023-01-01",
        end_date="2024-04-01"
    )


if __name__ == "__main__":
    main()