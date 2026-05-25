import os
import warnings

# 必须放在所有 import 之前！
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 彻底屏蔽 TF 底层 C++ 日志
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # 屏蔽 oneDNN 警告
warnings.filterwarnings('ignore')         # 屏蔽 Python 层面的 UserWarning

from dotenv import load_dotenv

# 导入你的四大核心 Agent
# from agents.cnn_vision_agent import CNNVisionAgent
from agents.trend_temperature_agent import TrendTemperatureAgent
from agents.news_rag_agent import FundamentalNewsAgent
from agents.risk_agent import RiskControlAgent
from agents.meta_decision import MetaDecisionAgent  # 确保你的文件名是 meta_decision.py
from backtest.engine import BacktestEngine


def main():
    print("==================================================")
    print("🚀 启动 Multi-Agent 混合量化交易系统")
    print("==================================================")

    # 1. 加载环境变量 (极度重要：从本地 .env 文件读取 API Key)
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ 错误: 未找到 DEEPSEEK_API_KEY，请检查 .env 文件！")
        return

    # 2. 初始化四大智能体
    print(">>> 正在初始化智能体团队...")
    try:
        # cnn_agent = CNNVisionAgent(model_path='models/cnn_model.h5')
        cnn_agent = TrendTemperatureAgent()
        # print("  ✅ 视觉分析师 (CNN Vision Agent) 加载完成")
        print("  ✅ 趋势温度分析师 (Trend Temperature Agent) 加载完成")

        news_agent = FundamentalNewsAgent(api_key=api_key)
        print("  ✅ 基本面分析师 (News RAG Agent) 加载完成")

        risk_agent = RiskControlAgent()
        print("  ✅ 风控管理员 (Risk Control Agent) 加载完成")

        meta_agent = MetaDecisionAgent(api_key=api_key)
        print("  ✅ 首席决策官 (Meta-Decision Agent) 加载完成")
    except Exception as e:
        print(f"❌ 智能体初始化失败: {e}")
        return

    # 3. 组装并启动回测引擎
    print(">>> 组装回测引擎...")
    engine = BacktestEngine(
        cnn_agent=cnn_agent,
        news_agent=news_agent,
        risk_agent=risk_agent,
        meta_agent=meta_agent
    )

    # 4. 执行回测 (测试最近几天的数据)
    print(">>> 开始执行历史回测...")
    engine.run_backtest(
        ticker="105.NVDA",  # akshare中英伟达的代码
        period="daily",  # 改为日线
        # 时间拉长到 2023 年至 2024 年之间（英伟达AI暴涨期）
        start_date="2023-01-01 00:00:00",
        end_date="2024-04-01 00:00:00"
    )


if __name__ == "__main__":
    main()