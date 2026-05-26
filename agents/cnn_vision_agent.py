# agents/cnn_vision_agent.py
import numpy as np
from keras.models import load_model
from utils.image_utils import save_to_candlestick_to_buf, preprocess_image_from_buf

class CNNVisionAgent:
    def __init__(self, model_path='models/cnn_model.h5'):
        self.model = load_model(model_path)
        self.img_size = 128

    def analyze(self, df_window):
        """
        输入过去 10 个时间步的 DataFrame，在内存中生成 K 线图并输出分析报告。
        """
        try:
            # 1. 实时将 df_window 绘制到内存缓冲区 (io.BytesIO)
            img_buf = save_to_candlestick_to_buf(df_window, img_size=self.img_size)

            # 2. 从内存读取并完成归一化
            x_test = preprocess_image_from_buf(img_buf, self.img_size)

            # 3. 模型预测
            predicted = self.model.predict(x_test, verbose=0)
            y_pred = np.argmax(predicted, axis=1)[0]

            # 4. 根据实际训练标签进行映射：0=看跌，1=看涨，2=观望
            if y_pred == 1:
                signal = "看涨 (Buy)"
            elif y_pred == 0:
                signal = "看跌 (Sell)"
            else:
                signal = "震荡/观望 (Wait)"

            # 关闭内存流，释放内存资源
            img_buf.close()

            return f"CNN视觉分析师报告：基于过去10个时间步的K线形态提取，当前呈现 {signal} 模式。"

        except Exception as e:
            # 视觉绘制或预测异常时的安全兜底
            return f"CNN视觉分析师报告：画图或预测异常，默认输出震荡/观望 ({e})"