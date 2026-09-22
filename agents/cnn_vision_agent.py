# agents/cnn_vision_agent.py
import numpy as np
from keras.models import load_model
from utils.image_utils import save_to_candlestick_to_buf, preprocess_image_from_buf


class CNNVisionAgent:
    def __init__(self, model_path='models/cnn_model.h5', img_size=128):
        self.model = load_model(model_path)
        self.img_size = img_size

    def analyze(self, df_window):
        """
        输入过去时间窗口的 DataFrame，在内存中生成 K 线图并输出统一格式的报告。
        """
        img_buf = None
        try:
            # 1. 绘制 K 线图至内存缓冲区
            img_buf = save_to_candlestick_to_buf(df_window, img_size=self.img_size)

            # 2. 图像预处理与归一化
            x_test = preprocess_image_from_buf(img_buf, self.img_size)

            # 3. 模型预测 (0: 看跌, 1: 看涨, 2: 观望)
            predicted = self.model.predict(x_test, verbose=0)
            y_pred = int(np.argmax(predicted, axis=1)[0])
            confidence = float(np.max(predicted))

            # 4. 对齐 MetaDecisionAgent 标准信号格式
            if y_pred == 1:
                pattern = "强烈看涨 (BULLISH)"
                desc = "K线图像呈现典型多头突破/上升中继形态"
            elif y_pred == 0:
                pattern = "破位看跌 (BEARISH)"
                desc = "K线图像呈现典型顶部反转/破位下行形态"
            else:
                pattern = "中性震荡 (NEUTRAL)"
                desc = "K线形态方向不明或处于无序横盘区间"

            return f"趋势视觉报告：形态判定为【{pattern}】(置信度: {confidence:.2f})，理由：{desc}。"

        except Exception as e:
            return f"趋势视觉报告：形态判定为【中性震荡 (NEUTRAL)】，理由：视觉绘制或推理异常 ({str(e)})。"

        finally:
            if img_buf is not None:
                img_buf.close()