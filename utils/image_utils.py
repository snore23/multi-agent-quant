# utils/image_utils.py
import io
import pandas as pd
import numpy as np
import mplfinance as mpf
from PIL import Image


def save_to_candlestick_to_buf(df_window, img_size=128):
    """
    将时间窗口内的 K 线数据保存为图像，并输出到内存缓冲区中，避免落盘 I/O。
    """
    buf = io.BytesIO()

    # 确保 dataframe 的 index 是 Datetime 格式
    if not isinstance(df_window.index, pd.DatetimeIndex):
        df_window = df_window.copy()
        df_window['datetime'] = pd.to_datetime(df_window['datetime'])
        df_window.set_index('datetime', inplace=True)

    # 自定义 K 线图风格 (红涨绿跌)
    mc = mpf.make_marketcolors(up='r', down='g', edge='inherit', wick='inherit')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='', y_on_right=False)

    kwargs = dict(
        type='candle',
        style=s,
        volume=False,
        axisoff=True,
        figsize=(img_size / 100, img_size / 100)  # 1.28 x 1.28 英寸
    )

    # 使用字典格式传递 savefig 参数，指定保存在内存 buf 且格式为 png
    save_config = dict(fname=buf, dpi=100, bbox_inches='tight', pad_inches=0, format='png')

    mpf.plot(df_window, **kwargs, savefig=save_config)
    buf.seek(0)  # 将指针移回内存流起点以便读取
    return buf


def preprocess_image_from_buf(buf, img_size=128):
    """
    从内存缓冲区读取图片为 numpy 数组并归一化至 [0, 1] 之间。
    """
    img = Image.open(buf).convert('RGB')
    img = img.resize((img_size, img_size))
    img_array = np.array(img) / 255.0
    # 增加 batch_size 维度 (1, 128, 128, 3)
    return np.expand_dims(img_array, axis=0)