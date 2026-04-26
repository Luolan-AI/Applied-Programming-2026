# Copyright 2025 n-squared LAB @ FAU Erlangen-Nürnberg

"""
EMG Signal Processing Exercise

Students should complete the TODO sections.
Do not change function names unless instructed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal


def load_emg_data(filename: str):
    """
    Load EMG data from a pickle file and extract:
    - the raw biosignal
    - the sampling rate

    Expected structure:
        data["biosignal"]
        data["device_information"]["sampling_frequency"]
    """

    # TODO: load the pickle file with pandas
    data = pd.read_pickle(filename) #pandas.read_pickle()读取文件,文件名字在后面的main函数

    print("Data structure:")
    print("-" * 50)
    print(f"Data type: {type(data)}")
    print(f"Data shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
    print("\nAvailable keys in data:")
    print("-" * 50)
    for key in data.keys():
        print(f"- {key}")
    print("-" * 50)

    # TODO: extract the EMG signal
    emg_signal = data['biosignal']

    # TODO: extract the sampling rate
    sampling_rate = data["device_information"]["sampling_frequency"]

    print("\nEMG Signal information:")
    print("-" * 50)
    print(f"Signal shape: {emg_signal.shape}")
    print(f"Number of channels: {emg_signal.shape[0]}")
    print(f"Window size: {emg_signal.shape[1]}")
    print(f"Number of windows: {emg_signal.shape[2]}")
    print(f"Sampling rate: {sampling_rate} Hz")

    return emg_signal, sampling_rate


def restructure_emg_data(emg_signal: np.ndarray):
    """
    Convert EMG from:
        (channels, window_size, n_windows)
    to:
        (channels, total_samples)
    """

    # TODO: determine the number of channels
    # 通道数就是原始数据的第一个维度的大小 (索引为0)
    num_channels = emg_signal.shape[0]

    # TODO: transpose and reshape so each row is one continuous channel
    # 这一步是关键逻辑：
    # 1. transpose(0, 2, 1): 把原本是 (通道数, 窗口内点数, 窗口数) 的结构
    #    转换成 (通道数, 窗口数, 窗口内点数)。相当于把时间顺序理正。
    # 2. reshape(num_channels, -1): 保持通道数不变，把后面的所有数据展平成一条直线。
    #    -1 的意思是让 Python 自动帮你计算后面加起来到底有多少个数据点。
    #reshape(num_channels, -1) 的潜台词其实是：
    # 把这堆数据变成一个 2维 矩阵。第一维度必须是 num_channels，剩下的所有数据，
    # 不管原来是几维、有多碎，全部给我揉成一团，顺着填进第二维度里，具体有多长你自己算。”
    channel_data = emg_signal.transpose(0, 2, 1).reshape(num_channels, -1)

    print("\nRestructured EMG Data:")
    print("-" * 50)
    print(f"Original shape: {emg_signal.shape}")
    print(f"New shape: {channel_data.shape}")
    print(f"Number of channels: {num_channels}")
    print(f"Total samples per channel: {channel_data.shape[1]}")

    return channel_data, num_channels


def bandpass_filter_emg(
    channel_data: np.ndarray,
    sampling_rate: float,
    low_cut: float = 20,
    high_cut: float = 450,
):
    """
    Apply a Butterworth bandpass filter to each channel.
    """

    # TODO: compute the Nyquist frequency
    #这是一个基础的信号处理定理：为了不失真地记录一个信号，
    # 你的采样率必须至少是信号最高频率的两倍。反过来说，系统能处理的最高频率，
    # 就是采样率的一半。这被称为奈奎斯特频率。
    nyquist = sampling_rate / 2

    # TODO: validate low_cut and high_cut
    # Raise ValueError if the frequencies are invalid.
    #作为一个严谨的函数，要防止别人输入瞎填的值
    # （比如最低频率比最高频率还大，或者超过了奈奎斯特限制)。
    if low_cut <= 0 or high_cut >= nyquist or low_cut >= high_cut:
        raise ValueError('Cutoff frequencies are invalid')

    # TODO: normalize the cutoff frequencies
    #Python 底层的滤波器公式只认 0 到 1 之间的比例值。
    # 0 代表没有频率，1 代表奈奎斯特频率的上限。
    # 所以我们要把绝对的赫兹（Hz）变成比例。
    low = low_cut / nyquist
    high = high_cut / nyquist

    print("\nFilter Design Parameters:")
    print("-" * 50)
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Nyquist frequency: {nyquist} Hz")
    print(f"Low cutoff: {low_cut} Hz ({low:.4f} normalized)")
    print(f"High cutoff: {high_cut} Hz ({high:.4f} normalized)")

    # TODO: design a 4th order Butterworth bandpass filter
    #设计巴特沃斯滤波器 (Butterworth Filter)这里的 b 和 a 是滤波器的数学参数
    # （你可以把它们想象成筛网的孔径和材质）。signal.butter 是 Scipy 库自带的函数，
    # 4 代表 4 阶（阶数越高，过滤得越干净，但计算越慢），
    # btype='band' 表示我们要保留中间的频段（带通）。
    b, a = signal.butter(4, [low, high], btype='band')

    # TODO: pre-allocate filtered array
    #我们要创建一个和 channel_data 形状一模一样（38行，40014列）的空白矩阵，
    # 用来装过滤好的干净数据。np.zeros_like 就是“照葫芦画瓢建个空壳”的意思。
    filtered_channels = np.zeros_like(channel_data)

    # TODO: apply filtfilt to every channel
    #用一个 for 循环，
    # 把 38 条通道的数据一条一条送进过滤器 signal.filtfilt 里洗一遍，
    # 然后存进刚才准备好的空壳里。
    for i in range(channel_data.shape[0]):
        # filtfilt 会正向洗一遍，反向洗一遍，保证信号不发生时间偏移
        filtered_channels[i, :] = signal.filtfilt(b, a, channel_data[i, :])

    print("\nFiltered Signal Information:")
    print("-" * 50)
    print(f"Shape of filtered_channels: {filtered_channels.shape}")
    print(f"Type of filtered_channels: {type(filtered_channels)}")
    print(f"Filter cutoff frequencies: {low_cut} Hz to {high_cut} Hz")

    return filtered_channels


def compute_rms(filtered_channels: np.ndarray, sampling_rate: float, window_ms: float = 100):
    """
    Compute RMS envelope using a moving window.
    """

    # TODO: convert window size from ms to samples
    window_size = None

    # TODO: pre-allocate RMS array
    rms_signals = None

    # TODO: compute RMS for each channel
    # Hint:
    # 1. square the signal
    # 2. moving average with np.convolve(..., mode="same")
    # 3. square root
    
    # 1.转换窗口大小
    #电脑在底层是不认识“毫秒”的，它只认识“数据点”。
    # 你输入的参数是 window_ms = 100（100毫秒）。
    # 我们需要把它换算成这 100 毫秒里包含了多少个数据点。
    # 100ms = 0.1秒。你的采样率是 2000Hz (每秒2000个点)。
    # 0.1秒 * 2000个点/秒 = 200个点。必须用 int() 转成整数，因为数组的长度不能是小数。
    window_size = int(sampling_rate * (window_ms / 1000.0))
    
    #2.准备空矩阵 (pre-allocate RMS array)
    # 和上一步滤波时一样，我们要先建一个大小一模一样（38行，40014列）的空盒子，用来装最后的计算结果。
    rms_signals = np.zeros_like(filtered_channels)
    
    #3.计算每个通道的 RMS (S -> M -> R)
    # 这是最核心的计算环节。我们需要用一个 for 循环，把 38 根电极通道的数据一条一条抽出来处理。
    # 在做移动平均（M）时，我们要借助一个叫“卷积”的数学方法。
    # 你可以把我们要做的滑动窗口想象成一个模具，模具里有 200 个均等的权重，加起来等于 1。
    #手动打造一个平均值滑动窗口
    #np.ones(window_size)生成200个1,除以200后,这就是一个每个权重都是1/200的平均分配器
    window = np.ones(window_size) / window_size
    
    #计算RMS for each channel
    for i in range(filtered_channels.shape[0]):
        #取出当前第i个通道的数据
        current_signal =  filtered_channels[i, :]
        
        squared_signal = current_signal ** 2
        
        #代码里用的 mode="same"（中心对齐），虽然在首尾边缘处被迫悬空补 0，显得有些别扭，
        # 但它保住了两个最重要的命根子：时间绝对对齐：波峰发生在哪一秒，算完平均后依然在那一秒。
        # 长度绝对一致：进去 40014 个点，出来依然是完美的 40014 个点。
        mean_signal = np.convolve(squared_signal, window, mode='same')
        rms_signals[i, :] = np.sqrt(mean_signal)

    print("\nRMS Signal Information:")
    print("-" * 50)
    print(f"Number of channels: {filtered_channels.shape[0]}")
    print(f"Shape of RMS signals: {rms_signals.shape}")
    print(f"Window size: {window_size} samples ({window_size / sampling_rate * 1000:.1f} ms)")

    return rms_signals


def plot_emg_processing(
    channel_data: np.ndarray,
    filtered_channels: np.ndarray,
    rms_signals: np.ndarray,
    sampling_rate: float,
    selected_channel: int = 0,
):
    """
    Plot raw, filtered, and RMS signal for one channel.
    """

    t = np.arange(channel_data.shape[1]) / sampling_rate

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    ax1.plot(t, channel_data[selected_channel, :])
    ax1.set_title(f"Original EMG Signal - Channel {selected_channel + 1}")
    ax1.set_ylabel("Amplitude (V)")

    ax2.plot(t, filtered_channels[selected_channel, :])
    ax2.set_title(f"Bandpass Filtered Signal - Channel {selected_channel + 1}")
    ax2.set_ylabel("Amplitude (V)")

    ax3.plot(t, rms_signals[selected_channel, :])
    ax3.set_title(f"RMS Signal - Channel {selected_channel + 1}")
    ax3.set_ylabel("Amplitude (V)")
    ax3.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()


def main():
    # TODO: get the filepath of the pkl file (Use / not \)
    filename = "/Users/luolan/Applied-Programming-2026/recording.pkl"

    emg_signal, sampling_rate = load_emg_data(filename)
    channel_data, _ = restructure_emg_data(emg_signal)
    filtered_channels = bandpass_filter_emg(channel_data, sampling_rate)
    rms_signals = compute_rms(filtered_channels, sampling_rate)

    # Change the channel index if needed
    plot_emg_processing(
        channel_data,
        filtered_channels,
        rms_signals,
        sampling_rate,
        selected_channel=0,
    )


if __name__ == "__main__":
    main()
