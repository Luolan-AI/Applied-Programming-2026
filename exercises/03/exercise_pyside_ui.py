# Copyright 2026 n-squared LAB @ FAU Erlangen-Nuernberg

"""
练习 3 - PySide6 用户界面与嵌入式 Matplotlib
Exercise 3 - PySide6 UI with Embedded Matplotlib

功能 / Features:
- 通道选择(下拉菜单) / Channel selection (dropdown)
- 信号类型选择(下拉菜单) / Signal type selection (dropdown)
- 绘图颜色改变(按钮) / Plot color change (button)
- 自动更新图表 / Auto-updating plot

学生需要完成 TODO 部分 / Students should complete the TODO sections.
"""

import sys
import numpy as np
import pandas as pd
from scipy import signal

from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ======================
# 数据处理 / Data Processing
# ======================

def load_emg_data(filename: str):
    """从 pickle 文件加载 EMG 数据
    Load EMG data from pickle file
    
    参数 / Parameters:
        filename: pickle 文件路径 / pickle file path
    
    返回 / Returns:
        emg_signal: EMG 信号数组 / EMG signal array
        sampling_rate: 采样率 / Sampling frequency
    """
    data = pd.read_pickle(filename)
    emg_signal = data["biosignal"]
    sampling_rate = data["device_information"]["sampling_frequency"]
    return emg_signal, sampling_rate


def restructure_emg_data(emg_signal: np.ndarray):
    """重新组织 EMG 数据结构以便按通道访问
    Restructure EMG data for easy channel access
    
    参数 / Parameters:
        emg_signal: 原始 EMG 信号数组 / Raw EMG signal array
    
    返回 / Returns:
        channel_data: 每行一个通道的数据矩阵 / Data matrix with channels as rows
    """
    num_channels = emg_signal.shape[0]
    channel_data = emg_signal.transpose(2, 1, 0).reshape(-1, num_channels).T
    return channel_data


def bandpass_filter_channel(channel: np.ndarray, sampling_rate: float):
    """对信号进行带通滤波 (20-450 Hz)
    Apply bandpass filter to signal (20-450 Hz)
    
    参数 / Parameters:
        channel: 输入信号 / Input signal
        sampling_rate: 采样率 / Sampling frequency
    
    返回 / Returns:
        filtered: 滤波后的信号 / Filtered signal
    """
    nyquist = sampling_rate / 2
    low = 20 / nyquist
    high = 450 / nyquist
    b, a = signal.butter(4, [low, high], btype="band")
    return signal.filtfilt(b, a, channel)


def compute_rms_channel(channel: np.ndarray, sampling_rate: float):
    """计算信号的均方根 (RMS)
    Compute root mean square (RMS) of signal
    
    参数 / Parameters:
        channel: 输入信号 / Input signal
        sampling_rate: 采样率 / Sampling frequency
    
    返回 / Returns:
        rms: 均方根值 / RMS values
    """
    window_size = int(0.1 * sampling_rate)  # 100ms 窗口 / 100ms window
    kernel = np.ones(window_size) / window_size
    squared = channel ** 2
    mean_squared = np.convolve(squared, kernel, mode="same")
    return np.sqrt(mean_squared)


# ======================
# 用户界面 / UI
# ======================

class EMGViewer(QMainWindow):
    def __init__(self, channel_data, sampling_rate):
        super().__init__()

        self.channel_data = channel_data
        self.sampling_rate = sampling_rate

        self.colors = ["blue", "red", "green", "black"]
        self.color_index = 0

        # TODO 1: 设置窗口标题和大小 / Set window title and size
        # 任务 / Task:
        #   - 设置窗口标题为 "EMG Signal Viewer" / Set the window title to "EMG Signal Viewer"
        #   - 设置窗口大小为 1000 x 700 / Set the window size to 1000 x 700
        # 提示 / Hint:
        #   - 使用 self.setWindowTitle(title) / Use self.setWindowTitle(title)
        #   - 使用 self.resize(width, height) / Use self.resize(width, height)
        # 讲解 / Explanation:
        #   QMainWindow 是一个顶级窗口，需要设置标题(显示在窗口栏)
        #   和大小(决定初始显示尺寸)
        self.setWindowTitle("EMG Signal Viewer")
        self.resize(1000, 700)

        # 中央部件 / Central widget
        # TODO 2: 创建并设置中央部件 / Create and set central widget
        # 任务 / Task:
        #   - 创建一个 QWidget 对象 / Create a QWidget object
        #   - 将其设置为主窗口的中央部件 / Set it as the main window's central widget
        # 提示 / Hint:
        #   - 使用 QWidget() 创建部件 / Use QWidget() to create widget
        #   - 使用 self.setCentralWidget(widget) 设置 / Use self.setCentralWidget(widget)
        # 讲解 / Explanation:
        #   QMainWindow 需要一个中央部件来容纳所有其他控件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 布局 / Layouts
        # TODO 3: 创建主布局和控件布局 / Create main and control layouts
        # 任务 / Task:
        #   - 创建一个竖直布局 main_layout 并附加到中央部件
        #     / Create a vertical layout main_layout and attach to central widget
        #   - 创建一个水平布局 controls_layout / Create a horizontal layout controls_layout
        # 提示 / Hint:
        #   - 使用 QVBoxLayout() 创建竖直布局 / Use QVBoxLayout() for vertical layout
        #   - 使用 QHBoxLayout() 创建水平布局 / Use QHBoxLayout() for horizontal layout
        #   - 使用 central_widget.setLayout(main_layout) / Use central_widget.setLayout(main_layout)
        # 讲解 / Explanation:
        #   布局用来管理和排列部件。QVBoxLayout 从上到下排列，QHBoxLayout 从左到右排列
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        controls_layout = QHBoxLayout()

        # 通道选择器 / Channel selector
        # TODO 4: 创建通道标签和下拉菜单 / Create channel label and combobox
        # 任务 / Task:
        #   - 创建一个标签，文本为 "Channel:" / Create a label with text "Channel:"
        #   - 创建一个下拉菜单 / Create a combobox
        #   - 根据通道数量添加选项 ("Channel 1", "Channel 2", ...)
        #     / Add items based on number of channels
        # 提示 / Hint:
        #   - 使用 QLabel("Channel:") 创建标签 / Use QLabel("Channel:") 
        #   - 使用 QComboBox() 创建下拉菜单 / Use QComboBox()
        #   - 使用 combo.addItems([items]) 添加选项 / Use combo.addItems([items])
        #   - 通道数量可以从 self.channel_data.shape[0] 获取
        #     / Channel count from self.channel_data.shape[0]
        # 讲解 / Explanation:
        #   QComboBox 提供下拉列表供用户选择。通常与 QLabel 配对显示选项含义
        self.channel_label = QLabel("Channel:")
        self.channel_combo = QComboBox()
        num_channels = self.channel_data.shape[0]
        self.channel_combo.addItems([f"Channel {i+1}" for i in range(num_channels)])

        # 信号类型选择器 / Signal selector
        # TODO 5: 创建信号类型标签和下拉菜单 / Create signal type label and combobox
        # 任务 / Task:
        #   - 创建一个标签，文本为 "Signal:" / Create a label with text "Signal:"
        #   - 创建一个下拉菜单 / Create a combobox
        #   - 添加三个选项："Original", "Filtered", "RMS"
        #     / Add three items: "Original", "Filtered", "RMS"
        # 提示 / Hint:
        #   - 使用 QLabel("Signal:") 创建标签 / Use QLabel("Signal:")
        #   - 使用 QComboBox() 创建下拉菜单 / Use QComboBox()
        #   - 使用 combo.addItems(["Original", "Filtered", "RMS"]) / Use combo.addItems(...)
        # 讲解 / Explanation:
        #   此下拉菜单允许用户在三种信号处理类型间切换：
        #   - Original: 原始未处理信号 / raw signal
        #   - Filtered: 经过带通滤波处理 / bandpass filtered
        #   - RMS: 均方根包络 / RMS envelope
        self.signal_label = QLabel("Signal:")
        self.signal_combo = QComboBox()
        self.signal_combo.addItems(["Original", "Filtered", "RMS"])

        # 按钮：改变颜色 / Button: change color
        # TODO 6: 创建改变颜色按钮 / Create change color button
        # 任务 / Task:
        #   - 创建一个按钮，文本为 "Change Color" / Create a button with text "Change Color"
        # 提示 / Hint:
        #   - 使用 QPushButton("Change Color") / Use QPushButton("Change Color")
        # 讲解 / Explanation:
        #   按钮允许用户切换图表的颜色。点击时会调用 change_color() 方法
        self.color_button = QPushButton("Change Color")

        # 添加控件到布局 / Add controls to layout
        # TODO 7: 将控件添加到控件布局 / Add widgets to controls layout
        # 任务 / Task:
        #   按以下顺序将控件添加到 controls_layout 中：
        #   / Add widgets to controls_layout in this order:
        #   channel_label, channel_combo, signal_label, signal_combo, color_button
        # 提示 / Hint:
        #   - 使用 layout.addWidget(widget) 添加部件 / Use layout.addWidget(widget)
        #   - 按顺序添加：标签1, 下拉菜单1, 标签2, 下拉菜单2, 按钮
        #     / Add in order: label1, combobox1, label2, combobox2, button
        # 讲解 / Explanation:
        #   水平布局会自动从左到右排列这些部件
        controls_layout.addWidget(self.channel_label)
        controls_layout.addWidget(self.channel_combo)
        controls_layout.addWidget(self.signal_label)
        controls_layout.addWidget(self.signal_combo)
        controls_layout.addWidget(self.color_button)

        # TODO 8: 将控件布局添加到主布局 / Add controls layout to main layout
        # 任务 / Task:
        #   - 将 controls_layout 添加到 main_layout 中
        #     / Add controls_layout to main_layout
        # 提示 / Hint:
        #   - 使用 main_layout.addLayout(controls_layout) / Use main_layout.addLayout(controls_layout)
        # 讲解 / Explanation:
        #   这样所有控件会出现在图表上方
        main_layout.addLayout(controls_layout)

        # Matplotlib 图表 / Matplotlib figure
        # TODO 9: 创建 Matplotlib 图表和画布 / Create matplotlib figure and canvas
        # 任务 / Task:
        #   - 创建一个 Figure 对象，尺寸为 (8, 5) / Create a Figure with figsize=(8, 5)
        #   - 从该 Figure 创建一个 FigureCanvas / Create a FigureCanvas from the figure
        #   - 添加一个子图 (subplot) / Add one subplot using add_subplot(111)
        # 提示 / Hint:
        #   - 使用 Figure(figsize=(8, 5)) 创建图表 / Use Figure(figsize=(8, 5))
        #   - 使用 FigureCanvas(figure) 创建画布 / Use FigureCanvas(figure)
        #   - 使用 figure.add_subplot(111) 添加子图 / Use figure.add_subplot(111)
        # 讲解 / Explanation:
        #   Figure 是 matplotlib 的绘图对象，FigureCanvas 是 Qt 部件，
        #   允许在 PySide6 应用中嵌入 matplotlib 图表。
        #   add_subplot(111) 表示 1行1列的网格中的第1个子图
        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # TODO 10: 将画布添加到主布局 / Add canvas to main layout
        # 任务 / Task:
        #   - 将画布添加到 main_layout 中 / Add canvas to main_layout
        # 提示 / Hint:
        #   - 使用 main_layout.addWidget(self.canvas) / Use main_layout.addWidget(self.canvas)
        # 讲解 / Explanation:
        #   图表将出现在控件下方，占据主布局的剩余空间
        main_layout.addWidget(self.canvas)

        # 信号连接 / Connections
        # TODO 11: 连接信号到槽 / Connect signals to slots
        # 任务 / Task:
        #   - 连接 channel_combo 的 currentIndexChanged 信号到 self.update_plot
        #     / Connect channel_combo.currentIndexChanged to self.update_plot
        #   - 连接 signal_combo 的 currentIndexChanged 信号到 self.update_plot
        #     / Connect signal_combo.currentIndexChanged to self.update_plot
        #   - 连接 color_button 的 clicked 信号到 self.change_color
        #     / Connect color_button.clicked to self.change_color
        # 提示 / Hint:
        #   - 使用 widget.signal.connect(method) / Use widget.signal.connect(method)
        #   - 当用户改变选择时，update_plot() 会自动被调用
        #     / update_plot() will be called when selection changes
        # 讲解 / Explanation:
        #   信号-槽机制是 Qt 的核心。当信号发出时，连接的槽函数自动执行。
        #   例如：用户改变通道 → currentIndexChanged 信号发出 → update_plot() 执行 → 图表更新
        self.channel_combo.currentIndexChanged.connect(self.update_plot)
        self.signal_combo.currentIndexChanged.connect(self.update_plot)
        self.color_button.clicked.connect(self.change_color)

        # TODO 12: 调用 update_plot 显示初始图表 / Call update_plot for initial plot
        # 任务 / Task:
        #   - 调用 self.update_plot() 一次来显示初始图表
        #     / Call self.update_plot() once to display initial plot
        # 提示 / Hint:
        #   - 直接在 __init__ 末尾调用 self.update_plot()
        #     / Call self.update_plot() at the end of __init__
        # 讲解 / Explanation:
        #   在创建完所有控件后调用 update_plot() 会根据当前下拉菜单的选择
        #   绘制初始图表，否则用户打开窗口时会看到空白画布
        self.update_plot()

    # ======================
    # 逻辑函数 / Logic
    # ======================

    def change_color(self):
        """改变绘图颜色 / Change the plot color
        
        讲解 / Explanation:
        - 颜色索引循环增加 (0→1→2→3→0→...) / Color index cycles through available colors
        - 使用模运算 (%) 确保索引在有效范围内 / Modulo operator wraps index
        - 调用 update_plot() 重新绘图 / Calls update_plot() to redraw
        """
        self.color_index = (self.color_index + 1) % len(self.colors)
        self.update_plot()

    def update_plot(self):
        """更新图表显示 / Update the plot display
        
        逻辑流程 / Logic flow:
        1. 从下拉菜单获取当前选择的通道和信号类型
           / Get selected channel and signal type from comboboxes
        2. 提取该通道的原始数据和时间轴
           / Extract raw data and time axis for the channel
        3. 根据选择的信号类型对数据进行处理
           / Process data based on selected signal type
        4. 清除之前的绘图并绘制新信号
           / Clear previous plot and draw new signal
        5. 设置图表标题、标签和网格
           / Set title, labels, and grid
        6. 刷新画布显示 / Refresh canvas
        """
        ch = self.channel_combo.currentIndex()  # 获取当前通道索引 / Get current channel index
        signal_type = self.signal_combo.currentText()  # 获取信号类型 / Get signal type

        raw = self.channel_data[ch, :]  # 提取该通道数据 / Extract channel data
        t = np.arange(len(raw)) / self.sampling_rate  # 创建时间轴 / Create time axis

        # 根据信号类型选择要绘制的数据 / Select data based on signal type
        if signal_type == "Original":
            y = raw
        elif signal_type == "Filtered":
            y = bandpass_filter_channel(raw, self.sampling_rate)  # 带通滤波 / Bandpass filter
        else:  # RMS
            y = compute_rms_channel(raw, self.sampling_rate)  # 计算RMS / Compute RMS

        color = self.colors[self.color_index]  # 获取当前颜色 / Get current color

        # 绘制图表 / Plot
        self.ax.clear()  # 清除之前的绘图 / Clear previous plot
        self.ax.plot(t, y, color=color)
        self.ax.set_title(f"{signal_type} - Channel {ch+1}")
        self.ax.set_xlabel("Time (s)")  # 横轴：时间 / X-axis: Time
        self.ax.set_ylabel("Amplitude")  # 纵轴：幅度 / Y-axis: Amplitude
        self.ax.grid(True)  # 显示网格 / Show grid

        self.canvas.draw()  # 刷新画布 / Refresh canvas


# ======================
# 主程序 / Main
# ======================

def main():
    """主程序入口 / Main program entry point
    
    流程 / Flow:
    1. 加载 EMG 数据 / Load EMG data
    2. 重新组织数据结构 / Restructure data
    3. 创建 Qt 应用和窗口 / Create Qt application and window
    4. 显示窗口并运行事件循环 / Show window and run event loop
    """
    filename = "recording.pkl"  # EMG 数据文件 / EMG data file

    emg_signal, sampling_rate = load_emg_data(filename)  # 加载数据 / Load data
    channel_data = restructure_emg_data(emg_signal)  # 重新组织 / Restructure

    app = QApplication(sys.argv)  # 创建 Qt 应用 / Create Qt application
    window = EMGViewer(channel_data, sampling_rate)  # 创建窗口 / Create window
    window.show()  # 显示窗口 / Show window
    sys.exit(app.exec())  # 运行事件循环 / Run event loop


if __name__ == "__main__":
    main()
