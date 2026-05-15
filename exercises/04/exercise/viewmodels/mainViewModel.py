from PySide6.QtCore import QObject, QTimer, Signal

from models.signal_model import SignalModel


class MainViewModel(QObject):
    # 中文：当绘图更新时发出的信号，传递 x 和 y 数据
    # English: Signal emitted when plot is updated, carries x and y data
    plot_updated = Signal(object, object)

    def __init__(self):
        super().__init__()

        # TODO 1:
        # 中文：创建 SignalModel 对象
        # 使用以下参数：
        # - sampling_rate=1000 （采样率）
        # - duration=100 （信号持续时间）
        # - window_size=5000 （窗口大小）
        # - step_size=20 （每次步进的样本数）
        #
        # English: Create the SignalModel.
        # Use:
        # - sampling_rate=1000
        # - duration=100
        # - window_size=5000
        # - step_size=20
        self.model = SignalModel(
            sampling_rate=1000,
            duration=100,
            window_size=5000,
            step_size=20,        
        )

        # TODO 2:
        # 中文：初始化以下实例变量：
        # - current_index （当前窗口的起始索引，初值为 0）
        # - is_plotting （是否正在绘图，初值为 False）
        #
        # English: Initialize:
        # - current_index (current window start index, initial value 0)
        # - is_plotting (whether plotting is active, initial value False)
        self.current_index = 0
        self.is_plotting = False

        # TODO 3:
        # 中文：创建一个 QTimer 对象，并连接其 timeout 信号到 self.update_plot 方法
        # 这样每当定时器超时时，update_plot 方法就会被调用
        #
        # English: Create a QTimer and connect its timeout signal to self.update_plot
        # This way, update_plot will be called each time the timer times out
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)

    def start_plotting(self):
        # TODO 4:
        # 中文：仅在当前未绘图时才开始绘图
        # 步骤：
        # - 检查 is_plotting 是否为 False
        # - 如果是，设置 is_plotting 为 True
        # - 启动定时器，间隔设为 10 毫秒（timer.start(10)）
        #
        # English: Start plotting only if plotting is currently stopped.
        # Steps:
        # - Check if is_plotting is False
        # - If so, set is_plotting to True
        # - Start the timer with an interval of 10 ms (timer.start(10))
        if not self.is_plotting:
            self.is_plotting = True
            self.timer.start(10)
        pass

    def stop_plotting(self):
        # TODO 5:
        # 中文：仅在当前正在绘图时才停止绘图
        # 步骤：
        # - 检查 is_plotting 是否为 True
        # - 如果是，设置 is_plotting 为 False
        # - 停止定时器（timer.stop()）
        #
        # English: Stop plotting only if plotting is currently running.
        # Steps:
        # - Check if is_plotting is True
        # - If so, set is_plotting to False
        # - Stop the timer (timer.stop())
        if self.is_plotting:
            self.is_plotting = False
            self.timer.stop()
        pass

    def update_plot(self):
        if not self.model.has_enough_data(self.current_index):
            self.current_index = 0

        x, y = self.model.get_window(self.current_index)
        self.plot_updated.emit(x, y)

        self.current_index += self.model.step_size