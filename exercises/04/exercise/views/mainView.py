from PySide6.QtWidgets import QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

from views.plotView import VisPyPlotWidget


class MainView(QMainWindow):
    def __init__(self, view_model):
        super().__init__()

        # TODO 1:
        # 中文：将传入的 ViewModel 对象存储到实例变量 self.view_model 中
        # English: Store the provided ViewModel in an instance variable.
        self.view_model = view_model

        self.setWindowTitle("VisPy EMG Viewer")
        self.resize(1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.info_label = QLabel("Press 'Start Plotting' to begin.")
        self.plot_widget = VisPyPlotWidget()
        self.toggle_button = QPushButton("Start Plotting")

        layout.addWidget(self.info_label)
        layout.addWidget(self.plot_widget, stretch=1)
        layout.addWidget(self.toggle_button)

        # TODO 2:
        # 中文：连接按钮的 clicked 信号到 self.toggle_plotting 方法
        # 语法：self.toggle_button.clicked.connect(self.toggle_plotting)
        #
        # English: Connect the button click to self.toggle_plotting.
        # Syntax: self.toggle_button.clicked.connect(self.toggle_plotting)
        self.toggle_button.clicked.connect(self.toggle_plotting)
        # TODO 3:
        # 中文：连接 ViewModel 的 plot_updated 信号到绘图小部件的 update_plot 方法
        # 这样当 ViewModel 发出 plot_updated 信号时，绘图会自动更新
        # 语法：self.view_model.plot_updated.connect(self.plot_widget.update_plot)
        #
        # English: Connect the ViewModel's plot_updated signal
        # to the plot widget's update_plot method.
        # Syntax: self.view_model.plot_updated.connect(self.plot_widget.update_plot)
        self.view_model.plot_updated.connect(self.plot_widget.update_plot)
        
    def toggle_plotting(self):
        # TODO 4:
        # 中文：根据当前绘图状态切换绘图的开始/停止
        # 如果 ViewModel 当前正在绘图（self.view_model.is_plotting == True）：
        #   - 调用 self.view_model.stop_plotting()
        #   - 将按钮文本改为 "Start Plotting"
        #   - 更新标签文本（self.info_label.setText(...)）
        #
        # 否则（未在绘图）：
        #   - 调用 self.view_model.start_plotting()
        #   - 将按钮文本改为 "Stop Plotting"
        #   - 更新标签文本
        #
        # English: Toggle plotting based on current state.
        # If the ViewModel is currently plotting (self.view_model.is_plotting == True):
        #   - Call self.view_model.stop_plotting()
        #   - Change button text to "Start Plotting"
        #   - Update the label text
        #
        # Otherwise:
        #   - Call self.view_model.start_plotting()
        #   - Change button text to "Stop Plotting"
        #   - Update the label text
        if self.view_model.is_plotting:
            self.view_model.stop_plotting()
            self.toggle_button.setText("Start Plotting")
            self.info_label.setText("Plotting stopped. Press 'Start Plotting' to resume.")
        else:
            self.view_model.start_plotting()
            self.toggle_button.setText("Stop Plotting")
            self.info_label.setText("Plotting in progress...")
            
        pass