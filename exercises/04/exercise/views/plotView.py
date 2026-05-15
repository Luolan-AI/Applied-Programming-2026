from PySide6.QtWidgets import QVBoxLayout, QWidget
from vispy import scene
import numpy as np


class VisPyPlotWidget(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.canvas = scene.SceneCanvas(
            keys="interactive",
            show=False,
            bgcolor="white",
            size=(1000, 600),
        )

        grid = self.canvas.central_widget.add_grid(margin=10)

        self.y_axis = scene.AxisWidget(orientation="left")
        self.x_axis = scene.AxisWidget(orientation="bottom")

        self.y_axis.width_max = 50
        self.x_axis.height_max = 40

        grid.add_widget(self.y_axis, row=0, col=0)

        self.view = grid.add_view(row=0, col=1)
        self.view.camera = "panzoom"

        grid.add_widget(self.x_axis, row=1, col=1)

        self.x_axis.link_view(self.view)
        self.y_axis.link_view(self.view)

        self.line = scene.Line(
            pos=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
            color=(0.1, 0.3, 0.8, 1.0),
            parent=self.view.scene,
            width=2,
        )

        layout.addWidget(self.canvas.native)

    def update_plot(self, x, y):
        # 将输入数据转换为 float 类型的 numpy 数组
        # Convert input data to float numpy arrays
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        # TODO 1:
        # 中文：将 x 和 y 合并成一个 (N, 2) 的位置数组
        # 例如：pos = np.column_stack([x, y])
        #
        # English: Combine x and y into an (N, 2) position array.
        # Example: pos = np.column_stack([x, y])
        pos = np.column_stack([x, y])

        # TODO 2:
        # 中文：用新的位置数据更新线条的数据
        # 语法：self.line.set_data(pos)
        #
        # English: Update the line data with the new positions.
        # Syntax: self.line.set_data(pos)

        # 计算 y 轴的填充量（为了更好的可视化范围）
        # Calculate y-axis padding for better visualization
        self.line.set_data(pos)
        y_pad = max(0.1, 0.1 * (y.max() - y.min() + 1e-9))

        # TODO 3:
        # 中文：更新摄像机的范围，使当前数据窗口可见
        # 使用：
        # - x 的最小值和最大值
        # - y 的最小值和最大值（加上填充）
        # 语法：self.view.camera.set_range(
        #         x=(x.min(), x.max()),
        #         y=(y.min() - y_pad, y.max() + y_pad)
        #       )
        #
        # English: Update the camera range so the current data window is visible.
        # Use:
        # - x min/max
        # - y min/max with padding
        # Syntax: self.view.camera.set_range(
        #           x=(x.min(), x.max()),
        #           y=(y.min() - y_pad, y.max() + y_pad)
        #         )
        self.view.camera.set_range(
                 x=(x.min(), x.max()),
                 y=(y.min() - y_pad, y.max() + y_pad)
               )