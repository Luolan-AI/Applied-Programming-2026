# 练习 3 — PySide6 用户界面编程与嵌入式 Matplotlib
# Exercise 3 — PySide6 UI Programming & Embedded Matplotlib

---

## 概述 / Overview

在这个练习中，你将学习如何使用 **PySide6** 构建 **桌面应用程序**，并将其与 Matplotlib 集成以进行信号可视化。
In this exercise, you will learn how to build a **desktop application** using **PySide6** and integrate it with Matplotlib for signal visualization.

这是课程中的一个关键转变 / This is a key transition in the course:

* 从 **脚本 → 应用程序** / From **scripts → applications**
* 从 **静态图表 → 交互式界面** / From **static plots → interactive UI**
* 从 **顺序代码 → 事件驱动编程** / From **sequential code → event-driven programming**

到最后，你应该理解图形用户界面的基本构成部分 / By the end, you should understand the basic building blocks of a graphical user interface:

* 窗口 / windows
* 部件 / widgets
* 布局 / layouts
* 按钮 / buttons
* 下拉菜单 / dropdowns
* 滑块 / sliders
* 复选框 / checkboxes
* 单选按钮 / radio buttons
* 文本输入 / text inputs
* 表格 / tables
* 选项卡 / tabs
* 菜单 / menus
* 对话框 / dialogs
* 信号和槽 / signals and slots
* 嵌入式图表 / embedded plots

---

## 为什么要学习 GUI 编程？/ Why GUI Programming?

到目前为止，你的程序通常：/ So far, your programs usually:

* 运行一次 / run once
* 产生输出 / produce output
* 退出 / exit

现在我们想要 / Now we want:

* 持续运行的应用程序 / persistent applications
* 用户交互 / user interaction
* 动态更新 / dynamic updates
* 视觉反馈 / visual feedback
* 可重用的界面 / reusable interfaces

图形用户界面允许用户在不编辑源代码的情况下控制程序。/ A graphical user interface allows users to control a program without editing the source code.

示例 / Example:

不是手动更改这个 / Instead of changing this manually:

```python
channel = 3
signal_type = "RMS"
plot_color = "red"
```

我们可以让用户用以下方式选择这些值 / we can let the user select these values with:

* 下拉菜单 / a dropdown
* 按钮 / a button
* 滑块 / a slider
* 复选框 / a checkbox

---

## 什么是 PySide6？/ What is PySide6?

PySide6 是 **Qt 的官方 Python 绑定**，是一个强大的跨平台 UI 框架。/ PySide6 is the **official Python binding of Qt**, a powerful cross-platform UI framework.

它提供 / It provides:

* **部件（Widgets）** → UI 元素，如按钮、下拉菜单、标签和滑块 / **Widgets** → UI elements such as buttons, dropdowns, labels and sliders
* **布局（Layouts）** → 排列部件的规则 / **Layouts** → rules for arranging widgets
* **信号和槽（Signals & Slots）** → UI 元素和 Python 函数之间的通信 / **Signals & Slots** → communication between UI elements and Python functions
* **事件循环（Event loop）** → 持续监听用户交互 / **Event loop** → continuously listens for user interaction

---

## PySide6 应用程序的结构 / Structure of a PySide6 Application

每个 PySide6 应用程序都有相同的核心结构。/ Every PySide6 application has the same core structure.

### 1. 导入所需的类 / Import the required classes

```python
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
```

---

### 2. 创建应用程序 / Create the application

```python
app = QApplication(sys.argv)
```

应用程序对象 / The application object:

* 管理事件循环 / manages the event loop
* 接收鼠标和键盘事件 / receives mouse and keyboard events
* 保持程序运行 / keeps the program alive

---

### 3. 创建主窗口 / Create the main window

```python
window = QMainWindow()
```

这是 **顶级容器** / This is the **top-level container**.

---

### 4. 添加中央部件 / Add a central widget

```python
central_widget = QWidget()
window.setCentralWidget(central_widget)
```

重要 / Important:

* `QMainWindow` 不能直接包含布局 / `QMainWindow` cannot directly contain layouts
* 所有东西都要放在 `QWidget` 内 / everything goes inside a `QWidget`
* `QWidget` 接收布局 / the `QWidget` receives the layout

---

### 5. 创建布局 / Create a layout

```python
layout = QVBoxLayout()
central_widget.setLayout(layout)
```

---

### 6. 显示窗口 / Show the window

```python
window.show()
app.exec()
```

`app.exec()` 启动事件循环 / `app.exec()` starts the event loop.

---

## 最小 PySide6 应用 / Minimal PySide6 App

```python
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel

app = QApplication(sys.argv)

window = QMainWindow()
window.setWindowTitle("My First PySide6 App")
window.resize(600, 400)

central_widget = QWidget()
window.setCentralWidget(central_widget)

layout = QVBoxLayout()
central_widget.setLayout(layout)

label = QLabel("Hello PySide6!")
layout.addWidget(label)

window.show()
app.exec()
```

---

## 详细的布局 / Layouts in Detail

布局控制部件的排列方式 / Layouts control how widgets are arranged.

---

### QVBoxLayout — 竖直布局 / QVBoxLayout — Vertical Layout

从上到下排列部件 / Arranges widgets from top to bottom.

```text
+------------------+
|     Widget 1     |
+------------------+
|     Widget 2     |
+------------------+
|     Widget 3     |
+------------------+
```

```python
layout = QVBoxLayout()
layout.addWidget(widget1)
layout.addWidget(widget2)
layout.addWidget(widget3)
```

常见使用场景 / Typical use:

* 主应用程序结构 / main application structure
* 表单 / forms
* 堆栈控件 / stacked controls

---

### QHBoxLayout — 水平布局 / QHBoxLayout — Horizontal Layout

从左到右排列部件 / Arranges widgets side by side.

```text
+-------+-------+-------+
|Widget1|Widget2|Widget3|
+-------+-------+-------+
```

```python
layout = QHBoxLayout()
layout.addWidget(widget1)
layout.addWidget(widget2)
layout.addWidget(widget3)
```

常见使用场景 / Typical use:

* 工具栏行 / toolbar rows
* 相邻的按钮 / buttons next to each other
* 标签 + 输入对 / label + input pairs

---

### QGridLayout — 网格布局 / QGridLayout — Grid Layout

按行、列排列部件 / Arranges widgets in rows and columns.

```text
+----------+----------+
| Row 0,0  | Row 0,1  |
+----------+----------+
| Row 1,0  | Row 1,1  |
+----------+----------+
```

```python
from PySide6.QtWidgets import QGridLayout, QLabel, QLineEdit

grid = QGridLayout()

grid.addWidget(QLabel("Name:"), 0, 0)
grid.addWidget(QLineEdit(), 0, 1)

grid.addWidget(QLabel("Age:"), 1, 0)
grid.addWidget(QLineEdit(), 1, 1)
```

常见使用场景 / Typical use:

* 设置对话框 / settings dialogs
* 表单 / forms
* 结构化输入面板 / structured input panels

---

### QFormLayout — 表单布局 / QFormLayout — Form Layout

最适合标签-输入表单 / Best for label-input forms.

```python
from PySide6.QtWidgets import QFormLayout, QLineEdit, QSpinBox

form = QFormLayout()
form.addRow("Patient name:", QLineEdit())
form.addRow("Age:", QSpinBox())
```

常见使用场景 / Typical use:

* 参数输入 / parameter input
* 配置面板 / configuration panels
* 数据输入 / data entry

---

### 嵌套布局 / Nested Layouts

你可以组合布局 / You can combine layouts.

```text
+------------------------------+
|           Plot Area          |
+------------------------------+
| Channel | Signal | Button    |
+------------------------------+
```

```python
main_layout = QVBoxLayout()
controls_layout = QHBoxLayout()

controls_layout.addWidget(channel_combo)
controls_layout.addWidget(signal_combo)
controls_layout.addWidget(color_button)

main_layout.addLayout(controls_layout)
main_layout.addWidget(plot_widget)
```

Standard pattern:

* vertical layout → overall structure
* horizontal layout → control row
* grid or form layout → parameter input

---

# 部件库 / Widget Gallery

下面的例子演示了实际应用中使用的常见部件 / The following examples show common widgets used in real applications.

---

## QLabel — 显示文本 / QLabel — Display Text

```python
label = QLabel("Channel:")
```

有用于 / Useful for:

* 标题 / titles
* 描述文本 / descriptions
* 状态文本 / status text
* 输入旁的标签 / labels next to inputs

你可以动态更新标签 / You can update a label dynamically:

```python
label.setText("New value selected")
```

---

## QPushButton — 按钮 / QPushButton — Button

```python
button = QPushButton("Start")
```

将按钮连接到一个函数 / Connect the button to a function:

```python
def start_measurement():
    print("Measurement started")

button.clicked.connect(start_measurement)
```

有用于 / Useful for:

* 启动一个动作 / starting an action
* 停止一个动作 / stopping an action
* 保存数据 / saving data
* 重置一个视图 / resetting a view
* 打开一个文件 / opening a file

---

## QComboBox — 下拉菜单 / QComboBox — Dropdown

```python
combo = QComboBox()
combo.addItems(["Original", "Filtered", "RMS"])
```

读取选中的文本 / Read the selected text:

```python
selected = combo.currentText()
```

对变化做出反应 / React to changes:

```python
def selection_changed():
    print(combo.currentText())

combo.currentIndexChanged.connect(selection_changed)
```

有用于 / Useful for:

* 选择一个通道 / selecting a channel
* 选择一个信号类型 / selecting a signal type
* 选择一个模式 / selecting a mode
* 选择一个设备 / selecting a device

---

## QSlider — 滑块 / QSlider — Slider

```python
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSlider

slider = QSlider(Qt.Horizontal)
slider.setMinimum(0)
slider.setMaximum(100)
slider.setValue(50)
```

对滑块移动做出反应 / React to slider movement:

```python
def slider_changed(value):
    print("Slider value:", value)

slider.valueChanged.connect(slider_changed)
```

有用于 / Useful for:

* 幅度 / amplitude
* 频率 / frequency
* 滤波截断频率 / filter cutoff
* 缩放级别 / zoom level
* 播放位置 / playback position

---

## QSpinBox — 整数输入 / QSpinBox — Integer Input

```python
from PySide6.QtWidgets import QSpinBox

spinbox = QSpinBox()
spinbox.setMinimum(1)
spinbox.setMaximum(64)
spinbox.setValue(1)
```

对值的变化做出反应 / React to value changes:

```python
def channel_changed(value):
    print("Channel:", value)

spinbox.valueChanged.connect(channel_changed)
```

有用于 / Useful for:

* 通道号 / channel number
* 重复次数 / repetitions
* 样本索引 / sample index
* 计数值 / count values

---

## QDoubleSpinBox — 小数输入 / QDoubleSpinBox — Decimal Input

```python
from PySide6.QtWidgets import QDoubleSpinBox

cutoff_spinbox = QDoubleSpinBox()
cutoff_spinbox.setMinimum(0.1)
cutoff_spinbox.setMaximum(1000.0)
cutoff_spinbox.setValue(20.0)
cutoff_spinbox.setSuffix(" Hz")
```

有用于 / Useful for:

* 频率值 / frequency values
* 增益 / gain
* 阈值 / thresholds
* 时间常数 / time constants

---

## QCheckBox — 复选框 / QCheckBox — On/Off Option

```python
from PySide6.QtWidgets import QCheckBox

checkbox = QCheckBox("Show grid")
checkbox.setChecked(True)
```

对变化做出反应 / React to changes:

```python
def grid_changed(checked):
    print("Grid enabled:", checked)

checkbox.toggled.connect(grid_changed)
```

有用于 / Useful for:

* 显示网格 / show grid
* 启用滤波器 / enable filter
* 正一化信号 / normalize signal
* 显示标记 / display markers

---

## QRadioButton — 单选按钮 / QRadioButton — Choose One Option

单选按钮在用户应算一个小事件组中一个选项时会会很有用 / Radio buttons are useful when the user should choose **one option from a small group**.

```python
from PySide6.QtWidgets import QRadioButton, QButtonGroup

radio_raw = QRadioButton("Raw")
radio_filtered = QRadioButton("Filtered")
radio_rms = QRadioButton("RMS")

radio_raw.setChecked(True)

group = QButtonGroup()
group.addButton(radio_raw)
group.addButton(radio_filtered)
group.addButton(radio_rms)
```

有用于 / Useful for:

* 显示模式 / display mode
* 获取模式 / acquisition mode
* 分析方法 / analysis method

---

## QLineEdit — 单行文本输入 / QLineEdit — Single-Line Text Input

```python
from PySide6.QtWidgets import QLineEdit

name_input = QLineEdit()
name_input.setPlaceholderText("Enter participant name")
```

读取文本 / Read the text:

```python
name = name_input.text()
```

当用户按下Enter键时做出反应 / React when the user presses Enter:

```python
def name_entered():
    print(name_input.text())

name_input.returnPressed.connect(name_entered)
```

有用于 / Useful for:

* 名称 / names
* ID / IDs
* 文件名 / file names
* 短参数 / short parameters

---

## QTextEdit — 多行文本输入 / QTextEdit — Multi-Line Text Input

```python
from PySide6.QtWidgets import QTextEdit

notes = QTextEdit()
notes.setPlaceholderText("Write notes here...")
```

读取文本 / Read the text:

```python
text = notes.toPlainText()
```

有用于 / Useful for:

* 评论 / comments
* 实验笔记 / experiment notes
* 日志 / logs
* 描述文本 / descriptions

---

## QListWidget — 简单列表 / QListWidget — Simple List

```python
from PySide6.QtWidgets import QListWidget

list_widget = QListWidget()
list_widget.addItems(["Trial 1", "Trial 2", "Trial 3"])
```

对选择做出反应 / React to selection:

```python
def trial_selected():
    item = list_widget.currentItem()
    if item is not None:
        print(item.text())

list_widget.currentItemChanged.connect(trial_selected)
```

有用于 / Useful for:

* 试验列表 / trial lists
* 文件列表 / file lists
* 选择的通道 / selected channels
* 测量会话 / measurement sessions

---

## QTableWidget — 表格 / QTableWidget — Table

```python
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

table = QTableWidget()
table.setRowCount(3)
table.setColumnCount(2)
table.setHorizontalHeaderLabels(["Channel", "RMS"])

table.setItem(0, 0, QTableWidgetItem("1"))
table.setItem(0, 1, QTableWidgetItem("0.24"))
```

有用于 / Useful for:

* 测量结果 / measurement results
* 通道值 / channel values
* 配置表 / configuration tables
* 汇总统计 / summary statistics

---

## QProgressBar — 进度指示器 / QProgressBar — Progress Indicator

```python
from PySide6.QtWidgets import QProgressBar

progress = QProgressBar()
progress.setMinimum(0)
progress.setMaximum(100)
progress.setValue(25)
```

有用于 / Useful for:

* 加载数据 / loading data
* 处理信号 / processing signals
* 记录进度 / recording progress
* 导出进度 / export progress

---

## QTabWidget — 选项卡 / QTabWidget — Tabs

```python
from PySide6.QtWidgets import QTabWidget, QWidget, QVBoxLayout, QLabel

tabs = QTabWidget()

page1 = QWidget()
page1_layout = QVBoxLayout()
page1_layout.addWidget(QLabel("Signal view"))
page1.setLayout(page1_layout)

page2 = QWidget()
page2_layout = QVBoxLayout()
page2_layout.addWidget(QLabel("Settings"))
page2.setLayout(page2_layout)

tabs.addTab(page1, "Plot")
tabs.addTab(page2, "Settings")
```

有用于 / Useful for:

* 分离视图 / separating views
* 设置页面 / settings pages
* 图表对比结果 / plots vs results
* 简单多页应用 / simple multi-page applications

---

## QMessageBox — 对话框窗口 / QMessageBox — Dialog Window

```python
from PySide6.QtWidgets import QMessageBox

QMessageBox.information(window, "Info", "Data saved successfully.")
```

有用于 / Useful for:

* 警告 / warnings
* 错误 / errors
* 确认 / confirmations
* 短反馈 / short feedback

---

## QFileDialog — 打开或保存文件 / QFileDialog — Open or Save Files

```python
from PySide6.QtWidgets import QFileDialog

filename, _ = QFileDialog.getOpenFileName(
    window,
    "Open File",
    "",
    "CSV Files (*.csv);;All Files (*)"
)

if filename:
    print(filename)
```

有用于 / Useful for:

* 打开 CSV 文件 / opening CSV files
* 加载信号数据 / loading signal data
* 保存结果 / saving results
* 导出图表 / exporting plots

---

# 信号和槽 / Signals and Slots

这是 GUI 编程中最重要的概念 / This is the most important concept in GUI programming.

而不是直接调用函数 / Instead of calling functions directly:

```python
do_something()
```

我们连接事件到函数 / we connect events to functions:

```python
button.clicked.connect(do_something)
```

含义 / Meaning:

> 当这个事件发生时，执行这个函数 / When this event happens, execute this function.

---

## 信号对比槽 / Signals vs Slots

| 榀念 / Concept | 含义 / Meaning | 例子 / Example |
| ------- | ------- | ------- |
| 信号 / Signal | ሴ查面发生了什么 / Something happened | button.clicked |
| 槽 / Slot | 应对的函数 / Function that reacts | update_plot |

---

## 常见的信号 / Common Signals

| 部件 / Widget | 信号 / Common signal | 含义 / Meaning |
| ------ | ------------- | ------- |
| QPushButton | clicked | 按钮被点击 / Button was clicked |
| QComboBox | currentIndexChanged | 选择设改 / Selection changed |
| QSlider | valueChanged | 滑块被移动 / Slider moved |
| QSpinBox | valueChanged | 整数设改 / Number changed |
| QCheckBox | toggled | 复选框变化 / Checkbox changed |
| QLineEdit | textChanged | 文本变化 / Text changed |
| QLineEdit | returnPressed | Enter键被按下 / Enter was pressed |
| QListWidget | currentItemChanged | 选中项变化 / Selected item changed |

---

## 多个信号可以触发同一个函数 / Multiple Signals Can Trigger the Same Function

```python
channel_combo.currentIndexChanged.connect(update_plot)
signal_combo.currentIndexChanged.connect(update_plot)
amplitude_slider.valueChanged.connect(update_plot)
grid_checkbox.toggled.connect(update_plot)
```

这很强大，因为 / This is powerful because:

* UI 可以以许多方式变化 / the UI can change in many ways
* 图表更新逻辑只写一次 / the plot update logic is written only once
* 程序更易于维持 / the program remains easier to maintain

---

## 重要的思维模型 / Important Mental Model

GUI 代码这样工作 / GUI code works like this:

```text
WAIT -> EVENT -> FUNCTION -> UPDATE UI
```

而不是这样 / Not like this:

```text
RUN -> FINISH
```

---

# 事件驱动编程 / Event-Driven Programming

不类似于普通脚本，GUI 代码不是简单从上到下运行一次 / Unlike normal scripts, GUI code does not simply run once from top to bottom.

相反 / Instead:

1. 应用程序启动 / the application starts
2. 窗口出现 / the window appears
3. 程序等待 / the program waits
4. 用户互动 / the user interacts
5. 信号被发出 / a signal is emitted
6. 连接的函数事棋 / a connected function runs
7. UI 更新 / the UI updates
8. 程序再次等待 / the program waits again

事件案例 / Examples of events:

* 按钮点击 / button click
* 下拉菜单修改 / dropdown change
* 滑块移动 / slider movement
* 复选框切换 / checkbox toggle
* 文本输入 / text input
* 文件选择 / file selection

---

# 在 PySide6 中嵌入 Matplotlib / Integrating Matplotlib into PySide6

通常 Matplotlib 会打开自己的窗口 / Normally, Matplotlib opens its own window.

在这个练习中，我们将图表嵌入到 PySide6 应用程序中 / In this exercise, we embed the plot into the PySide6 application.

```python
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
```

---

## 设置 / Setup

```python
fig = Figure()
canvas = FigureCanvas(fig)
ax = fig.add_subplot(111)
```

将画布添加到布局 / Add the canvas to a layout:

```python
layout.addWidget(canvas)
```

---

## 更新图表 / Updating the Plot

```python
ax.clear()
ax.plot(x, y)
canvas.draw()
```

重要 / Important:

* ax.clear() 删除旧图表 / removes the old plot
* ax.plot(x, y) 绘制新图表 / draws the new plot
* canvas.draw() 更新可视UI / updates the visible UI

---

# 小示例 — 现板演示 / Mini Examples for Class Demonstration

下面的示例作来意非常效。它们对于实时辞手在构建完整应用是有用的 / The following examples are intentionally small. They are useful for live coding before building the full application.

---

## Mini Example 1 — Button Changes Label Text

```python
import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton

app = QApplication(sys.argv)

window = QWidget()
window.setWindowTitle("Button Example")

layout = QVBoxLayout()
window.setLayout(layout)

label = QLabel("Click the button")
button = QPushButton("Click me")

layout.addWidget(label)
layout.addWidget(button)

def on_button_clicked():
    label.setText("Button was clicked")

button.clicked.connect(on_button_clicked)

window.show()
app.exec()
```

Concepts:

* button
* label update
* signal-slot connection

---

## Mini Example 2 — Dropdown Changes Label Text

```python
import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QComboBox

app = QApplication(sys.argv)

window = QWidget()
window.setWindowTitle("Dropdown Example")

layout = QVBoxLayout()
window.setLayout(layout)

label = QLabel("Select a signal type")
combo = QComboBox()
combo.addItems(["Original", "Filtered", "RMS"])

layout.addWidget(combo)
layout.addWidget(label)

def on_selection_changed():
    label.setText("Selected: " + combo.currentText())

combo.currentIndexChanged.connect(on_selection_changed)

window.show()
app.exec()
```

Concepts:

* dropdown
* selected text
* automatic UI update

---

## Mini Example 3 — Slider Controls a Number

```python
import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QSlider

app = QApplication(sys.argv)

window = QWidget()
window.setWindowTitle("Slider Example")

layout = QVBoxLayout()
window.setLayout(layout)

label = QLabel("Value: 50")
slider = QSlider(Qt.Horizontal)
slider.setMinimum(0)
slider.setMaximum(100)
slider.setValue(50)

layout.addWidget(label)
layout.addWidget(slider)

def on_slider_changed(value):
    label.setText(f"Value: {value}")

slider.valueChanged.connect(on_slider_changed)

window.show()
app.exec()
```

Concepts:

* slider
* signal with a value argument
* dynamic text update

---

## Mini Example 4 — Checkbox Shows or Hides Text

```python
import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QCheckBox

app = QApplication(sys.argv)

window = QWidget()
window.setWindowTitle("Checkbox Example")

layout = QVBoxLayout()
window.setLayout(layout)

checkbox = QCheckBox("Show details")
label = QLabel("These are additional details.")
label.setVisible(False)

layout.addWidget(checkbox)
layout.addWidget(label)

def on_checkbox_toggled(checked):
    label.setVisible(checked)

checkbox.toggled.connect(on_checkbox_toggled)

window.show()
app.exec()
```

Concepts:

* checkbox
* boolean state
* showing and hiding widgets

---

## Mini Example 5 — Input Field and Button

```python
import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton

app = QApplication(sys.argv)

window = QWidget()
window.setWindowTitle("Text Input Example")

layout = QVBoxLayout()
window.setLayout(layout)

input_field = QLineEdit()
input_field.setPlaceholderText("Enter your name")

button = QPushButton("Submit")
label = QLabel("Waiting for input...")

layout.addWidget(input_field)
layout.addWidget(button)
layout.addWidget(label)

def on_submit():
    name = input_field.text()
    label.setText(f"Hello, {name}!")

button.clicked.connect(on_submit)
input_field.returnPressed.connect(on_submit)

window.show()
app.exec()
```

Concepts:

* text input
* button interaction
* Enter key interaction

---

# Full Demo Application — Interactive Signal Viewer

This example combines several UI elements:

* dropdowns
* buttons
* sliders
* checkboxes
* spin boxes
* tabs
* embedded Matplotlib plot
* status label

Students can use this as a reference for what is possible with PySide6.

---

## Complete Example Code

```python
import sys
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSlider,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QLineEdit,
    QTextEdit,
    QListWidget,
    QTableWidget,
    QTableWidgetItem,
    QProgressBar,
    QTabWidget,
    QMessageBox,
    QFileDialog,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class SignalViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Interactive Signal Viewer")
        self.resize(1100, 750)

        self.colors = ["tab:blue", "tab:red", "tab:green", "tab:orange", "tab:purple"]
        self.color_index = 0

        self.x = np.linspace(0, 2 * np.pi, 1000)

        self.setup_ui()
        self.connect_signals()
        self.update_plot()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.plot_page = QWidget()
        self.settings_page = QWidget()
        self.results_page = QWidget()

        self.tabs.addTab(self.plot_page, "Plot")
        self.tabs.addTab(self.settings_page, "Settings")
        self.tabs.addTab(self.results_page, "Results")

        self.setup_plot_page()
        self.setup_settings_page()
        self.setup_results_page()

    def setup_plot_page(self):
        layout = QVBoxLayout()
        self.plot_page.setLayout(layout)

        controls_layout = QHBoxLayout()

        self.channel_combo = QComboBox()
        self.channel_combo.addItems([f"Channel {i}" for i in range(1, 9)])

        self.signal_combo = QComboBox()
        self.signal_combo.addItems(["Original", "Filtered", "RMS", "Absolute"])

        self.color_button = QPushButton("Change Color")
        self.reset_button = QPushButton("Reset")
        self.info_button = QPushButton("Info")
        self.open_file_button = QPushButton("Open File")

        controls_layout.addWidget(QLabel("Channel:"))
        controls_layout.addWidget(self.channel_combo)
        controls_layout.addWidget(QLabel("Signal:"))
        controls_layout.addWidget(self.signal_combo)
        controls_layout.addWidget(self.color_button)
        controls_layout.addWidget(self.reset_button)
        controls_layout.addWidget(self.info_button)
        controls_layout.addWidget(self.open_file_button)

        layout.addLayout(controls_layout)

        parameter_layout = QHBoxLayout()

        self.amplitude_slider = QSlider(Qt.Horizontal)
        self.amplitude_slider.setMinimum(1)
        self.amplitude_slider.setMaximum(100)
        self.amplitude_slider.setValue(50)

        self.frequency_spinbox = QDoubleSpinBox()
        self.frequency_spinbox.setMinimum(0.1)
        self.frequency_spinbox.setMaximum(20.0)
        self.frequency_spinbox.setValue(1.0)
        self.frequency_spinbox.setSingleStep(0.1)
        self.frequency_spinbox.setSuffix(" Hz")

        self.noise_spinbox = QDoubleSpinBox()
        self.noise_spinbox.setMinimum(0.0)
        self.noise_spinbox.setMaximum(1.0)
        self.noise_spinbox.setValue(0.1)
        self.noise_spinbox.setSingleStep(0.05)

        self.grid_checkbox = QCheckBox("Show grid")
        self.grid_checkbox.setChecked(True)

        self.markers_checkbox = QCheckBox("Show markers")
        self.markers_checkbox.setChecked(False)

        parameter_layout.addWidget(QLabel("Amplitude:"))
        parameter_layout.addWidget(self.amplitude_slider)
        parameter_layout.addWidget(QLabel("Frequency:"))
        parameter_layout.addWidget(self.frequency_spinbox)
        parameter_layout.addWidget(QLabel("Noise:"))
        parameter_layout.addWidget(self.noise_spinbox)
        parameter_layout.addWidget(self.grid_checkbox)
        parameter_layout.addWidget(self.markers_checkbox)

        layout.addLayout(parameter_layout)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

    def setup_settings_page(self):
        layout = QGridLayout()
        self.settings_page.setLayout(layout)

        self.participant_input = QLineEdit()
        self.participant_input.setPlaceholderText("Enter participant name")

        self.trial_spinbox = QSpinBox()
        self.trial_spinbox.setMinimum(1)
        self.trial_spinbox.setMaximum(100)
        self.trial_spinbox.setValue(1)

        self.notes_edit = QTextEdit()
        self.notes_edit.setPlaceholderText("Write experiment notes here...")

        layout.addWidget(QLabel("Participant:"), 0, 0)
        layout.addWidget(self.participant_input, 0, 1)
        layout.addWidget(QLabel("Trial:"), 1, 0)
        layout.addWidget(self.trial_spinbox, 1, 1)
        layout.addWidget(QLabel("Notes:"), 2, 0)
        layout.addWidget(self.notes_edit, 2, 1)

    def setup_results_page(self):
        layout = QVBoxLayout()
        self.results_page.setLayout(layout)

        self.trial_list = QListWidget()
        self.trial_list.addItems(["Trial 1", "Trial 2", "Trial 3"])

        self.results_table = QTableWidget()
        self.results_table.setRowCount(8)
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Channel", "Mean", "RMS"])

        for row in range(8):
            self.results_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            self.results_table.setItem(row, 1, QTableWidgetItem("0.00"))
            self.results_table.setItem(row, 2, QTableWidgetItem("0.00"))

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)

        layout.addWidget(QLabel("Recorded trials:"))
        layout.addWidget(self.trial_list)
        layout.addWidget(QLabel("Results:"))
        layout.addWidget(self.results_table)
        layout.addWidget(QLabel("Processing progress:"))
        layout.addWidget(self.progress_bar)

    def connect_signals(self):
        self.channel_combo.currentIndexChanged.connect(self.update_plot)
        self.signal_combo.currentIndexChanged.connect(self.update_plot)
        self.amplitude_slider.valueChanged.connect(self.update_plot)
        self.frequency_spinbox.valueChanged.connect(self.update_plot)
        self.noise_spinbox.valueChanged.connect(self.update_plot)
        self.grid_checkbox.toggled.connect(self.update_plot)
        self.markers_checkbox.toggled.connect(self.update_plot)

        self.color_button.clicked.connect(self.change_color)
        self.reset_button.clicked.connect(self.reset_controls)
        self.info_button.clicked.connect(self.show_info)
        self.open_file_button.clicked.connect(self.open_file)

        self.participant_input.textChanged.connect(self.update_status)
        self.trial_spinbox.valueChanged.connect(self.update_status)
        self.trial_list.currentItemChanged.connect(self.trial_selected)

    def generate_signal(self):
        channel = self.channel_combo.currentIndex() + 1
        amplitude = self.amplitude_slider.value() / 50
        frequency = self.frequency_spinbox.value()
        noise_level = self.noise_spinbox.value()

        signal = amplitude * np.sin(frequency * self.x + channel * 0.4)
        noise = noise_level * np.random.randn(len(self.x))
        signal = signal + noise

        signal_type = self.signal_combo.currentText()

        if signal_type == "Filtered":
            kernel = np.ones(25) / 25
            signal = np.convolve(signal, kernel, mode="same")
        elif signal_type == "RMS":
            window_size = 50
            squared = signal ** 2
            kernel = np.ones(window_size) / window_size
            signal = np.sqrt(np.convolve(squared, kernel, mode="same"))
        elif signal_type == "Absolute":
            signal = np.abs(signal)

        return signal

    def update_plot(self):
        y = self.generate_signal()
        color = self.colors[self.color_index]

        self.ax.clear()

        if self.markers_checkbox.isChecked():
            self.ax.plot(self.x, y, color=color, marker="o", markevery=50)
        else:
            self.ax.plot(self.x, y, color=color)

        self.ax.set_title(
            f"{self.channel_combo.currentText()} — {self.signal_combo.currentText()}"
        )
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(self.grid_checkbox.isChecked())

        self.canvas.draw()

        self.update_results_table(y)
        self.status_label.setText(
            f"Updated plot: {self.channel_combo.currentText()}, {self.signal_combo.currentText()}"
        )

    def update_results_table(self, y):
        selected_channel = self.channel_combo.currentIndex()
        mean_value = np.mean(y)
        rms_value = np.sqrt(np.mean(y ** 2))

        self.results_table.setItem(
            selected_channel,
            1,
            QTableWidgetItem(f"{mean_value:.3f}")
        )
        self.results_table.setItem(
            selected_channel,
            2,
            QTableWidgetItem(f"{rms_value:.3f}")
        )

        progress = int((selected_channel + 1) / 8 * 100)
        self.progress_bar.setValue(progress)

    def change_color(self):
        self.color_index = (self.color_index + 1) % len(self.colors)
        self.update_plot()

    def reset_controls(self):
        self.channel_combo.setCurrentIndex(0)
        self.signal_combo.setCurrentIndex(0)
        self.amplitude_slider.setValue(50)
        self.frequency_spinbox.setValue(1.0)
        self.noise_spinbox.setValue(0.1)
        self.grid_checkbox.setChecked(True)
        self.markers_checkbox.setChecked(False)
        self.status_label.setText("Controls reset")
        self.update_plot()

    def show_info(self):
        QMessageBox.information(
            self,
            "About this app",
            "This demo shows common PySide6 widgets and an embedded Matplotlib plot."
        )

    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Signal File",
            "",
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)"
        )

        if filename:
            self.status_label.setText(f"Selected file: {filename}")

    def update_status(self):
        participant = self.participant_input.text()
        trial = self.trial_spinbox.value()

        if participant:
            self.status_label.setText(f"Participant: {participant}, Trial: {trial}")
        else:
            self.status_label.setText(f"Trial: {trial}")

    def trial_selected(self):
        item = self.trial_list.currentItem()
        if item is not None:
            self.status_label.setText(f"Selected {item.text()}")


app = QApplication(sys.argv)
window = SignalViewer()
window.show()
app.exec()
```

---

# 你的任务 / Your Task in This Exercise

你将学会不断完成 TODO 部分来实现界面 / You will implement the UI by completing the TODOs.

---

## 基本任务 / Basic Task

### 1. 窗口设置 / Window Setup

* 设置窗口标题 / Set the window title
* 设置窗口大小 / Set the window size

---

### 2. 中央部件 / Central Widget

* 创建一个 QWidget / Create a QWidget
* 用 setCentralWidget() 绑定 / Assign it using setCentralWidget()

---

### 3. 布局 / Layouts

创建 / Create:

* QVBoxLayout → 主布局 / main layout
* QHBoxLayout → 控件布局 / control layout

---

### 4. 通道选择 / Channel Selection

创建 / Create:

* QLabel("Channel:")
* QComboBox()

用以下内容填充下拉菜单 / Fill the dropdown with:

```text
Channel 1, Channel 2, Channel 3, ...
```

---

### 5. 信号类型选择 / Signal Selection

用以下内容的下拉菜单 / Create a dropdown with:

```text
Original, Filtered, RMS
```

---

### 6. 按钮 / Button

创建 / Create:

```python
QPushButton("Change Color")
```

---

### 7. 布局组成 / Layout Assembly

* 将所有控件添加到水平布局 / Add all controls to the horizontal layout
* 将水平布局添加到竖直布局 / Add the horizontal layout to the vertical layout

---

### 8. Matplotlib 集成 / Matplotlib Integration

创建 / Create:

* Figure
* FigureCanvas
* Axes

将画布添加到布局 / Add the canvas to the layout.

---

### 9. 连接信号 / Connect Signals

连接 / Connect:

```python
channel_combo.currentIndexChanged.connect(update_plot)
signal_combo.currentIndexChanged.connect(update_plot)
color_button.clicked.connect(change_color)
```

---

### 10. 初始图表 / Initial Plot

调用 / Call:

```python
update_plot()
```

一次，在设置的最后 / once at the end of the setup.

---

# 可选的扩展任务 / Optional Extension Tasks

互成綂整的学生可以下加更多部件 / Students who finish early can add more widgets.

---

## 扩展 1 — 添加滑块 / Extension 1 — Add a Slider

添加幅度滑块 / Add an amplitude slider:

```python
amplitude_slider = QSlider(Qt.Horizontal)
amplitude_slider.setMinimum(1)
amplitude_slider.setMaximum(100)
amplitude_slider.setValue(50)
```

连接到 / Connect it to:

```python
amplitude_slider.valueChanged.connect(update_plot)
```

使用滑块值来比例伸缩信号幅度 / Use the slider value to scale the signal amplitude.

---

## 扩展 2 — 添加复选框 / Extension 2 — Add a Checkbox

添加复选框 / Add a checkbox:

```python
grid_checkbox = QCheckBox("Show grid")
grid_checkbox.setChecked(True)
```

连接到 / Connect it to:

```python
grid_checkbox.toggled.connect(update_plot)
```

在 update_plot() 内部使用它 / Use it inside update_plot():

```python
ax.grid(grid_checkbox.isChecked())
```

---

## 扩展 3 — 添加频率旋转框 / Extension 3 — Add a Frequency Spin Box

添加 / Add:

```python
frequency_spinbox = QDoubleSpinBox()
frequency_spinbox.setMinimum(0.1)
frequency_spinbox.setMaximum(20.0)
frequency_spinbox.setValue(1.0)
frequency_spinbox.setSuffix(" Hz")
```

使用此值改变已绘制信号的频率 / Use this value to change the plotted signal frequency.

---

## 扩展 4 — 添加重置按钮 / Extension 4 — Add a Reset Button

添加 / Add:

```python
reset_button = QPushButton("Reset")
```

然后创建 / Then create:

```python
def reset_controls():
    channel_combo.setCurrentIndex(0)
    signal_combo.setCurrentIndex(0)
    amplitude_slider.setValue(50)
```

连接 / Connect:

```python
reset_button.clicked.connect(reset_controls)
```

---

## 扩展 5 — 添加状态标签 / Extension 5 — Add a Status Label

在下方添加一个标签 / Add a label at the bottom:

```python
status_label = QLabel("Ready")
```

每当有所改变时更新它 / Update it whenever something changes:

```python
status_label.setText("Plot updated")
```

---

## 扩展 6 — 添加选项卡 / Extension 6 — Add Tabs

为以下部分添加选项卡 / Create tabs for:

* 图表 / Plot
* 设置 / Settings
* 结果 / Results

```python
tabs = QTabWidget()
tabs.addTab(plot_page, "Plot")
tabs.addTab(settings_page, "Settings")
tabs.addTab(results_page, "Results")
```

---

## 扩展 7 — 添加表格 / Extension 7 — Add a Table

使用表格来散布计算值 / Use a table to display calculated values:

* 通道号 / channel number
* 平均值 / mean
* RMS / RMS

```python
table = QTableWidget()
table.setRowCount(8)
table.setColumnCount(3)
table.setHorizontalHeaderLabels(["Channel", "Mean", "RMS"])
```

---

## 扩展 8 — 添加文件对话框 / Extension 8 — Add a File Dialog

添加一个打开文件对话框的按钮 / Add a button that opens a file dialog:

```python
filename, _ = QFileDialog.getOpenFileName(
    window,
    "Open File",
    "",
    "CSV Files (*.csv);;All Files (*)"
)
```

---

# 建议的教学顺序 / Suggested Teaching Sequence

对于 90 分钟的课程 / For a 90-minute lesson:

| 时间 / Time | 标简 / Topic |
| ---- | ----- |
| 0–10 分 / min | 为什么要 GUI 编程？脚本对比应用 / Why GUI programming? Scripts vs applications |
| 10–20 分 / min | 最小 PySide6 应用 / Minimal PySide6 application |
| 20–35 分 / min | 布局：竖直、水平、网格 / Layouts: vertical, horizontal, grid |
| 35–50 分 / min | 按钮、标签、下拉菜单、信号和槽 / Buttons, labels, dropdowns, signals and slots |
| 50–65 分 / min | 滑块、复选框、旋转框 / Sliders, checkboxes, spin boxes |
| 65–80 分 / min | 嵌入式 Matplotlib 图表 / Embedded Matplotlib plot |
| 80–90 分 / min | 扩展椀法和学生实验 / Extension ideas and student experimentation |


对于较短的课程，仅专注于 / For a shorter lesson, focus only on:

1. 最小应用 / minimal app
2. 布局 / layouts
3. 按钮 / button
4. 下拉菜单 / dropdown
5. 嵌入式图表 / embedded plot

---

# 常见错误 / Common Mistakes

## 错误 1 — 忘记 app.exec() / Mistake 1 — Forgetting app.exec()

没有这个，窗口可能会打开后立即关闭Without this, the window may open and immediately close.

---

## 错误 2 — 直接添加布局到 QMainWindow / Mistake 2 — Adding a Layout Directly to QMainWindow

错误 / Wrong:

```python
window.setLayout(layout)
```

正确 / Correct:

```python
central_widget = QWidget()
window.setCentralWidget(central_widget)
central_widget.setLayout(layout)
```

---

## 错误 3 — 调用函数而非连接扇 / Mistake 3 — Calling the Function Instead of Connecting It

错误 / Wrong:

```python
button.clicked.connect(update_plot())
```

正确 / Correct:

```python
button.clicked.connect(update_plot)
```

第一种写法会立即调用函数。第二种写法是将函数连接到按钮点击。The first version calls the function immediately.
The second version connects the function to the button click.

---

## 错误 4 — 忘记 canvas.draw() / Mistake 4 — Forgetting canvas.draw()

如果图表不更新，检查你是否调用了 / If the plot does not update, check whether you called:

```python
canvas.draw()
```

---

## 错误 5 — 使用本地变量候后应求 / Mistake 5 — Using Local Variables That Are Needed Later

如果一个部件之后需要在另一个方法中使用，将它存保为 self.部件名 / If a widget is needed in another method, store it as self.widget_name.

示例 / Example:

```python
self.channel_combo = QComboBox()
```

而非 / instead of:

```python
channel_combo = QComboBox()
```

---

# 关键要点 / Key Takeaways

* GUI 应用程序保持打开並等待用户交互 / A GUI application stays open and waits for user interaction.
* PySide6 应用是事件驱动的 / PySide6 applications are event-driven.
* 部件是可见的 UI 元素 / Widgets are the visible UI elements.
* 布局定义部件的排列方式 / Layouts define how widgets are arranged.
* 信号和槽将用户动作连接到 Python 函数 / Signals and slots connect user actions to Python functions.
* Matplotlib 图表可以直接嵌入 PySide6 应用 / Matplotlib plots can be embedded directly into PySide6 applications.
* 小部件可以组合成强大的应用 / Small widgets can be combined into powerful applications.

最重要的思维模型 / Most important mental model:

```text
WAIT -> EVENT -> FUNCTION -> UPDATE UI
```
