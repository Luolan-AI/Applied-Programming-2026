import sys

from PySide6.QtWidgets import QApplication

from views.mainView import MainView
from viewmodels.mainViewModel import MainViewModel


def main():
    # 创建 Qt 应用对象
    # Create Qt application object
    app = QApplication(sys.argv)

    # TODO 1:
    # 中文：创建 MainViewModel 对象
    # English: Create the ViewModel object.
    view_model = MainViewModel()

    # TODO 2:
    # 中文：创建 MainView 对象，并将 ViewModel 传入其中
    # English: Create the MainView and pass the ViewModel into it.
    view = MainView(view_model)

    # TODO 3:
    # 中文：显示窗口（调用 show() 方法）
    # English: Show the window.
    view.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()