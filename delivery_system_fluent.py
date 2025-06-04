"""
快递配送系统 - 使用QFluentWidgets美化界面
实现功能：
1. 分治法递归分区客户（每组5人）
2. 贪心算法优化配送路线
3. 可视化展示分区和路线
4. 计算总运输距离
"""

import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib

matplotlib.use("QtAgg")
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLabel,
    QSpinBox,
    QPushButton,
    QTextEdit,
    QGroupBox,
    QGridLayout,
    QSplitter,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from qfluentwidgets import (
    PushButton,
    SpinBox,
    TextEdit,
    TitleLabel,
    SubtitleLabel,
    BodyLabel,
    setTheme,
    Theme,
    FluentIcon,
    InfoBar,
    InfoBarPosition,
)

# 设置matplotlib支持中文
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class Customer:
    """客户类"""

    def __init__(self, x, y, customer_id):
        self.x = x
        self.y = y
        self.id = customer_id
        self.cargo_weight = random.randint(1, 10)  # 货物重量1-10kg

    def distance_to(self, other):
        """计算到另一个点的距离"""
        if isinstance(other, Customer):
            return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
        else:  # 假设other是(x, y)元组
            return math.sqrt((self.x - other[0]) ** 2 + (self.y - other[1]) ** 2)


class DeliveryZone:
    """配送区域类"""

    def __init__(self, customers, zone_id):
        self.customers = customers
        self.zone_id = zone_id
        self.route = []
        self.total_distance = 0
        self.total_weight = sum(c.cargo_weight for c in customers)

    def calculate_greedy_route(self):
        """使用贪心算法计算最优路线"""
        if not self.customers:
            return

        # 从原点(0,0)开始
        current_pos = (0, 0)
        unvisited = self.customers.copy()
        route = [current_pos]
        total_distance = 0

        while unvisited:
            # 找到距离当前位置最近的客户
            nearest_customer = min(unvisited, key=lambda c: c.distance_to(current_pos))

            # 计算距离并添加到路线
            distance = nearest_customer.distance_to(current_pos)
            total_distance += distance

            route.append((nearest_customer.x, nearest_customer.y))
            current_pos = (nearest_customer.x, nearest_customer.y)
            unvisited.remove(nearest_customer)

        # 返回原点
        return_distance = math.sqrt(current_pos[0] ** 2 + current_pos[1] ** 2)
        total_distance += return_distance
        route.append((0, 0))

        self.route = route
        self.total_distance = total_distance
        return total_distance


class DeliveryAlgorithm:
    """配送算法类"""

    @staticmethod
    def divide_customers(customers, group_size=5):
        """使用分治法递归分区客户"""
        if len(customers) <= group_size:
            return [customers]

        # 按x坐标排序
        customers_sorted = sorted(customers, key=lambda c: c.x)
        mid = len(customers_sorted) // 2

        # 递归分治
        left_groups = DeliveryAlgorithm.divide_customers(
            customers_sorted[:mid], group_size
        )
        right_groups = DeliveryAlgorithm.divide_customers(
            customers_sorted[mid:], group_size
        )

        return left_groups + right_groups

    @staticmethod
    def optimize_routes(zones):
        """优化所有区域的配送路线"""
        total_distance = 0
        for zone in zones:
            distance = zone.calculate_greedy_route()
            total_distance += distance
        return total_distance


class VisualizationWidget(QWidget):
    """可视化组件"""

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.zones = []

    def init_ui(self):
        layout = QVBoxLayout()

        # 创建matplotlib图形
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def plot_delivery_system(self, zones):
        """绘制配送系统图"""
        self.zones = zones
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # 颜色列表
        colors = [
            "red",
            "blue",
            "green",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]

        # 绘制原点
        ax.plot(0, 0, "ko", markersize=12, label="配送中心")

        # 绘制每个区域
        for i, zone in enumerate(zones):
            color = colors[i % len(colors)]

            # 绘制客户点
            x_coords = [c.x for c in zone.customers]
            y_coords = [c.y for c in zone.customers]
            ax.scatter(
                x_coords,
                y_coords,
                c=color,
                s=100,
                label=f"区域{zone.zone_id} (重量:{zone.total_weight}kg)",
            )

            # 绘制客户ID
            for customer in zone.customers:
                ax.annotate(
                    f"C{customer.id}",
                    (customer.x, customer.y),
                    xytext=(5, 5),
                    textcoords="offset points",
                )

            # 绘制配送路线
            if zone.route:
                route_x = [point[0] for point in zone.route]
                route_y = [point[1] for point in zone.route]
                ax.plot(route_x, route_y, color=color, linewidth=2, alpha=0.7)

        ax.set_xlabel("X坐标")
        ax.set_ylabel("Y坐标")
        ax.set_title("快递配送系统 - 分区配送路线图")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        self.figure.tight_layout()
        self.canvas.draw()


class CalculationThread(QThread):
    """计算线程"""

    finished = pyqtSignal(list, float)

    def __init__(self, num_customers):
        super().__init__()
        self.num_customers = num_customers

    def run(self):
        # 生成随机客户
        customers = []
        for i in range(self.num_customers):
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            customers.append(Customer(x, y, i + 1))

        # 使用分治法分区
        customer_groups = DeliveryAlgorithm.divide_customers(customers, 5)

        # 创建配送区域
        zones = []
        for i, group in enumerate(customer_groups):
            zone = DeliveryZone(group, i + 1)
            zones.append(zone)

        # 优化路线
        total_distance = DeliveryAlgorithm.optimize_routes(zones)

        self.finished.emit(zones, total_distance)


class DeliverySystemWindow(QMainWindow):
    """主窗口类"""

    def __init__(self):
        super().__init__()
        self.zones = []
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("智能快递配送系统")
        self.setGeometry(100, 100, 1400, 800)

        # 设置主题
        setTheme(Theme.AUTO)

        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # 创建左侧控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)

        # 创建右侧可视化区域
        self.visualization = VisualizationWidget()
        main_layout.addWidget(self.visualization)

        # 设置布局比例
        main_layout.setStretch(0, 1)  # 控制面板
        main_layout.setStretch(1, 3)  # 可视化区域

    def create_control_panel(self):
        """创建控制面板"""
        panel = QWidget()
        panel.setMaximumWidth(350)
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # 标题
        title = TitleLabel("快递配送系统")
        layout.addWidget(title)

        # 参数设置组
        param_group = QGroupBox("参数设置")
        param_layout = QGridLayout()
        param_group.setLayout(param_layout)

        # 客户数量设置
        param_layout.addWidget(BodyLabel("客户数量:"), 0, 0)
        self.customer_count = SpinBox()
        self.customer_count.setRange(20, 100)
        self.customer_count.setValue(30)
        param_layout.addWidget(self.customer_count, 0, 1)

        layout.addWidget(param_group)

        # 操作按钮
        self.generate_btn = PushButton("生成配送方案", icon=FluentIcon.PLAY)
        self.generate_btn.clicked.connect(self.generate_delivery_plan)
        layout.addWidget(self.generate_btn)

        self.analyze_btn = PushButton("算法对比分析", icon=FluentIcon.CAFE)
        self.analyze_btn.clicked.connect(self.analyze_algorithms)
        layout.addWidget(self.analyze_btn)

        # 结果显示区域
        result_group = QGroupBox("结果信息")
        result_layout = QVBoxLayout()
        result_group.setLayout(result_layout)

        self.result_text = TextEdit()
        self.result_text.setMaximumHeight(300)
        result_layout.addWidget(self.result_text)

        layout.addWidget(result_group)

        # 添加弹性空间
        layout.addStretch()

        return panel

    def generate_delivery_plan(self):
        """生成配送方案"""
        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("计算中...")

        # 创建计算线程
        self.calc_thread = CalculationThread(self.customer_count.value())
        self.calc_thread.finished.connect(self.on_calculation_finished)
        self.calc_thread.start()

    def on_calculation_finished(self, zones, total_distance):
        """计算完成回调"""
        self.zones = zones

        # 更新可视化
        self.visualization.plot_delivery_system(zones)

        # 更新结果显示
        result_text = f"=== 配送方案结果 ===\n\n"
        result_text += f"总客户数: {sum(len(zone.customers) for zone in zones)}\n"
        result_text += f"分区数量: {len(zones)}\n"
        result_text += f"总运输距离: {total_distance:.2f}\n\n"

        for zone in zones:
            result_text += f"区域 {zone.zone_id}:\n"
            result_text += f"  客户数: {len(zone.customers)}\n"
            result_text += f"  总重量: {zone.total_weight}kg\n"
            result_text += f"  配送距离: {zone.total_distance:.2f}\n"
            result_text += (
                f"  客户: {', '.join([f'C{c.id}' for c in zone.customers])}\n\n"
            )

        self.result_text.setPlainText(result_text)

        # 恢复按钮
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("生成配送方案")

        # 显示成功消息
        InfoBar.success(
            title="成功",
            content=f"已生成配送方案，总距离: {total_distance:.2f}",
            orient=Qt.Orientation.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=3000,
            parent=self,
        )

    def analyze_algorithms(self):
        """算法对比分析"""
        if not self.zones:
            InfoBar.warning(
                title="提示",
                content="请先生成配送方案",
                orient=Qt.Orientation.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self,
            )
            return

        # 这里可以实现不同算法的对比
        analysis_text = "=== 算法分析对比 ===\n\n"
        analysis_text += "当前使用算法:\n"
        analysis_text += "1. 分治法 - 客户分区 (每组5人)\n"
        analysis_text += "2. 贪心算法 - 路径优化\n\n"
        analysis_text += "算法特点:\n"
        analysis_text += "• 分治法确保负载均衡\n"
        analysis_text += "• 贪心算法快速求解近似最优解\n"
        analysis_text += "• 时间复杂度: O(n log n)\n"
        analysis_text += "• 适用于大规模配送问题\n\n"

        # 计算一些统计信息
        distances = [zone.total_distance for zone in self.zones]
        analysis_text += f"距离统计:\n"
        analysis_text += f"• 最短路线: {min(distances):.2f}\n"
        analysis_text += f"• 最长路线: {max(distances):.2f}\n"
        analysis_text += f"• 平均距离: {np.mean(distances):.2f}\n"
        analysis_text += f"• 标准差: {np.std(distances):.2f}\n"

        self.result_text.setPlainText(analysis_text)


def main():
    """主函数"""
    app = QApplication(sys.argv)

    # 设置应用程序信息
    app.setApplicationName("智能快递配送系统")
    app.setApplicationVersion("1.0")

    window = DeliverySystemWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
