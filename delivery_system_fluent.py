# -*- coding: gb2312 -*-
"""
�������ϵͳ - ʹ��QFluentWidgets��������
ʵ�ֹ��ܣ�
1. ���η��ݹ�����ͻ���ÿ��5�ˣ�
2. ̰���㷨�Ż�����·��
3. ���ӻ�չʾ������·��
4. �������������
"""

import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('QtAgg')
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QLabel, QSpinBox, QPushButton, QTextEdit,
                            QGroupBox, QGridLayout, QSplitter)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from qfluentwidgets import (PushButton, SpinBox, TextEdit, TitleLabel, 
                           SubtitleLabel, BodyLabel, setTheme, Theme,
                           FluentIcon, InfoBar, InfoBarPosition)

# ����matplotlib֧������
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Customer:
    """�ͻ���"""
    def __init__(self, x, y, customer_id):
        self.x = x
        self.y = y
        self.id = customer_id
        self.cargo_weight = random.randint(1, 10)  # ��������1-10kg
    
    def distance_to(self, other):
        """���㵽��һ����ľ���"""
        if isinstance(other, Customer):
            return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
        else:  # ����other��(x, y)Ԫ��
            return math.sqrt((self.x - other[0])**2 + (self.y - other[1])**2)

class DeliveryZone:
    """����������"""
    def __init__(self, customers, zone_id):
        self.customers = customers
        self.zone_id = zone_id
        self.route = []
        self.total_distance = 0
        self.total_weight = sum(c.cargo_weight for c in customers)
    
    def calculate_greedy_route(self):
        """ʹ��̰���㷨��������·��"""
        if not self.customers:
            return
        
        # ��ԭ��(0,0)��ʼ
        current_pos = (0, 0)
        unvisited = self.customers.copy()
        route = [current_pos]
        total_distance = 0
        
        while unvisited:
            # �ҵ����뵱ǰλ������Ŀͻ�
            nearest_customer = min(unvisited, 
                                 key=lambda c: c.distance_to(current_pos))
            
            # ������벢��ӵ�·��
            distance = nearest_customer.distance_to(current_pos)
            total_distance += distance
            
            route.append((nearest_customer.x, nearest_customer.y))
            current_pos = (nearest_customer.x, nearest_customer.y)
            unvisited.remove(nearest_customer)
        
        # ����ԭ��
        return_distance = math.sqrt(current_pos[0]**2 + current_pos[1]**2)
        total_distance += return_distance
        route.append((0, 0))
        
        self.route = route
        self.total_distance = total_distance
        return total_distance

class DeliveryAlgorithm:
    """�����㷨��"""
    
    @staticmethod
    def divide_customers(customers, group_size=5):
        """ʹ�÷��η��ݹ�����ͻ�"""
        if len(customers) <= group_size:
            return [customers]
        
        # ��x��������
        customers_sorted = sorted(customers, key=lambda c: c.x)
        mid = len(customers_sorted) // 2
        
        # �ݹ����
        left_groups = DeliveryAlgorithm.divide_customers(
            customers_sorted[:mid], group_size)
        right_groups = DeliveryAlgorithm.divide_customers(
            customers_sorted[mid:], group_size)
        
        return left_groups + right_groups
    
    @staticmethod
    def optimize_routes(zones):
        """�Ż��������������·��"""
        total_distance = 0
        for zone in zones:
            distance = zone.calculate_greedy_route()
            total_distance += distance
        return total_distance

class VisualizationWidget(QWidget):
    """���ӻ����"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.zones = []
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # ����matplotlibͼ��
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
    
    def plot_delivery_system(self, zones):
        """��������ϵͳͼ"""
        self.zones = zones
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # ��ɫ�б�
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 
                 'pink', 'gray', 'olive', 'cyan']
        
        # ����ԭ��
        ax.plot(0, 0, 'ko', markersize=12, label='��������')
        
        # ����ÿ������
        for i, zone in enumerate(zones):
            color = colors[i % len(colors)]
            
            # ���ƿͻ���
            x_coords = [c.x for c in zone.customers]
            y_coords = [c.y for c in zone.customers]
            ax.scatter(x_coords, y_coords, c=color, s=100, 
                      label=f'����{zone.zone_id} (����:{zone.total_weight}kg)')
            
            # ���ƿͻ�ID
            for customer in zone.customers:
                ax.annotate(f'C{customer.id}', 
                           (customer.x, customer.y), 
                           xytext=(5, 5), textcoords='offset points')
            
            # ��������·��
            if zone.route:
                route_x = [point[0] for point in zone.route]
                route_y = [point[1] for point in zone.route]
                ax.plot(route_x, route_y, color=color, linewidth=2, alpha=0.7)
        
        ax.set_xlabel('X����')
        ax.set_ylabel('Y����')
        ax.set_title('�������ϵͳ - ��������·��ͼ')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        self.figure.tight_layout()
        self.canvas.draw()

class CalculationThread(QThread):
    """�����߳�"""
    finished = pyqtSignal(list, float)
    
    def __init__(self, num_customers):
        super().__init__()
        self.num_customers = num_customers
    
    def run(self):
        # ��������ͻ�
        customers = []
        for i in range(self.num_customers):
            x = random.randint(-50, 50)
            y = random.randint(-50, 50)
            customers.append(Customer(x, y, i+1))
        
        # ʹ�÷��η�����
        customer_groups = DeliveryAlgorithm.divide_customers(customers, 5)
        
        # ������������
        zones = []
        for i, group in enumerate(customer_groups):
            zone = DeliveryZone(group, i+1)
            zones.append(zone)
        
        # �Ż�·��
        total_distance = DeliveryAlgorithm.optimize_routes(zones)
        
        self.finished.emit(zones, total_distance)

class DeliverySystemWindow(QMainWindow):
    """��������"""
    
    def __init__(self):
        super().__init__()
        self.zones = []
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('���ܿ������ϵͳ')
        self.setGeometry(100, 100, 1400, 800)
        
        # ��������
        setTheme(Theme.AUTO)
        
        # �������봰�ڲ���
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # ����������
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # �������������
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        # �����Ҳ���ӻ�����
        self.visualization = VisualizationWidget()
        main_layout.addWidget(self.visualization)
        
        # ���ò��ֱ���
        main_layout.setStretch(0, 1)  # �������
        main_layout.setStretch(1, 3)  # ���ӻ�����
        
    def create_control_panel(self):
        """�����������"""
        panel = QWidget()
        panel.setMaximumWidth(350)
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # ����
        title = TitleLabel('�������ϵͳ')
        layout.addWidget(title)
        
        # ����������
        param_group = QGroupBox('��������')
        param_layout = QGridLayout()
        param_group.setLayout(param_layout)
        
        # �ͻ���������
        param_layout.addWidget(BodyLabel('�ͻ�����:'), 0, 0)
        self.customer_count = SpinBox()
        self.customer_count.setRange(20, 100)
        self.customer_count.setValue(30)
        param_layout.addWidget(self.customer_count, 0, 1)
        
        layout.addWidget(param_group)
        
        # ������ť
        self.generate_btn = PushButton('�������ͷ���', icon=FluentIcon.PLAY)
        self.generate_btn.clicked.connect(self.generate_delivery_plan)
        layout.addWidget(self.generate_btn)
        
        self.analyze_btn = PushButton('�㷨�Աȷ���', icon=FluentIcon.CAFE)
        self.analyze_btn.clicked.connect(self.analyze_algorithms)
        layout.addWidget(self.analyze_btn)
        
        # �����ʾ����
        result_group = QGroupBox('�����Ϣ')
        result_layout = QVBoxLayout()
        result_group.setLayout(result_layout)
        
        self.result_text = TextEdit()
        self.result_text.setMaximumHeight(300)
        result_layout.addWidget(self.result_text)
        
        layout.addWidget(result_group)
        
        # ��ӵ��Կռ�
        layout.addStretch()
        
        return panel
    
    def generate_delivery_plan(self):
        """�������ͷ���"""
        self.generate_btn.setEnabled(False)
        self.generate_btn.setText('������...')
        
        # ���������߳�
        self.calc_thread = CalculationThread(self.customer_count.value())
        self.calc_thread.finished.connect(self.on_calculation_finished)
        self.calc_thread.start()
    
    def on_calculation_finished(self, zones, total_distance):
        """������ɻص�"""
        self.zones = zones
        
        # ���¿��ӻ�
        self.visualization.plot_delivery_system(zones)
        
        # ���½����ʾ
        result_text = f"=== ���ͷ������ ===\n\n"
        result_text += f"�ܿͻ���: {sum(len(zone.customers) for zone in zones)}\n"
        result_text += f"��������: {len(zones)}\n"
        result_text += f"���������: {total_distance:.2f}\n\n"
        
        for zone in zones:
            result_text += f"���� {zone.zone_id}:\n"
            result_text += f"  �ͻ���: {len(zone.customers)}\n"
            result_text += f"  ������: {zone.total_weight}kg\n"
            result_text += f"  ���;���: {zone.total_distance:.2f}\n"
            result_text += f"  �ͻ�: {', '.join([f'C{c.id}' for c in zone.customers])}\n\n"
        
        self.result_text.setPlainText(result_text)
        
        # �ָ���ť
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText('�������ͷ���')
        
        # ��ʾ�ɹ���Ϣ
        InfoBar.success(
            title='�ɹ�',
            content=f'���������ͷ������ܾ���: {total_distance:.2f}',
            orient=Qt.Orientation.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=3000,
            parent=self
        )
    
    def analyze_algorithms(self):
        """�㷨�Աȷ���"""
        if not self.zones:
            InfoBar.warning(
                title='��ʾ',
                content='�����������ͷ���',
                orient=Qt.Orientation.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP,
                duration=2000,
                parent=self
            )
            return
        
        # �������ʵ�ֲ�ͬ�㷨�ĶԱ�
        analysis_text = "=== �㷨�����Ա� ===\n\n"
        analysis_text += "��ǰʹ���㷨:\n"
        analysis_text += "1. ���η� - �ͻ����� (ÿ��5��)\n"
        analysis_text += "2. ̰���㷨 - ·���Ż�\n\n"
        analysis_text += "�㷨�ص�:\n"
        analysis_text += "? ���η�ȷ�����ؾ���\n"
        analysis_text += "? ̰���㷨�������������Ž�\n"
        analysis_text += "? ʱ�临�Ӷ�: O(n log n)\n"
        analysis_text += "? �����ڴ��ģ��������\n\n"
        
        # ����һЩͳ����Ϣ
        distances = [zone.total_distance for zone in self.zones]
        analysis_text += f"����ͳ��:\n"
        analysis_text += f"? ���·��: {min(distances):.2f}\n"
        analysis_text += f"? �·��: {max(distances):.2f}\n"
        analysis_text += f"? ƽ������: {np.mean(distances):.2f}\n"
        analysis_text += f"? ��׼��: {np.std(distances):.2f}\n"
        
        self.result_text.setPlainText(analysis_text)

def main():
    """������"""
    app = QApplication(sys.argv)
    
    # ����Ӧ�ó�����Ϣ
    app.setApplicationName('���ܿ������ϵͳ')
    app.setApplicationVersion('1.0')
    
    window = DeliverySystemWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
