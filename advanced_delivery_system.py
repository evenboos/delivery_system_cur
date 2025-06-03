# -*- coding: gb2312 -*-
"""
高级快递配送系统 - 包含多种算法对比和扩展功能
新增功能：
1. 多种分区算法（分治法、K-means聚类、基于容量限制）
2. 多种路径优化算法（贪心、最近邻、2-opt优化）
3. 货车容量限制
4. 算法性能对比
5. 参数调优
"""

import sys
import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('QtAgg')
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QLabel, QSpinBox, QPushButton, QTextEdit,
                            QGroupBox, QGridLayout, QComboBox, QCheckBox,
                            QTabWidget, QTableWidget, QTableWidgetItem)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from qfluentwidgets import (PushButton, SpinBox, TextEdit, TitleLabel, 
                           SubtitleLabel, BodyLabel, setTheme, Theme,
                           FluentIcon, InfoBar, InfoBarPosition, ComboBox,
                           CheckBox, TableWidget)

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Customer:
    """增强客户类"""
    def __init__(self, x, y, customer_id, cargo_weight=None):
        self.x = x
        self.y = y
        self.id = customer_id
        self.cargo_weight = cargo_weight or random.randint(1, 15)  # 货物重量1-15kg
        self.priority = random.choice(['normal', 'urgent', 'express'])  # 优先级
        self.time_window = (8, 18)  # 配送时间窗口
    
    def distance_to(self, other):
        """计算到另一个点的距离"""
        if isinstance(other, Customer):
            return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
        else:
            return math.sqrt((self.x - other[0])**2 + (self.y - other[1])**2)

class Vehicle:
    """货车类"""
    def __init__(self, vehicle_id, capacity=50):
        self.id = vehicle_id
        self.capacity = capacity  # 容量限制（kg）
        self.current_load = 0
        self.route = [(0, 0)]  # 从原点开始
        self.customers = []
        self.total_distance = 0
    
    def can_add_customer(self, customer):
        """检查是否可以添加客户"""
        return self.current_load + customer.cargo_weight <= self.capacity
    
    def add_customer(self, customer):
        """添加客户到车辆"""
        if self.can_add_customer(customer):
            self.customers.append(customer)
            self.current_load += customer.cargo_weight
            return True
        return False

class PartitionAlgorithm:
    """分区算法类"""
    
    @staticmethod
    def divide_and_conquer(customers, group_size=5):
        """分治法分区"""
        if len(customers) <= group_size:
            return [customers]
        
        customers_sorted = sorted(customers, key=lambda c: c.x)
        mid = len(customers_sorted) // 2
        
        left_groups = PartitionAlgorithm.divide_and_conquer(
            customers_sorted[:mid], group_size)
        right_groups = PartitionAlgorithm.divide_and_conquer(
            customers_sorted[mid:], group_size)
        
        return left_groups + right_groups
    
    @staticmethod
    def kmeans_partition(customers, num_clusters):
        """K-means聚类分区"""
        if len(customers) <= num_clusters:
            return [[c] for c in customers]
        
        # 初始化聚类中心
        centers = random.sample(customers, num_clusters)
        center_coords = [(c.x, c.y) for c in centers]
        
        for _ in range(10):  # 最多迭代10次
            clusters = [[] for _ in range(num_clusters)]
            
            # 分配客户到最近的聚类中心
            for customer in customers:
                distances = [customer.distance_to(center) for center in center_coords]
                closest_center = distances.index(min(distances))
                clusters[closest_center].append(customer)
            
            # 更新聚类中心
            new_centers = []
            for cluster in clusters:
                if cluster:
                    center_x = sum(c.x for c in cluster) / len(cluster)
                    center_y = sum(c.y for c in cluster) / len(cluster)
                    new_centers.append((center_x, center_y))
                else:
                    new_centers.append(center_coords[len(new_centers)])
            
            if new_centers == center_coords:
                break
            center_coords = new_centers
        
        return [cluster for cluster in clusters if cluster]
    
    @staticmethod
    def capacity_based_partition(customers, vehicle_capacity=50):
        """基于容量限制的分区"""
        groups = []
        current_group = []
        current_weight = 0
        
        # 按优先级和重量排序
        sorted_customers = sorted(customers, 
                                key=lambda c: (c.priority != 'urgent', c.cargo_weight))
        
        for customer in sorted_customers:
            if current_weight + customer.cargo_weight <= vehicle_capacity:
                current_group.append(customer)
                current_weight += customer.cargo_weight
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [customer]
                current_weight = customer.cargo_weight
        
        if current_group:
            groups.append(current_group)
        
        return groups

class RouteOptimizer:
    """路径优化算法类"""
    
    @staticmethod
    def greedy_nearest(customers):
        """贪心最近邻算法"""
        if not customers:
            return [], 0
        
        current_pos = (0, 0)
        unvisited = customers.copy()
        route = [current_pos]
        total_distance = 0
        
        while unvisited:
            nearest = min(unvisited, key=lambda c: c.distance_to(current_pos))
            distance = nearest.distance_to(current_pos)
            total_distance += distance
            
            route.append((nearest.x, nearest.y))
            current_pos = (nearest.x, nearest.y)
            unvisited.remove(nearest)
        
        # 返回原点
        return_distance = math.sqrt(current_pos[0]**2 + current_pos[1]**2)
        total_distance += return_distance
        route.append((0, 0))
        
        return route, total_distance
    
    @staticmethod
    def two_opt_improve(route, customers):
        """2-opt优化算法"""
        def calculate_route_distance(route_points):
            distance = 0
            for i in range(len(route_points) - 1):
                p1, p2 = route_points[i], route_points[i + 1]
                distance += math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            return distance
        
        def two_opt_swap(route, i, k):
            new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
            return new_route
        
        if len(route) <= 3:
            return route, calculate_route_distance(route)
        
        best_route = route.copy()
        best_distance = calculate_route_distance(route)
        improved = True
        
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for k in range(i + 1, len(route) - 1):
                    new_route = two_opt_swap(best_route, i, k)
                    new_distance = calculate_route_distance(new_route)
                    
                    if new_distance < best_distance:
                        best_route = new_route
                        best_distance = new_distance
                        improved = True
        
        return best_route, best_distance
    
    @staticmethod
    def dynamic_programming_tsp(customers):
        """动态规划TSP算法 - 保证全局最优解（适用于小规模问题）"""
        if not customers:
            return [], 0
        
        if len(customers) > 10:  # DP算法复杂度过高，限制问题规模
            # 对于大规模问题，回退到贪心算法
            return RouteOptimizer.greedy_nearest(customers)
        
        n = len(customers)
        # 添加原点作为起始点
        points = [(0, 0)] + [(c.x, c.y) for c in customers]
        
        # 计算距离矩阵
        dist = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            for j in range(n + 1):
                if i != j:
                    dist[i][j] = math.sqrt((points[i][0] - points[j][0])**2 + 
                                         (points[i][1] - points[j][1])**2)
        
        # DP状态：dp[mask][i] 表示访问了mask集合中的城市，当前在城市i的最短距离
        dp = {}
        parent = {}
        
        # 初始化：从原点0开始
        dp[(1, 0)] = 0  # mask=1表示只访问了原点
        
        # 填充DP表
        for mask in range(1, 1 << (n + 1)):
            for u in range(n + 1):
                if not (mask & (1 << u)):
                    continue
                
                for v in range(n + 1):
                    if u == v or not (mask & (1 << v)):
                        continue
                    
                    prev_mask = mask ^ (1 << u)
                    if (prev_mask, v) in dp:
                        new_dist = dp[(prev_mask, v)] + dist[v][u]
                        if (mask, u) not in dp or new_dist < dp[(mask, u)]:
                            dp[(mask, u)] = new_dist
                            parent[(mask, u)] = v
        
        # 找到最优解：回到原点
        full_mask = (1 << (n + 1)) - 1
        min_cost = float('inf')
        last_city = -1
        
        for i in range(1, n + 1):
            if (full_mask, i) in dp:
                cost = dp[(full_mask, i)] + dist[i][0]
                if cost < min_cost:
                    min_cost = cost
                    last_city = i
        
        # 重构路径
        if last_city == -1:
            return RouteOptimizer.greedy_nearest(customers)
        
        path = []
        mask = full_mask
        curr = last_city
        
        while (mask, curr) in parent:
            path.append(curr)
            next_curr = parent[(mask, curr)]
            mask ^= (1 << curr)
            curr = next_curr
        
        path.append(0)  # 起始点
        path.reverse()
        
        # 转换为坐标路径
        route = [points[i] for i in path] + [(0, 0)]
        
        return route, min_cost
    
    @staticmethod
    def genetic_algorithm_tsp(customers, population_size=50, generations=100, mutation_rate=0.1):
        """遗传算法TSP"""
        if not customers:
            return [], 0
        
        n = len(customers)
        points = [(0, 0)] + [(c.x, c.y) for c in customers]
        
        def calculate_distance(route):
            """计算路径总距离"""
            total = 0
            for i in range(len(route) - 1):
                p1, p2 = points[route[i]], points[route[i + 1]]
                total += math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            return total
        
        def create_individual():
            """创建个体（随机路径）"""
            route = [0] + list(range(1, n + 1)) + [0]  # 从原点开始和结束
            middle = route[1:-1]
            random.shuffle(middle)
            return route[0:1] + middle + route[-1:]
        
        def crossover(parent1, parent2):
            """交叉操作 - 顺序交叉"""
            size = len(parent1) - 2  # 除去起始和结束的原点
            start, end = sorted(random.sample(range(1, size + 1), 2))
            
            child = [0] * (size + 2)
            child[0] = child[-1] = 0  # 起始和结束都是原点
            
            # 复制父代1的一段
            child[start:end] = parent1[start:end]
            
            # 填充剩余位置
            remaining = [item for item in parent2[1:-1] if item not in child[start:end]]
            j = 0
            for i in range(1, size + 1):
                if child[i] == 0:
                    child[i] = remaining[j]
                    j += 1
            
            return child
        
        def mutate(individual):
            """变异操作 - 交换两个城市"""
            if random.random() < mutation_rate:
                i, j = random.sample(range(1, len(individual) - 1), 2)
                individual[i], individual[j] = individual[j], individual[i]
            return individual
        
        # 初始化种群
        population = [create_individual() for _ in range(population_size)]
        
        # 进化过程
        for generation in range(generations):
            # 计算适应度（距离越短适应度越高）
            fitness_scores = [(1.0 / (calculate_distance(ind) + 1), ind) for ind in population]
            fitness_scores.sort(reverse=True)
            
            # 选择优秀个体
            elite_size = population_size // 4
            new_population = [ind for _, ind in fitness_scores[:elite_size]]
            
            # 生成新个体
            while len(new_population) < population_size:
                # 轮盘赌选择
                total_fitness = sum(score for score, _ in fitness_scores)
                r1 = random.uniform(0, total_fitness)
                r2 = random.uniform(0, total_fitness)
                
                parent1 = parent2 = None
                cumsum = 0
                for score, ind in fitness_scores:
                    cumsum += score
                    if parent1 is None and cumsum >= r1:
                        parent1 = ind
                    if parent2 is None and cumsum >= r2:
                        parent2 = ind
                    if parent1 and parent2:
                        break
                
                if parent1 and parent2:
                    child = crossover(parent1, parent2)
                    child = mutate(child)
                    new_population.append(child)
            
            population = new_population
        
        # 返回最优解
        best_individual = min(population, key=calculate_distance)
        best_distance = calculate_distance(best_individual)
        best_route = [points[i] for i in best_individual]
        
        return best_route, best_distance
    
    @staticmethod
    def ant_colony_optimization(customers, num_ants=20, iterations=50, alpha=1.0, beta=2.0, evaporation=0.5):
        """蚁群算法TSP"""
        if not customers:
            return [], 0
        
        n = len(customers)
        points = [(0, 0)] + [(c.x, c.y) for c in customers]
        
        # 计算距离矩阵
        distances = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            for j in range(n + 1):
                if i != j:
                    distances[i][j] = math.sqrt((points[i][0] - points[j][0])**2 + 
                                              (points[i][1] - points[j][1])**2)
        
        # 初始化信息素矩阵
        pheromones = [[1.0] * (n + 1) for _ in range(n + 1)]
        
        best_route = None
        best_distance = float('inf')
        
        for iteration in range(iterations):
            all_routes = []
            
            # 每只蚂蚁构建路径
            for ant in range(num_ants):
                current = 0  # 从原点开始
                visited = {0}
                route = [0]
                
                # 访问所有客户
                while len(visited) < n + 1:
                    unvisited = [i for i in range(n + 1) if i not in visited]
                    
                    # 计算转移概率
                    probabilities = []
                    for next_city in unvisited:
                        pheromone = pheromones[current][next_city] ** alpha
                        visibility = (1.0 / distances[current][next_city]) ** beta if distances[current][next_city] > 0 else 0
                        probabilities.append(pheromone * visibility)
                    
                    # 轮盘赌选择下一个城市
                    if sum(probabilities) > 0:
                        probabilities = [p / sum(probabilities) for p in probabilities]
                        next_city = np.random.choice(unvisited, p=probabilities)
                    else:
                        next_city = random.choice(unvisited)
                    
                    route.append(next_city)
                    visited.add(next_city)
                    current = next_city
                
                # 回到原点
                route.append(0)
                
                # 计算路径距离
                distance = sum(distances[route[i]][route[i + 1]] for i in range(len(route) - 1))
                all_routes.append((route, distance))
                
                # 更新最优解
                if distance < best_distance:
                    best_distance = distance
                    best_route = route.copy()
            
            # 更新信息素
            # 蒸发
            for i in range(n + 1):
                for j in range(n + 1):
                    pheromones[i][j] *= (1 - evaporation)
            
            # 增强
            for route, distance in all_routes:
                deposit = 1.0 / distance if distance > 0 else 0
                for i in range(len(route) - 1):
                    pheromones[route[i]][route[i + 1]] += deposit
                    pheromones[route[i + 1]][route[i]] += deposit
        
        # 转换为坐标路径
        result_route = [points[i] for i in best_route]
        return result_route, best_distance
    
    @staticmethod
    def simulated_annealing_tsp(customers, initial_temp=1000, cooling_rate=0.95, min_temp=1):
        """模拟退火算法TSP"""
        if not customers:
            return [], 0
        
        n = len(customers)
        points = [(0, 0)] + [(c.x, c.y) for c in customers]
        
        def calculate_distance(route):
            """计算路径总距离"""
            total = 0
            for i in range(len(route) - 1):
                p1, p2 = points[route[i]], points[route[i + 1]]
                total += math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            return total
        
        def get_neighbor(route):
            """生成邻居解 - 随机交换两个城市"""
            new_route = route.copy()
            i, j = random.sample(range(1, len(route) - 1), 2)  # 不包括起始和结束的原点
            new_route[i], new_route[j] = new_route[j], new_route[i]
            return new_route
        
        # 初始解 - 贪心算法生成
        current_route = [0] + list(range(1, n + 1)) + [0]
        current_distance = calculate_distance(current_route)
        
        best_route = current_route.copy()
        best_distance = current_distance
        
        temperature = initial_temp
        
        while temperature > min_temp:
            for _ in range(100):  # 在每个温度下尝试100次
                new_route = get_neighbor(current_route)
                new_distance = calculate_distance(new_route)
                
                # 计算接受概率
                delta = new_distance - current_distance
                if delta < 0 or random.random() < math.exp(-delta / temperature):
                    current_route = new_route
                    current_distance = new_distance
                    
                    # 更新最优解
                    if current_distance < best_distance:
                        best_route = current_route.copy()
                        best_distance = current_distance
            
            temperature *= cooling_rate
        
        # 转换为坐标路径
        result_route = [points[i] for i in best_route]
        return result_route, best_distance

class DeliveryZone:
    """增强配送区域类"""
    def __init__(self, customers, zone_id, vehicle_capacity=50):
        self.customers = customers
        self.zone_id = zone_id
        self.vehicle_capacity = vehicle_capacity
        self.route = []
        self.total_distance = 0
        self.total_weight = sum(c.cargo_weight for c in customers)
        self.algorithm_used = ""
        self.calculation_time = 0
    def optimize_route(self, algorithm='greedy'):
        """使用指定算法优化路线"""
        start_time = time.time()
        
        if algorithm == 'greedy':
            self.route, self.total_distance = RouteOptimizer.greedy_nearest(self.customers)
            self.algorithm_used = "贪心最近邻"
        elif algorithm == '2opt':
            greedy_route, _ = RouteOptimizer.greedy_nearest(self.customers)
            self.route, self.total_distance = RouteOptimizer.two_opt_improve(greedy_route, self.customers)
            self.algorithm_used = "2-opt优化"
        elif algorithm == 'dp':
            self.route, self.total_distance = RouteOptimizer.dynamic_programming_tsp(self.customers)
            self.algorithm_used = "动态规划(DP)"
        elif algorithm == 'ga':
            self.route, self.total_distance = RouteOptimizer.genetic_algorithm_tsp(self.customers)
            self.algorithm_used = "遗传算法(GA)"
        elif algorithm == 'aco':
            self.route, self.total_distance = RouteOptimizer.ant_colony_optimization(self.customers)
            self.algorithm_used = "蚁群算法(ACO)"
        elif algorithm == 'sa':
            self.route, self.total_distance = RouteOptimizer.simulated_annealing_tsp(self.customers)
            self.algorithm_used = "模拟退火(SA)"
        
        self.calculation_time = time.time() - start_time
        return self.total_distance

class AdvancedVisualizationWidget(QWidget):
    """高级可视化组件"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.zones = []
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 创建matplotlib图形
        self.figure = Figure(figsize=(14, 10))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
    
    def plot_advanced_delivery_system(self, zones, algorithm_info=""):
        """绘制高级配送系统图"""
        self.zones = zones
        self.figure.clear()
        
        # 创建子图
        ax1 = self.figure.add_subplot(221)  # 路线图
        ax2 = self.figure.add_subplot(222)  # 容量分布
        ax3 = self.figure.add_subplot(223)  # 距离分析
        ax4 = self.figure.add_subplot(224)  # 算法性能
        
        # 绘制路线图
        self.plot_routes(ax1, zones)
        
        # 绘制容量分布
        self.plot_capacity_distribution(ax2, zones)
        
        # 绘制距离分析
        self.plot_distance_analysis(ax3, zones)
        
        # 绘制算法性能
        self.plot_algorithm_performance(ax4, zones)
        
        self.figure.suptitle(f'高级快递配送系统分析 - {algorithm_info}', fontsize=14)
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_routes(self, ax, zones):
        """绘制配送路线"""
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 
                 'pink', 'gray', 'olive', 'cyan']
        
        ax.plot(0, 0, 'ko', markersize=15, label='配送中心')
        
        for i, zone in enumerate(zones):
            color = colors[i % len(colors)]
            
            # 绘制客户点
            x_coords = [c.x for c in zone.customers]
            y_coords = [c.y for c in zone.customers]
            
            # 根据货物重量调整点的大小
            sizes = [c.cargo_weight * 10 for c in zone.customers]
            ax.scatter(x_coords, y_coords, c=color, s=sizes, alpha=0.7,
                      label=f'区域{zone.zone_id}({zone.total_weight}kg)')
            
            # 绘制配送路线
            if zone.route:
                route_x = [point[0] for point in zone.route]
                route_y = [point[1] for point in zone.route]
                ax.plot(route_x, route_y, color=color, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.set_title('配送路线图')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def plot_capacity_distribution(self, ax, zones):
        """绘制容量分布图"""
        zone_ids = [f'区域{zone.zone_id}' for zone in zones]
        weights = [zone.total_weight for zone in zones]
        
        bars = ax.bar(zone_ids, weights, color='skyblue', alpha=0.7)
        ax.axhline(y=50, color='red', linestyle='--', label='容量限制(50kg)')
        
        # 在柱子上显示数值
        for bar, weight in zip(bars, weights):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{weight}kg', ha='center', va='bottom')
        
        ax.set_ylabel('重量 (kg)')
        ax.set_title('各区域货物重量分布')
        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    def plot_distance_analysis(self, ax, zones):
        """绘制距离分析图"""
        distances = [zone.total_distance for zone in zones]
        zone_ids = [f'区域{zone.zone_id}' for zone in zones]
        
        ax.bar(zone_ids, distances, color='lightgreen', alpha=0.7)
        ax.set_ylabel('距离')
        ax.set_title('各区域配送距离')
        
        # 添加平均线
        avg_distance = np.mean(distances)
        ax.axhline(y=avg_distance, color='red', linestyle='--', 
                  label=f'平均距离: {avg_distance:.1f}')
        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    def plot_algorithm_performance(self, ax, zones):
        """绘制算法性能图"""
        calc_times = [zone.calculation_time * 1000 for zone in zones]  # 转换为毫秒
        zone_ids = [f'区域{zone.zone_id}' for zone in zones]
        
        ax.bar(zone_ids, calc_times, color='orange', alpha=0.7)
        ax.set_ylabel('计算时间 (ms)')
        ax.set_title('算法计算时间')
        plt.setp(ax.get_xticklabels(), rotation=45)

class AdvancedCalculationThread(QThread):
    """高级计算线程"""
    finished = pyqtSignal(list, float, dict)
    
    def __init__(self, num_customers, partition_method, route_method, vehicle_capacity):
        super().__init__()
        self.num_customers = num_customers
        self.partition_method = partition_method
        self.route_method = route_method
        self.vehicle_capacity = vehicle_capacity
    
    def run(self):
        start_time = time.time()
        
        # 生成随机客户
        customers = []
        for i in range(self.num_customers):
            x = random.randint(-60, 60)
            y = random.randint(-60, 60)
            customers.append(Customer(x, y, i+1))
        
        # 选择分区算法
        if self.partition_method == "分治法":
            customer_groups = PartitionAlgorithm.divide_and_conquer(customers, 5)
        elif self.partition_method == "K-means聚类":
            num_clusters = max(1, len(customers) // 5)
            customer_groups = PartitionAlgorithm.kmeans_partition(customers, num_clusters)
        elif self.partition_method == "容量限制":
            customer_groups = PartitionAlgorithm.capacity_based_partition(
                customers, self.vehicle_capacity)
        
        # 创建配送区域并优化路线
        zones = []
        total_distance = 0
        
        for i, group in enumerate(customer_groups):
            if group:  # 确保组不为空
                zone = DeliveryZone(group, i+1, self.vehicle_capacity)
                distance = zone.optimize_route(self.route_method.lower().replace('-', ''))
                total_distance += distance
                zones.append(zone)
        
        # 计算统计信息
        calculation_time = time.time() - start_time
        stats = {
            'total_time': calculation_time,
            'partition_method': self.partition_method,
            'route_method': self.route_method,
            'num_zones': len(zones),
            'avg_distance': total_distance / len(zones) if zones else 0,
            'max_distance': max(zone.total_distance for zone in zones) if zones else 0,
            'min_distance': min(zone.total_distance for zone in zones) if zones else 0
        }
        
        self.finished.emit(zones, total_distance, stats)

class AdvancedDeliverySystemWindow(QMainWindow):
    """高级主窗口类"""
    
    def __init__(self):
        super().__init__()
        self.zones = []
        self.comparison_results = []
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('高级智能快递配送系统 v2.0')
        self.setGeometry(50, 50, 1600, 900)
        
        # 设置主题
        setTheme(Theme.AUTO)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        
        # 创建主要分析标签页
        self.create_main_tab()
        
        # 创建算法对比标签页
        self.create_comparison_tab()
    
    def create_main_tab(self):
        """创建主要分析标签页"""
        main_tab = QWidget()
        self.tab_widget.addTab(main_tab, "主要分析")
        
        layout = QHBoxLayout()
        main_tab.setLayout(layout)
        
        # 左侧控制面板
        control_panel = self.create_advanced_control_panel()
        layout.addWidget(control_panel)
        
        # 右侧可视化区域
        self.visualization = AdvancedVisualizationWidget()
        layout.addWidget(self.visualization)
        
        layout.setStretch(0, 1)
        layout.setStretch(1, 3)
    
    def create_comparison_tab(self):
        """创建算法对比标签页"""
        comparison_tab = QWidget()
        self.tab_widget.addTab(comparison_tab, "算法对比")
        
        layout = QVBoxLayout()
        comparison_tab.setLayout(layout)
        
        # 对比控制按钮
        btn_layout = QHBoxLayout()
        self.compare_btn = PushButton('运行算法对比', icon=FluentIcon.SYNC)
        self.compare_btn.clicked.connect(self.run_algorithm_comparison)
        btn_layout.addWidget(self.compare_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # 对比结果表格
        self.comparison_table = TableWidget()
        self.comparison_table.setColumnCount(8)
        self.comparison_table.setHorizontalHeaderLabels([
            '分区算法', '路径算法', '总距离', '区域数', '平均距离', 
            '最大距离', '最小距离', '计算时间(s)'
        ])
        layout.addWidget(self.comparison_table)
        
        # 对比结果文本
        self.comparison_text = TextEdit()
        self.comparison_text.setMaximumHeight(200)
        layout.addWidget(self.comparison_text)
    
    def create_advanced_control_panel(self):
        """创建高级控制面板"""
        panel = QWidget()
        panel.setMaximumWidth(400)
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # 标题
        title = TitleLabel('高级配送系统')
        layout.addWidget(title)
        
        # 基本参数组
        basic_group = QGroupBox('基本参数')
        basic_layout = QGridLayout()
        basic_group.setLayout(basic_layout)
        
        basic_layout.addWidget(BodyLabel('客户数量:'), 0, 0)
        self.customer_count = SpinBox()
        self.customer_count.setRange(20, 200)
        self.customer_count.setValue(40)
        basic_layout.addWidget(self.customer_count, 0, 1)
        
        basic_layout.addWidget(BodyLabel('车辆容量(kg):'), 1, 0)
        self.vehicle_capacity = SpinBox()
        self.vehicle_capacity.setRange(30, 100)
        self.vehicle_capacity.setValue(50)
        basic_layout.addWidget(self.vehicle_capacity, 1, 1)
        
        layout.addWidget(basic_group)
        
        # 算法选择组
        algorithm_group = QGroupBox('算法选择')
        algorithm_layout = QGridLayout()
        algorithm_group.setLayout(algorithm_layout)
        
        algorithm_layout.addWidget(BodyLabel('分区算法:'), 0, 0)
        self.partition_combo = ComboBox()
        self.partition_combo.addItems(['分治法', 'K-means聚类', '容量限制'])
        algorithm_layout.addWidget(self.partition_combo, 0, 1)
        algorithm_layout.addWidget(BodyLabel('路径算法:'), 1, 0)
        self.route_combo = ComboBox()
        self.route_combo.addItems(['Greedy', '2-Opt', 'DP', 'GA', 'ACO', 'SA'])
        algorithm_layout.addWidget(self.route_combo, 1, 1)
        
        layout.addWidget(algorithm_group)
        
        # 操作按钮
        self.generate_btn = PushButton('生成优化方案', icon=FluentIcon.PLAY)
        self.generate_btn.clicked.connect(self.generate_advanced_plan)
        layout.addWidget(self.generate_btn)
        
        # 结果显示
        result_group = QGroupBox('详细结果')
        result_layout = QVBoxLayout()
        result_group.setLayout(result_layout)
        
        self.result_text = TextEdit()
        result_layout.addWidget(self.result_text)
        
        layout.addWidget(result_group)
        layout.addStretch()
        
        return panel
    
    def generate_advanced_plan(self):
        """生成高级配送方案"""
        self.generate_btn.setEnabled(False)
        self.generate_btn.setText('计算中...')
        
        self.calc_thread = AdvancedCalculationThread(
            self.customer_count.value(),
            self.partition_combo.currentText(),
            self.route_combo.currentText(),
            self.vehicle_capacity.value()
        )
        self.calc_thread.finished.connect(self.on_advanced_calculation_finished)
        self.calc_thread.start()
    
    def on_advanced_calculation_finished(self, zones, total_distance, stats):
        """高级计算完成回调"""
        self.zones = zones
        
        # 更新可视化
        algorithm_info = f"{stats['partition_method']} + {stats['route_method']}"
        self.visualization.plot_advanced_delivery_system(zones, algorithm_info)
        
        # 更新结果显示
        result_text = f"=== 高级配送方案结果 ===\n\n"
        result_text += f"算法组合: {algorithm_info}\n"
        result_text += f"总客户数: {sum(len(zone.customers) for zone in zones)}\n"
        result_text += f"使用车辆: {len(zones)} 辆\n"
        result_text += f"车辆容量: {self.vehicle_capacity.value()}kg\n"
        result_text += f"总运输距离: {total_distance:.2f}\n"
        result_text += f"平均距离: {stats['avg_distance']:.2f}\n"
        result_text += f"距离范围: {stats['min_distance']:.2f} - {stats['max_distance']:.2f}\n"
        result_text += f"总计算时间: {stats['total_time']:.3f}秒\n\n"
        
        # 详细区域信息
        for zone in zones:
            result_text += f"区域 {zone.zone_id} ({zone.algorithm_used}):\n"
            result_text += f"  客户数: {len(zone.customers)}\n"
            result_text += f"  总重量: {zone.total_weight}kg\n"
            result_text += f"  配送距离: {zone.total_distance:.2f}\n"
            result_text += f"  计算时间: {zone.calculation_time*1000:.1f}ms\n"
            result_text += f"  载重率: {zone.total_weight/self.vehicle_capacity.value()*100:.1f}%\n\n"
        
        self.result_text.setPlainText(result_text)
        
        # 恢复按钮
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText('生成优化方案')
        
        # 显示成功消息
        InfoBar.success(
            title='成功',
            content=f'方案生成完成！总距离: {total_distance:.2f}',
            orient=Qt.Orientation.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=3000,
            parent=self
        )
    def run_algorithm_comparison(self):
        """运行算法对比"""
        self.compare_btn.setEnabled(False)
        self.compare_btn.setText('对比中...')
        self.comparison_results = []
        
        # 定义要对比的算法组合 - 包含新添加的算法
        algorithms = [
            ('分治法', 'Greedy'),
            ('分治法', '2-Opt'),
            ('分治法', 'DP'),
            ('分治法', 'GA'),
            ('K-means聚类', 'Greedy'),
            ('K-means聚类', '2-Opt'),
            ('K-means聚类', 'ACO'),
            ('容量限制', 'Greedy'),
            ('容量限制', 'SA'),
            ('容量限制', 'GA')
        ]
        
        self.current_comparison = 0
        self.total_comparisons = len(algorithms)
        self.comparison_algorithms = algorithms
        
        # 开始第一个对比
        self.run_single_comparison()
    
    def run_single_comparison(self):
        """运行单个算法对比"""
        if self.current_comparison >= self.total_comparisons:
            self.finish_comparison()
            return
        
        partition_method, route_method = self.comparison_algorithms[self.current_comparison]
        
        self.comparison_thread = AdvancedCalculationThread(
            self.customer_count.value(),
            partition_method,
            route_method,
            self.vehicle_capacity.value()
        )
        self.comparison_thread.finished.connect(self.on_comparison_finished)
        self.comparison_thread.start()
    
    def on_comparison_finished(self, zones, total_distance, stats):
        """单个对比完成"""
        self.comparison_results.append({
            'zones': zones,
            'total_distance': total_distance,
            'stats': stats
        })
        
        # 更新表格
        row = self.current_comparison
        self.comparison_table.setRowCount(row + 1)
        
        self.comparison_table.setItem(row, 0, QTableWidgetItem(stats['partition_method']))
        self.comparison_table.setItem(row, 1, QTableWidgetItem(stats['route_method']))
        self.comparison_table.setItem(row, 2, QTableWidgetItem(f"{total_distance:.2f}"))
        self.comparison_table.setItem(row, 3, QTableWidgetItem(str(stats['num_zones'])))
        self.comparison_table.setItem(row, 4, QTableWidgetItem(f"{stats['avg_distance']:.2f}"))
        self.comparison_table.setItem(row, 5, QTableWidgetItem(f"{stats['max_distance']:.2f}"))
        self.comparison_table.setItem(row, 6, QTableWidgetItem(f"{stats['min_distance']:.2f}"))
        self.comparison_table.setItem(row, 7, QTableWidgetItem(f"{stats['total_time']:.3f}"))
        
        self.current_comparison += 1
        self.run_single_comparison()
    
    def finish_comparison(self):
        """完成所有对比"""
        # 分析结果
        best_result = min(self.comparison_results, key=lambda x: x['total_distance'])
        fastest_result = min(self.comparison_results, key=lambda x: x['stats']['total_time'])
        
        analysis_text = "=== 算法对比分析报告 ===\n\n"
        analysis_text += f"测试配置: {self.customer_count.value()}个客户, {self.vehicle_capacity.value()}kg容量\n\n"
        
        analysis_text += "最优距离方案:\n"
        analysis_text += f"  算法: {best_result['stats']['partition_method']} + {best_result['stats']['route_method']}\n"
        analysis_text += f"  总距离: {best_result['total_distance']:.2f}\n"
        analysis_text += f"  车辆数: {best_result['stats']['num_zones']}\n\n"
        
        analysis_text += "最快计算方案:\n"
        analysis_text += f"  算法: {fastest_result['stats']['partition_method']} + {fastest_result['stats']['route_method']}\n"
        analysis_text += f"  计算时间: {fastest_result['stats']['total_time']:.3f}秒\n"
        analysis_text += f"  总距离: {fastest_result['total_distance']:.2f}\n\n"
          # 算法特性分析
        analysis_text += "算法特性分析:\n"
        analysis_text += "分区算法:\n"
        analysis_text += "? 分治法: 保证负载均衡，适合大规模问题\n"
        analysis_text += "? K-means: 基于距离聚类，可能产生更优路线\n"
        analysis_text += "? 容量限制: 考虑实际约束，实用性强\n\n"
        analysis_text += "路径算法:\n"
        analysis_text += "? Greedy: 快速求解，适合实时应用，O(n?)复杂度\n"
        analysis_text += "? 2-Opt: 局部优化，质量较好，适中计算开销\n"
        analysis_text += "? DP: 保证全局最优，但仅适用于小规模问题(<10客户)\n"
        analysis_text += "? GA: 遗传算法，适合中大规模问题，解质量好\n"
        analysis_text += "? ACO: 蚁群算法，模拟自然行为，收敛稳定\n"
        analysis_text += "? SA: 模拟退火，能跳出局部最优，参数敏感\n"
        
        self.comparison_text.setPlainText(analysis_text)
        
        # 恢复按钮
        self.compare_btn.setEnabled(True)
        self.compare_btn.setText('运行算法对比')
        
        InfoBar.success(
            title='对比完成',
            content='算法对比分析已完成',
            orient=Qt.Orientation.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=2000,
            parent=self
        )

def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    app.setApplicationName('高级智能快递配送系统')
    app.setApplicationVersion('2.0')
    
    window = AdvancedDeliverySystemWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
