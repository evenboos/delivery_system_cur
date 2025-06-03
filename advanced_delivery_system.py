# -*- coding: gb2312 -*-
"""
�߼��������ϵͳ - ���������㷨�ԱȺ���չ����
�������ܣ�
1. ���ַ����㷨�����η���K-means���ࡢ�����������ƣ�
2. ����·���Ż��㷨��̰�ġ�����ڡ�2-opt�Ż���
3. ������������
4. �㷨���ܶԱ�
5. ��������
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

# ����matplotlib֧������
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Customer:
    """��ǿ�ͻ���"""
    def __init__(self, x, y, customer_id, cargo_weight=None):
        self.x = x
        self.y = y
        self.id = customer_id
        self.cargo_weight = cargo_weight or random.randint(1, 15)  # ��������1-15kg
        self.priority = random.choice(['normal', 'urgent', 'express'])  # ���ȼ�
        self.time_window = (8, 18)  # ����ʱ�䴰��
    
    def distance_to(self, other):
        """���㵽��һ����ľ���"""
        if isinstance(other, Customer):
            return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
        else:
            return math.sqrt((self.x - other[0])**2 + (self.y - other[1])**2)

class Vehicle:
    """������"""
    def __init__(self, vehicle_id, capacity=50):
        self.id = vehicle_id
        self.capacity = capacity  # �������ƣ�kg��
        self.current_load = 0
        self.route = [(0, 0)]  # ��ԭ�㿪ʼ
        self.customers = []
        self.total_distance = 0
    
    def can_add_customer(self, customer):
        """����Ƿ������ӿͻ�"""
        return self.current_load + customer.cargo_weight <= self.capacity
    
    def add_customer(self, customer):
        """��ӿͻ�������"""
        if self.can_add_customer(customer):
            self.customers.append(customer)
            self.current_load += customer.cargo_weight
            return True
        return False

class PartitionAlgorithm:
    """�����㷨��"""
    
    @staticmethod
    def divide_and_conquer(customers, group_size=5):
        """���η�����"""
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
        """K-means�������"""
        if len(customers) <= num_clusters:
            return [[c] for c in customers]
        
        # ��ʼ����������
        centers = random.sample(customers, num_clusters)
        center_coords = [(c.x, c.y) for c in centers]
        
        for _ in range(10):  # ������10��
            clusters = [[] for _ in range(num_clusters)]
            
            # ����ͻ�������ľ�������
            for customer in customers:
                distances = [customer.distance_to(center) for center in center_coords]
                closest_center = distances.index(min(distances))
                clusters[closest_center].append(customer)
            
            # ���¾�������
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
        """�����������Ƶķ���"""
        groups = []
        current_group = []
        current_weight = 0
        
        # �����ȼ�����������
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
    """·���Ż��㷨��"""
    
    @staticmethod
    def greedy_nearest(customers):
        """̰��������㷨"""
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
        
        # ����ԭ��
        return_distance = math.sqrt(current_pos[0]**2 + current_pos[1]**2)
        total_distance += return_distance
        route.append((0, 0))
        
        return route, total_distance
    
    @staticmethod
    def two_opt_improve(route, customers):
        """2-opt�Ż��㷨"""
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
        """��̬�滮TSP�㷨 - ��֤ȫ�����Ž⣨������С��ģ���⣩"""
        if not customers:
            return [], 0
        
        if len(customers) > 10:  # DP�㷨���Ӷȹ��ߣ����������ģ
            # ���ڴ��ģ���⣬���˵�̰���㷨
            return RouteOptimizer.greedy_nearest(customers)
        
        n = len(customers)
        # ���ԭ����Ϊ��ʼ��
        points = [(0, 0)] + [(c.x, c.y) for c in customers]
        
        # ����������
        dist = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            for j in range(n + 1):
                if i != j:
                    dist[i][j] = math.sqrt((points[i][0] - points[j][0])**2 + 
                                         (points[i][1] - points[j][1])**2)
        
        # DP״̬��dp[mask][i] ��ʾ������mask�����еĳ��У���ǰ�ڳ���i����̾���
        dp = {}
        parent = {}
        
        # ��ʼ������ԭ��0��ʼ
        dp[(1, 0)] = 0  # mask=1��ʾֻ������ԭ��
        
        # ���DP��
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
        
        # �ҵ����Ž⣺�ص�ԭ��
        full_mask = (1 << (n + 1)) - 1
        min_cost = float('inf')
        last_city = -1
        
        for i in range(1, n + 1):
            if (full_mask, i) in dp:
                cost = dp[(full_mask, i)] + dist[i][0]
                if cost < min_cost:
                    min_cost = cost
                    last_city = i
        
        # �ع�·��
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
        
        path.append(0)  # ��ʼ��
        path.reverse()
        
        # ת��Ϊ����·��
        route = [points[i] for i in path] + [(0, 0)]
        
        return route, min_cost
    
    @staticmethod
    def genetic_algorithm_tsp(customers, population_size=50, generations=100, mutation_rate=0.1):
        """�Ŵ��㷨TSP"""
        if not customers:
            return [], 0
        
        n = len(customers)
        points = [(0, 0)] + [(c.x, c.y) for c in customers]
        
        def calculate_distance(route):
            """����·���ܾ���"""
            total = 0
            for i in range(len(route) - 1):
                p1, p2 = points[route[i]], points[route[i + 1]]
                total += math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            return total
        
        def create_individual():
            """�������壨���·����"""
            route = [0] + list(range(1, n + 1)) + [0]  # ��ԭ�㿪ʼ�ͽ���
            middle = route[1:-1]
            random.shuffle(middle)
            return route[0:1] + middle + route[-1:]
        
        def crossover(parent1, parent2):
            """������� - ˳�򽻲�"""
            size = len(parent1) - 2  # ��ȥ��ʼ�ͽ�����ԭ��
            start, end = sorted(random.sample(range(1, size + 1), 2))
            
            child = [0] * (size + 2)
            child[0] = child[-1] = 0  # ��ʼ�ͽ�������ԭ��
            
            # ���Ƹ���1��һ��
            child[start:end] = parent1[start:end]
            
            # ���ʣ��λ��
            remaining = [item for item in parent2[1:-1] if item not in child[start:end]]
            j = 0
            for i in range(1, size + 1):
                if child[i] == 0:
                    child[i] = remaining[j]
                    j += 1
            
            return child
        
        def mutate(individual):
            """������� - ������������"""
            if random.random() < mutation_rate:
                i, j = random.sample(range(1, len(individual) - 1), 2)
                individual[i], individual[j] = individual[j], individual[i]
            return individual
        
        # ��ʼ����Ⱥ
        population = [create_individual() for _ in range(population_size)]
        
        # ��������
        for generation in range(generations):
            # ������Ӧ�ȣ�����Խ����Ӧ��Խ�ߣ�
            fitness_scores = [(1.0 / (calculate_distance(ind) + 1), ind) for ind in population]
            fitness_scores.sort(reverse=True)
            
            # ѡ���������
            elite_size = population_size // 4
            new_population = [ind for _, ind in fitness_scores[:elite_size]]
            
            # �����¸���
            while len(new_population) < population_size:
                # ���̶�ѡ��
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
        
        # �������Ž�
        best_individual = min(population, key=calculate_distance)
        best_distance = calculate_distance(best_individual)
        best_route = [points[i] for i in best_individual]
        
        return best_route, best_distance
    
    @staticmethod
    def ant_colony_optimization(customers, num_ants=20, iterations=50, alpha=1.0, beta=2.0, evaporation=0.5):
        """��Ⱥ�㷨TSP"""
        if not customers:
            return [], 0
        
        n = len(customers)
        points = [(0, 0)] + [(c.x, c.y) for c in customers]
        
        # ����������
        distances = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            for j in range(n + 1):
                if i != j:
                    distances[i][j] = math.sqrt((points[i][0] - points[j][0])**2 + 
                                              (points[i][1] - points[j][1])**2)
        
        # ��ʼ����Ϣ�ؾ���
        pheromones = [[1.0] * (n + 1) for _ in range(n + 1)]
        
        best_route = None
        best_distance = float('inf')
        
        for iteration in range(iterations):
            all_routes = []
            
            # ÿֻ���Ϲ���·��
            for ant in range(num_ants):
                current = 0  # ��ԭ�㿪ʼ
                visited = {0}
                route = [0]
                
                # �������пͻ�
                while len(visited) < n + 1:
                    unvisited = [i for i in range(n + 1) if i not in visited]
                    
                    # ����ת�Ƹ���
                    probabilities = []
                    for next_city in unvisited:
                        pheromone = pheromones[current][next_city] ** alpha
                        visibility = (1.0 / distances[current][next_city]) ** beta if distances[current][next_city] > 0 else 0
                        probabilities.append(pheromone * visibility)
                    
                    # ���̶�ѡ����һ������
                    if sum(probabilities) > 0:
                        probabilities = [p / sum(probabilities) for p in probabilities]
                        next_city = np.random.choice(unvisited, p=probabilities)
                    else:
                        next_city = random.choice(unvisited)
                    
                    route.append(next_city)
                    visited.add(next_city)
                    current = next_city
                
                # �ص�ԭ��
                route.append(0)
                
                # ����·������
                distance = sum(distances[route[i]][route[i + 1]] for i in range(len(route) - 1))
                all_routes.append((route, distance))
                
                # �������Ž�
                if distance < best_distance:
                    best_distance = distance
                    best_route = route.copy()
            
            # ������Ϣ��
            # ����
            for i in range(n + 1):
                for j in range(n + 1):
                    pheromones[i][j] *= (1 - evaporation)
            
            # ��ǿ
            for route, distance in all_routes:
                deposit = 1.0 / distance if distance > 0 else 0
                for i in range(len(route) - 1):
                    pheromones[route[i]][route[i + 1]] += deposit
                    pheromones[route[i + 1]][route[i]] += deposit
        
        # ת��Ϊ����·��
        result_route = [points[i] for i in best_route]
        return result_route, best_distance
    
    @staticmethod
    def simulated_annealing_tsp(customers, initial_temp=1000, cooling_rate=0.95, min_temp=1):
        """ģ���˻��㷨TSP"""
        if not customers:
            return [], 0
        
        n = len(customers)
        points = [(0, 0)] + [(c.x, c.y) for c in customers]
        
        def calculate_distance(route):
            """����·���ܾ���"""
            total = 0
            for i in range(len(route) - 1):
                p1, p2 = points[route[i]], points[route[i + 1]]
                total += math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            return total
        
        def get_neighbor(route):
            """�����ھӽ� - ���������������"""
            new_route = route.copy()
            i, j = random.sample(range(1, len(route) - 1), 2)  # ��������ʼ�ͽ�����ԭ��
            new_route[i], new_route[j] = new_route[j], new_route[i]
            return new_route
        
        # ��ʼ�� - ̰���㷨����
        current_route = [0] + list(range(1, n + 1)) + [0]
        current_distance = calculate_distance(current_route)
        
        best_route = current_route.copy()
        best_distance = current_distance
        
        temperature = initial_temp
        
        while temperature > min_temp:
            for _ in range(100):  # ��ÿ���¶��³���100��
                new_route = get_neighbor(current_route)
                new_distance = calculate_distance(new_route)
                
                # ������ܸ���
                delta = new_distance - current_distance
                if delta < 0 or random.random() < math.exp(-delta / temperature):
                    current_route = new_route
                    current_distance = new_distance
                    
                    # �������Ž�
                    if current_distance < best_distance:
                        best_route = current_route.copy()
                        best_distance = current_distance
            
            temperature *= cooling_rate
        
        # ת��Ϊ����·��
        result_route = [points[i] for i in best_route]
        return result_route, best_distance

class DeliveryZone:
    """��ǿ����������"""
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
        """ʹ��ָ���㷨�Ż�·��"""
        start_time = time.time()
        
        if algorithm == 'greedy':
            self.route, self.total_distance = RouteOptimizer.greedy_nearest(self.customers)
            self.algorithm_used = "̰�������"
        elif algorithm == '2opt':
            greedy_route, _ = RouteOptimizer.greedy_nearest(self.customers)
            self.route, self.total_distance = RouteOptimizer.two_opt_improve(greedy_route, self.customers)
            self.algorithm_used = "2-opt�Ż�"
        elif algorithm == 'dp':
            self.route, self.total_distance = RouteOptimizer.dynamic_programming_tsp(self.customers)
            self.algorithm_used = "��̬�滮(DP)"
        elif algorithm == 'ga':
            self.route, self.total_distance = RouteOptimizer.genetic_algorithm_tsp(self.customers)
            self.algorithm_used = "�Ŵ��㷨(GA)"
        elif algorithm == 'aco':
            self.route, self.total_distance = RouteOptimizer.ant_colony_optimization(self.customers)
            self.algorithm_used = "��Ⱥ�㷨(ACO)"
        elif algorithm == 'sa':
            self.route, self.total_distance = RouteOptimizer.simulated_annealing_tsp(self.customers)
            self.algorithm_used = "ģ���˻�(SA)"
        
        self.calculation_time = time.time() - start_time
        return self.total_distance

class AdvancedVisualizationWidget(QWidget):
    """�߼����ӻ����"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.zones = []
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # ����matplotlibͼ��
        self.figure = Figure(figsize=(14, 10))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
    
    def plot_advanced_delivery_system(self, zones, algorithm_info=""):
        """���Ƹ߼�����ϵͳͼ"""
        self.zones = zones
        self.figure.clear()
        
        # ������ͼ
        ax1 = self.figure.add_subplot(221)  # ·��ͼ
        ax2 = self.figure.add_subplot(222)  # �����ֲ�
        ax3 = self.figure.add_subplot(223)  # �������
        ax4 = self.figure.add_subplot(224)  # �㷨����
        
        # ����·��ͼ
        self.plot_routes(ax1, zones)
        
        # ���������ֲ�
        self.plot_capacity_distribution(ax2, zones)
        
        # ���ƾ������
        self.plot_distance_analysis(ax3, zones)
        
        # �����㷨����
        self.plot_algorithm_performance(ax4, zones)
        
        self.figure.suptitle(f'�߼��������ϵͳ���� - {algorithm_info}', fontsize=14)
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_routes(self, ax, zones):
        """��������·��"""
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 
                 'pink', 'gray', 'olive', 'cyan']
        
        ax.plot(0, 0, 'ko', markersize=15, label='��������')
        
        for i, zone in enumerate(zones):
            color = colors[i % len(colors)]
            
            # ���ƿͻ���
            x_coords = [c.x for c in zone.customers]
            y_coords = [c.y for c in zone.customers]
            
            # ���ݻ�������������Ĵ�С
            sizes = [c.cargo_weight * 10 for c in zone.customers]
            ax.scatter(x_coords, y_coords, c=color, s=sizes, alpha=0.7,
                      label=f'����{zone.zone_id}({zone.total_weight}kg)')
            
            # ��������·��
            if zone.route:
                route_x = [point[0] for point in zone.route]
                route_y = [point[1] for point in zone.route]
                ax.plot(route_x, route_y, color=color, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('X����')
        ax.set_ylabel('Y����')
        ax.set_title('����·��ͼ')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def plot_capacity_distribution(self, ax, zones):
        """���������ֲ�ͼ"""
        zone_ids = [f'����{zone.zone_id}' for zone in zones]
        weights = [zone.total_weight for zone in zones]
        
        bars = ax.bar(zone_ids, weights, color='skyblue', alpha=0.7)
        ax.axhline(y=50, color='red', linestyle='--', label='��������(50kg)')
        
        # ����������ʾ��ֵ
        for bar, weight in zip(bars, weights):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{weight}kg', ha='center', va='bottom')
        
        ax.set_ylabel('���� (kg)')
        ax.set_title('��������������ֲ�')
        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    def plot_distance_analysis(self, ax, zones):
        """���ƾ������ͼ"""
        distances = [zone.total_distance for zone in zones]
        zone_ids = [f'����{zone.zone_id}' for zone in zones]
        
        ax.bar(zone_ids, distances, color='lightgreen', alpha=0.7)
        ax.set_ylabel('����')
        ax.set_title('���������;���')
        
        # ���ƽ����
        avg_distance = np.mean(distances)
        ax.axhline(y=avg_distance, color='red', linestyle='--', 
                  label=f'ƽ������: {avg_distance:.1f}')
        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    def plot_algorithm_performance(self, ax, zones):
        """�����㷨����ͼ"""
        calc_times = [zone.calculation_time * 1000 for zone in zones]  # ת��Ϊ����
        zone_ids = [f'����{zone.zone_id}' for zone in zones]
        
        ax.bar(zone_ids, calc_times, color='orange', alpha=0.7)
        ax.set_ylabel('����ʱ�� (ms)')
        ax.set_title('�㷨����ʱ��')
        plt.setp(ax.get_xticklabels(), rotation=45)

class AdvancedCalculationThread(QThread):
    """�߼������߳�"""
    finished = pyqtSignal(list, float, dict)
    
    def __init__(self, num_customers, partition_method, route_method, vehicle_capacity):
        super().__init__()
        self.num_customers = num_customers
        self.partition_method = partition_method
        self.route_method = route_method
        self.vehicle_capacity = vehicle_capacity
    
    def run(self):
        start_time = time.time()
        
        # ��������ͻ�
        customers = []
        for i in range(self.num_customers):
            x = random.randint(-60, 60)
            y = random.randint(-60, 60)
            customers.append(Customer(x, y, i+1))
        
        # ѡ������㷨
        if self.partition_method == "���η�":
            customer_groups = PartitionAlgorithm.divide_and_conquer(customers, 5)
        elif self.partition_method == "K-means����":
            num_clusters = max(1, len(customers) // 5)
            customer_groups = PartitionAlgorithm.kmeans_partition(customers, num_clusters)
        elif self.partition_method == "��������":
            customer_groups = PartitionAlgorithm.capacity_based_partition(
                customers, self.vehicle_capacity)
        
        # �������������Ż�·��
        zones = []
        total_distance = 0
        
        for i, group in enumerate(customer_groups):
            if group:  # ȷ���鲻Ϊ��
                zone = DeliveryZone(group, i+1, self.vehicle_capacity)
                distance = zone.optimize_route(self.route_method.lower().replace('-', ''))
                total_distance += distance
                zones.append(zone)
        
        # ����ͳ����Ϣ
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
    """�߼���������"""
    
    def __init__(self):
        super().__init__()
        self.zones = []
        self.comparison_results = []
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('�߼����ܿ������ϵͳ v2.0')
        self.setGeometry(50, 50, 1600, 900)
        
        # ��������
        setTheme(Theme.AUTO)
        
        # ������ǩҳ
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        
        # ������Ҫ������ǩҳ
        self.create_main_tab()
        
        # �����㷨�Աȱ�ǩҳ
        self.create_comparison_tab()
    
    def create_main_tab(self):
        """������Ҫ������ǩҳ"""
        main_tab = QWidget()
        self.tab_widget.addTab(main_tab, "��Ҫ����")
        
        layout = QHBoxLayout()
        main_tab.setLayout(layout)
        
        # ���������
        control_panel = self.create_advanced_control_panel()
        layout.addWidget(control_panel)
        
        # �Ҳ���ӻ�����
        self.visualization = AdvancedVisualizationWidget()
        layout.addWidget(self.visualization)
        
        layout.setStretch(0, 1)
        layout.setStretch(1, 3)
    
    def create_comparison_tab(self):
        """�����㷨�Աȱ�ǩҳ"""
        comparison_tab = QWidget()
        self.tab_widget.addTab(comparison_tab, "�㷨�Ա�")
        
        layout = QVBoxLayout()
        comparison_tab.setLayout(layout)
        
        # �Աȿ��ư�ť
        btn_layout = QHBoxLayout()
        self.compare_btn = PushButton('�����㷨�Ա�', icon=FluentIcon.SYNC)
        self.compare_btn.clicked.connect(self.run_algorithm_comparison)
        btn_layout.addWidget(self.compare_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # �ԱȽ�����
        self.comparison_table = TableWidget()
        self.comparison_table.setColumnCount(8)
        self.comparison_table.setHorizontalHeaderLabels([
            '�����㷨', '·���㷨', '�ܾ���', '������', 'ƽ������', 
            '������', '��С����', '����ʱ��(s)'
        ])
        layout.addWidget(self.comparison_table)
        
        # �ԱȽ���ı�
        self.comparison_text = TextEdit()
        self.comparison_text.setMaximumHeight(200)
        layout.addWidget(self.comparison_text)
    
    def create_advanced_control_panel(self):
        """�����߼��������"""
        panel = QWidget()
        panel.setMaximumWidth(400)
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # ����
        title = TitleLabel('�߼�����ϵͳ')
        layout.addWidget(title)
        
        # ����������
        basic_group = QGroupBox('��������')
        basic_layout = QGridLayout()
        basic_group.setLayout(basic_layout)
        
        basic_layout.addWidget(BodyLabel('�ͻ�����:'), 0, 0)
        self.customer_count = SpinBox()
        self.customer_count.setRange(20, 200)
        self.customer_count.setValue(40)
        basic_layout.addWidget(self.customer_count, 0, 1)
        
        basic_layout.addWidget(BodyLabel('��������(kg):'), 1, 0)
        self.vehicle_capacity = SpinBox()
        self.vehicle_capacity.setRange(30, 100)
        self.vehicle_capacity.setValue(50)
        basic_layout.addWidget(self.vehicle_capacity, 1, 1)
        
        layout.addWidget(basic_group)
        
        # �㷨ѡ����
        algorithm_group = QGroupBox('�㷨ѡ��')
        algorithm_layout = QGridLayout()
        algorithm_group.setLayout(algorithm_layout)
        
        algorithm_layout.addWidget(BodyLabel('�����㷨:'), 0, 0)
        self.partition_combo = ComboBox()
        self.partition_combo.addItems(['���η�', 'K-means����', '��������'])
        algorithm_layout.addWidget(self.partition_combo, 0, 1)
        algorithm_layout.addWidget(BodyLabel('·���㷨:'), 1, 0)
        self.route_combo = ComboBox()
        self.route_combo.addItems(['Greedy', '2-Opt', 'DP', 'GA', 'ACO', 'SA'])
        algorithm_layout.addWidget(self.route_combo, 1, 1)
        
        layout.addWidget(algorithm_group)
        
        # ������ť
        self.generate_btn = PushButton('�����Ż�����', icon=FluentIcon.PLAY)
        self.generate_btn.clicked.connect(self.generate_advanced_plan)
        layout.addWidget(self.generate_btn)
        
        # �����ʾ
        result_group = QGroupBox('��ϸ���')
        result_layout = QVBoxLayout()
        result_group.setLayout(result_layout)
        
        self.result_text = TextEdit()
        result_layout.addWidget(self.result_text)
        
        layout.addWidget(result_group)
        layout.addStretch()
        
        return panel
    
    def generate_advanced_plan(self):
        """���ɸ߼����ͷ���"""
        self.generate_btn.setEnabled(False)
        self.generate_btn.setText('������...')
        
        self.calc_thread = AdvancedCalculationThread(
            self.customer_count.value(),
            self.partition_combo.currentText(),
            self.route_combo.currentText(),
            self.vehicle_capacity.value()
        )
        self.calc_thread.finished.connect(self.on_advanced_calculation_finished)
        self.calc_thread.start()
    
    def on_advanced_calculation_finished(self, zones, total_distance, stats):
        """�߼�������ɻص�"""
        self.zones = zones
        
        # ���¿��ӻ�
        algorithm_info = f"{stats['partition_method']} + {stats['route_method']}"
        self.visualization.plot_advanced_delivery_system(zones, algorithm_info)
        
        # ���½����ʾ
        result_text = f"=== �߼����ͷ������ ===\n\n"
        result_text += f"�㷨���: {algorithm_info}\n"
        result_text += f"�ܿͻ���: {sum(len(zone.customers) for zone in zones)}\n"
        result_text += f"ʹ�ó���: {len(zones)} ��\n"
        result_text += f"��������: {self.vehicle_capacity.value()}kg\n"
        result_text += f"���������: {total_distance:.2f}\n"
        result_text += f"ƽ������: {stats['avg_distance']:.2f}\n"
        result_text += f"���뷶Χ: {stats['min_distance']:.2f} - {stats['max_distance']:.2f}\n"
        result_text += f"�ܼ���ʱ��: {stats['total_time']:.3f}��\n\n"
        
        # ��ϸ������Ϣ
        for zone in zones:
            result_text += f"���� {zone.zone_id} ({zone.algorithm_used}):\n"
            result_text += f"  �ͻ���: {len(zone.customers)}\n"
            result_text += f"  ������: {zone.total_weight}kg\n"
            result_text += f"  ���;���: {zone.total_distance:.2f}\n"
            result_text += f"  ����ʱ��: {zone.calculation_time*1000:.1f}ms\n"
            result_text += f"  ������: {zone.total_weight/self.vehicle_capacity.value()*100:.1f}%\n\n"
        
        self.result_text.setPlainText(result_text)
        
        # �ָ���ť
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText('�����Ż�����')
        
        # ��ʾ�ɹ���Ϣ
        InfoBar.success(
            title='�ɹ�',
            content=f'����������ɣ��ܾ���: {total_distance:.2f}',
            orient=Qt.Orientation.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=3000,
            parent=self
        )
    def run_algorithm_comparison(self):
        """�����㷨�Ա�"""
        self.compare_btn.setEnabled(False)
        self.compare_btn.setText('�Ա���...')
        self.comparison_results = []
        
        # ����Ҫ�Աȵ��㷨��� - ��������ӵ��㷨
        algorithms = [
            ('���η�', 'Greedy'),
            ('���η�', '2-Opt'),
            ('���η�', 'DP'),
            ('���η�', 'GA'),
            ('K-means����', 'Greedy'),
            ('K-means����', '2-Opt'),
            ('K-means����', 'ACO'),
            ('��������', 'Greedy'),
            ('��������', 'SA'),
            ('��������', 'GA')
        ]
        
        self.current_comparison = 0
        self.total_comparisons = len(algorithms)
        self.comparison_algorithms = algorithms
        
        # ��ʼ��һ���Ա�
        self.run_single_comparison()
    
    def run_single_comparison(self):
        """���е����㷨�Ա�"""
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
        """�����Ա����"""
        self.comparison_results.append({
            'zones': zones,
            'total_distance': total_distance,
            'stats': stats
        })
        
        # ���±��
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
        """������жԱ�"""
        # �������
        best_result = min(self.comparison_results, key=lambda x: x['total_distance'])
        fastest_result = min(self.comparison_results, key=lambda x: x['stats']['total_time'])
        
        analysis_text = "=== �㷨�Աȷ������� ===\n\n"
        analysis_text += f"��������: {self.customer_count.value()}���ͻ�, {self.vehicle_capacity.value()}kg����\n\n"
        
        analysis_text += "���ž��뷽��:\n"
        analysis_text += f"  �㷨: {best_result['stats']['partition_method']} + {best_result['stats']['route_method']}\n"
        analysis_text += f"  �ܾ���: {best_result['total_distance']:.2f}\n"
        analysis_text += f"  ������: {best_result['stats']['num_zones']}\n\n"
        
        analysis_text += "�����㷽��:\n"
        analysis_text += f"  �㷨: {fastest_result['stats']['partition_method']} + {fastest_result['stats']['route_method']}\n"
        analysis_text += f"  ����ʱ��: {fastest_result['stats']['total_time']:.3f}��\n"
        analysis_text += f"  �ܾ���: {fastest_result['total_distance']:.2f}\n\n"
          # �㷨���Է���
        analysis_text += "�㷨���Է���:\n"
        analysis_text += "�����㷨:\n"
        analysis_text += "? ���η�: ��֤���ؾ��⣬�ʺϴ��ģ����\n"
        analysis_text += "? K-means: ���ھ�����࣬���ܲ�������·��\n"
        analysis_text += "? ��������: ����ʵ��Լ����ʵ����ǿ\n\n"
        analysis_text += "·���㷨:\n"
        analysis_text += "? Greedy: ������⣬�ʺ�ʵʱӦ�ã�O(n?)���Ӷ�\n"
        analysis_text += "? 2-Opt: �ֲ��Ż��������Ϻã����м��㿪��\n"
        analysis_text += "? DP: ��֤ȫ�����ţ�����������С��ģ����(<10�ͻ�)\n"
        analysis_text += "? GA: �Ŵ��㷨���ʺ��д��ģ���⣬��������\n"
        analysis_text += "? ACO: ��Ⱥ�㷨��ģ����Ȼ��Ϊ�������ȶ�\n"
        analysis_text += "? SA: ģ���˻��������ֲ����ţ���������\n"
        
        self.comparison_text.setPlainText(analysis_text)
        
        # �ָ���ť
        self.compare_btn.setEnabled(True)
        self.compare_btn.setText('�����㷨�Ա�')
        
        InfoBar.success(
            title='�Ա����',
            content='�㷨�Աȷ��������',
            orient=Qt.Orientation.Horizontal,
            isClosable=True,
            position=InfoBarPosition.TOP,
            duration=2000,
            parent=self
        )

def main():
    """������"""
    app = QApplication(sys.argv)
    
    app.setApplicationName('�߼����ܿ������ϵͳ')
    app.setApplicationVersion('2.0')
    
    window = AdvancedDeliverySystemWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
