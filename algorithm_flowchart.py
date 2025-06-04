"""
高级配送系统算法流程图
绘制完整的系统算法流程图，包括分区算法、路径优化算法的详细流程
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# 设置matplotlib支持中文
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

class AlgorithmFlowchartVisualization:
    """算法流程图可视化类"""
    
    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 3, figsize=(20, 14))
        self.fig.suptitle("高级智能快递配送系统 - 算法流程图", fontsize=16, fontweight='bold')
        
    def draw_box(self, ax, x, y, width, height, text, color='lightblue', text_color='black'):
        """绘制流程图框"""
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2, text, 
                ha='center', va='center', fontsize=9, 
                color=text_color, weight='bold')
    
    def draw_diamond(self, ax, x, y, width, height, text, color='yellow'):
        """绘制菱形决策框"""
        diamond = patches.RegularPolygon(
            (x + width/2, y + height/2), 4,
            radius=min(width, height)/2,
            orientation=np.pi/4,
            facecolor=color,
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(diamond)
        ax.text(x + width/2, y + height/2, text,
                ha='center', va='center', fontsize=8, weight='bold')
    
    def draw_arrow(self, ax, x1, y1, x2, y2, text=''):
        """绘制箭头"""
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
        if text:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y, text, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                   fontsize=7)
    
    def draw_main_flowchart(self):
        """绘制主流程图"""
        ax = self.axes[0, 0]
        ax.set_title("主要系统流程", fontsize=12, fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 20)
        ax.axis('off')
        
        # 主流程步骤
        steps = [
            (2, 18, 6, 1.5, "开始：输入客户数据", 'lightgreen'),
            (2, 16, 6, 1.5, "生成随机客户坐标", 'lightblue'),
            (2, 14, 6, 1.5, "选择分区算法", 'orange'),
            (2, 12, 6, 1.5, "客户分组", 'lightyellow'),
            (2, 10, 6, 1.5, "选择路径优化算法", 'orange'),
            (2, 8, 6, 1.5, "为每组生成最优路径", 'lightyellow'),
            (2, 6, 6, 1.5, "计算总距离和统计信息", 'lightcyan'),
            (2, 4, 6, 1.5, "可视化结果", 'lightpink'),
            (2, 2, 6, 1.5, "结束：输出优化方案", 'lightgreen')
        ]
        
        for i, (x, y, w, h, text, color) in enumerate(steps):
            self.draw_box(ax, x, y, w, h, text, color)
            if i < len(steps) - 1:
                self.draw_arrow(ax, x + w/2, y, x + w/2, y - 0.5)
    
    def draw_partition_algorithms(self):
        """绘制分区算法流程"""
        ax = self.axes[0, 1]
        ax.set_title("分区算法流程", fontsize=12, fontweight='bold')
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 20)
        ax.axis('off')
        
        # 分区算法选择
        self.draw_box(ax, 6, 18, 3, 1.5, "分区算法", 'orange')
        
        # 三种算法分支
        algorithms = [
            (1, 15, 3.5, 1.2, "分治法\n按X坐标排序\n递归分组", 'lightblue'),
            (5.5, 15, 3.5, 1.2, "K-means聚类\n距离聚类\n迭代优化", 'lightgreen'),
            (10, 15, 3.5, 1.2, "容量限制\n按重量分组\n满足约束", 'lightyellow')
        ]
        
        for x, y, w, h, text, color in algorithms:
            self.draw_box(ax, x, y, w, h, text, color)
            # 从主框指向各算法
            self.draw_arrow(ax, 7.5, 18, x + w/2, y + h)
        
        # 分治法详细流程
        self.draw_box(ax, 0.5, 12, 4, 1, "检查组大小 ≤ 5?", 'yellow')
        self.draw_box(ax, 0.5, 10, 4, 1, "按X坐标排序", 'lightcyan')
        self.draw_box(ax, 0.5, 8, 4, 1, "分成左右两部分", 'lightcyan')
        self.draw_box(ax, 0.5, 6, 4, 1, "递归处理子组", 'lightcyan')
        
        # K-means详细流程
        self.draw_box(ax, 5.5, 12, 4, 1, "初始化聚类中心", 'lightcyan')
        self.draw_box(ax, 5.5, 10, 4, 1, "分配客户到最近中心", 'lightcyan')
        self.draw_box(ax, 5.5, 8, 4, 1, "更新聚类中心", 'lightcyan')
        self.draw_box(ax, 5.5, 6, 4, 1, "收敛检查", 'yellow')
        
        # 容量限制详细流程
        self.draw_box(ax, 10.5, 12, 4, 1, "按优先级排序", 'lightcyan')
        self.draw_box(ax, 10.5, 10, 4, 1, "检查容量限制", 'yellow')
        self.draw_box(ax, 10.5, 8, 4, 1, "添加到当前组", 'lightcyan')
        self.draw_box(ax, 10.5, 6, 4, 1, "创建新组", 'lightcyan')
        
        # 最终结果
        self.draw_box(ax, 6, 3, 3, 1.5, "客户分组\n完成", 'lightgreen')
        
        # 箭头连接
        for i in range(3):
            self.draw_arrow(ax, 2.5 + i*5, 12, 2.5 + i*5, 13)
        
        # 指向最终结果
        for i in range(3):
            self.draw_arrow(ax, 2.5 + i*5, 6, 7.5, 4.5)
    
    def draw_route_optimization(self):
        """绘制路径优化算法流程"""
        ax = self.axes[0, 2]
        ax.set_title("路径优化算法流程", fontsize=12, fontweight='bold')
        ax.set_xlim(0, 18)
        ax.set_ylim(0, 20)
        ax.axis('off')
        
        # 路径算法选择
        self.draw_box(ax, 7, 18, 4, 1.5, "路径优化算法", 'orange')
        
        # 六种算法
        algorithms = [
            (0.5, 15, 2.5, 1.2, "贪心算法\nO(n²)", 'lightblue'),
            (3.5, 15, 2.5, 1.2, "2-Opt\n局部优化", 'lightgreen'),
            (6.5, 15, 2.5, 1.2, "动态规划\n全局最优", 'lightyellow'),
            (9.5, 15, 2.5, 1.2, "遗传算法\n进化计算", 'lightcoral'),
            (12.5, 15, 2.5, 1.2, "蚁群算法\n群体智能", 'lightpink'),
            (15.5, 15, 2.5, 1.2, "模拟退火\n概率接受", 'lightsteelblue')
        ]
        
        for i, (x, y, w, h, text, color) in enumerate(algorithms):
            self.draw_box(ax, x, y, w, h, text, color)
            self.draw_arrow(ax, 9, 18, x + w/2, y + h)
        
        # 详细流程示例（以遗传算法为例）
        y_pos = 12
        ga_steps = [
            "初始化种群",
            "计算适应度",
            "选择操作",
            "交叉操作",
            "变异操作",
            "新一代种群"
        ]
        
        for i, step in enumerate(ga_steps):
            self.draw_box(ax, 9, y_pos - i*1.5, 3, 1, step, 'lightcoral')
            if i < len(ga_steps) - 1:
                self.draw_arrow(ax, 10.5, y_pos - i*1.5, 10.5, y_pos - (i+1)*1.5 + 1)
        
        # 决策框
        self.draw_diamond(ax, 8.5, 3, 4, 2, "达到最大\n迭代次数?", 'yellow')
        
        # 最终结果
        self.draw_box(ax, 13, 2, 3, 1.5, "输出最优\n路径", 'lightgreen')
        
        # 循环箭头
        self.draw_arrow(ax, 8.5, 4, 5, 4, "否")
        self.draw_arrow(ax, 5, 4, 5, 12, "")
        self.draw_arrow(ax, 12.5, 4, 13, 2.7, "是")
    
    def draw_genetic_algorithm_detail(self):
        """绘制遗传算法详细流程"""
        ax = self.axes[1, 0]
        ax.set_title("遗传算法详细流程", fontsize=12, fontweight='bold')
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 20)
        ax.axis('off')
        
        # 遗传算法流程
        steps = [
            (3, 18, 6, 1.5, "开始：输入客户数据", 'lightgreen'),
            (3, 16, 6, 1.5, "初始化种群\n(随机生成路径)", 'lightblue'),
            (3, 14, 6, 1.5, "计算每个个体的适应度\n(1/距离)", 'lightyellow'),
            (3, 12, 6, 1.5, "选择优秀个体\n(轮盘赌选择)", 'lightcyan'),
            (3, 10, 6, 1.5, "交叉操作\n(顺序交叉)", 'lightcoral'),
            (3, 8, 6, 1.5, "变异操作\n(随机交换)", 'lightpink'),
            (3, 6, 6, 1.5, "生成新一代种群", 'lightsteelblue')
        ]
        
        for i, (x, y, w, h, text, color) in enumerate(steps):
            self.draw_box(ax, x, y, w, h, text, color)
            if i < len(steps) - 1:
                self.draw_arrow(ax, x + w/2, y, x + w/2, y - 0.5)
        
        # 决策框
        self.draw_diamond(ax, 4, 3, 4, 2, "达到最大\n代数?", 'yellow')
        
        # 循环箭头
        self.draw_arrow(ax, 4, 4, 1, 4, "否")
        self.draw_arrow(ax, 1, 4, 1, 14, "")
        self.draw_arrow(ax, 1, 14, 3, 14, "")
        
        # 输出结果
        self.draw_box(ax, 3, 0.5, 6, 1.5, "输出最优个体", 'lightgreen')
        self.draw_arrow(ax, 8, 4, 9, 2, "是")
        self.draw_arrow(ax, 9, 2, 9, 1.25, "")
        
        # 边界条件处理
        self.draw_box(ax, 0.5, 17, 2, 1, "客户数≤1?", 'yellow')
        self.draw_box(ax, 0.5, 15, 2, 1, "返回父代", 'lightgray')
        self.draw_arrow(ax, 1.5, 17, 1.5, 16, "是")
    
    def draw_ant_colony_detail(self):
        """绘制蚁群算法详细流程"""
        ax = self.axes[1, 1]
        ax.set_title("蚁群算法详细流程", fontsize=12, fontweight='bold')
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 20)
        ax.axis('off')
        
        # 蚁群算法流程
        steps = [
            (3, 18, 6, 1.5, "初始化信息素矩阵", 'lightgreen'),
            (3, 16, 6, 1.5, "每只蚂蚁构建路径", 'lightblue'),
            (3, 14, 6, 1.5, "计算转移概率\nP∝τᵃ·ηᵇ", 'lightyellow'),
            (3, 12, 6, 1.5, "轮盘赌选择下一城市", 'lightcyan'),
            (3, 10, 6, 1.5, "计算路径距离", 'lightcoral'),
            (3, 8, 6, 1.5, "信息素蒸发", 'lightpink'),
            (3, 6, 6, 1.5, "信息素增强", 'lightsteelblue')
        ]
        
        for i, (x, y, w, h, text, color) in enumerate(steps):
            self.draw_box(ax, x, y, w, h, text, color)
            if i < len(steps) - 1:
                self.draw_arrow(ax, x + w/2, y, x + w/2, y - 0.5)
        
        # 决策框
        self.draw_diamond(ax, 4, 3, 4, 2, "达到最大\n迭代次数?", 'yellow')
        
        # 循环箭头
        self.draw_arrow(ax, 4, 4, 1, 4, "否")
        self.draw_arrow(ax, 1, 4, 1, 16, "")
        self.draw_arrow(ax, 1, 16, 3, 16, "")
        
        # 输出结果
        self.draw_box(ax, 3, 0.5, 6, 1.5, "输出最优路径", 'lightgreen')
        self.draw_arrow(ax, 8, 4, 9, 2, "是")
        self.draw_arrow(ax, 9, 2, 9, 1.25, "")
        
        # 参数说明
        self.draw_box(ax, 0.5, 19, 2, 0.8, "α:信息素重要度\nβ:启发式重要度", 'lightyellow')
    
    def draw_simulated_annealing_detail(self):
        """绘制模拟退火算法详细流程"""
        ax = self.axes[1, 2]
        ax.set_title("模拟退火算法详细流程", fontsize=12, fontweight='bold')
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 20)
        ax.axis('off')
        
        # 模拟退火算法流程
        steps = [
            (3, 18, 6, 1.5, "生成初始解", 'lightgreen'),
            (3, 16, 6, 1.5, "设置初始温度T", 'lightblue'),
            (3, 14, 6, 1.5, "生成邻居解\n(随机交换)", 'lightyellow'),
            (3, 12, 6, 1.5, "计算能量差Δ", 'lightcyan'),
            (3, 10, 6, 1.5, "计算接受概率\ne^(-Δ/T)", 'lightcoral'),
            (3, 8, 6, 1.5, "更新当前解", 'lightpink'),
            (3, 6, 6, 1.5, "降低温度\nT = T × α", 'lightsteelblue')
        ]
        
        for i, (x, y, w, h, text, color) in enumerate(steps):
            self.draw_box(ax, x, y, w, h, text, color)
            if i < len(steps) - 1:
                self.draw_arrow(ax, x + w/2, y, x + w/2, y - 0.5)
        
        # 决策框们
        self.draw_diamond(ax, 1, 11, 3, 1.5, "Δ<0或\n随机接受?", 'yellow')
        self.draw_diamond(ax, 4, 3, 4, 2, "T<T_min?", 'yellow')
        
        # 边界条件处理
        self.draw_box(ax, 0.5, 17, 2, 0.8, "客户数<2?", 'yellow')
        self.draw_box(ax, 0.5, 15.5, 2, 0.8, "返回原解", 'lightgray')
        self.draw_arrow(ax, 1.5, 17, 1.5, 16.3, "是")
        
        # 循环箭头
        self.draw_arrow(ax, 4, 4, 1, 4, "否")
        self.draw_arrow(ax, 1, 4, 1, 14, "")
        self.draw_arrow(ax, 1, 14, 3, 14, "")
        
        # 条件箭头
        self.draw_arrow(ax, 2.5, 11.5, 3, 10.5, "是")
        self.draw_arrow(ax, 1, 11, 0.5, 11, "否")
        self.draw_arrow(ax, 0.5, 11, 0.5, 14, "")
        
        # 输出结果
        self.draw_box(ax, 3, 0.5, 6, 1.5, "输出最优解", 'lightgreen')
        self.draw_arrow(ax, 8, 4, 9, 2, "是")
        self.draw_arrow(ax, 9, 2, 9, 1.25, "")
    
    def save_flowchart(self, filename='algorithm_flowchart.png'):
        """保存流程图"""
        # 绘制所有流程图
        self.draw_main_flowchart()
        self.draw_partition_algorithms()
        self.draw_route_optimization()
        self.draw_genetic_algorithm_detail()
        self.draw_ant_colony_detail()
        self.draw_simulated_annealing_detail()
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

def create_algorithm_flowchart():
    """创建并显示算法流程图"""
    flowchart = AlgorithmFlowchartVisualization()
    flowchart.save_flowchart('d:/CodeSpace/algorithm/algorithm_flowchart.png')
    print("算法流程图已保存到: d:/CodeSpace/algorithm/algorithm_flowchart.png")

if __name__ == "__main__":
    create_algorithm_flowchart()
