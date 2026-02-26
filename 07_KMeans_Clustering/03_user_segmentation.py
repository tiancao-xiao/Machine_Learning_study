"""
案例：
    基于用户的年收入和消费指数，根据用户的相似性进行分类
"""

# 导包
import os
os.environ['OMP_NUM_THREADS'] = '1'     # 设置OMP程序运行时使用的线程数
os.environ['TK_SILENCE_DEPRECATION'] = '1'  # 抑制tkinter弃用警告

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score, silhouette_score
import pandas as pd

# 解决matplotlib后端问题：强制使用TkAgg后端（兼容PyCharm）
plt.switch_backend('TkAgg')

plt.rcParams['font.sans-serif'] = ['SimHei']    # 指定默认字体为黑体（支持中文）
plt.rcParams['axes.unicode_minus'] = False      # 解决负号显示为方块的问题

# 1. 定义函数，找聚类的质心数
def dm01_find_k():
    # 1. 加载数据集
    df = pd.read_csv('./data/customers.csv')
    # df.info()
    # print(df.head())

    # 2. 定义sse_list, sc_list, 记录：不同k值的评估效果
    sse_list = []       # sse: 只考虑簇内，越小越好
    sc_list = []        # sc: 考虑簇内和簇间，越大越好
    # 抽取特征 把原 df 中所有行、第 4 和第 5 列提取出来，结果 x 是一个 DataFrame（包含这两列）
    x = df.iloc[:, 3:5]
    print(x)

    # 3. 定义for训练，测试不同k值的评估效果
    for k in range(2, 20):
        # 3.1 创建(KMeans模型)对象
        estimator = KMeans(n_clusters=k, max_iter=100, random_state=23)
        # 3.2 模型训练
        estimator.fit(x)
        # 3.3 模型预测
        y_pred = estimator.predict(x)
        # 3.4 分别把评分添加到对应的列表中
        sse_list.append(estimator.inertia_)
        sc_list.append(silhouette_score(x, y_pred))

    #  4. 绘制折线图，看看k值哪个最好
    # ========== 核心修改：将两张图放在同一个画布的子图中，一次显示 ==========
    plt.figure(figsize=(12, 6))

    # 子图1：SSE折线图
    plt.subplot(1, 2, 1)    # 生成1行2列，在第1个位置
    plt.plot(range(2, 20), sse_list, label='SSE', color='blue', linewidth=1.5)
    plt.title('不同K值的SSE变化（肘部法则）', fontsize=11)  # 适配字体大小
    plt.xlabel('聚类数K', fontsize=10)
    plt.ylabel('SSE（簇内平方和）', fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)  # 网格透明化，避免视觉拥挤

    # 子图2：SC折线图
    plt.subplot(1, 2, 2)
    plt.plot(range(2, 20), sc_list, label='SC', color='orange', linewidth=1.5)
    plt.title('不同K值的轮廓系数变化', fontsize=11)
    plt.xlabel('聚类数K', fontsize=10)
    plt.ylabel('SC（轮廓系数）', fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    # 精细化调整间距：pad增加子图内外边距，h_pad/v_pad调整横竖间距
    plt.tight_layout(pad=2, h_pad=1, w_pad=2)
    plt.show()

    # 结论：k=5时效果最好

# 2. 定义函数，实现：模型训练，模型预测，模型评估
def dm02_train_predict_evaluate():
    # 1. 加载数据集
    df = pd.read_csv('./data/customers.csv')

    # 2. 提取特征
    x = df.iloc[:, 3:5]
    # print(x.head())
    # print(x.values)

    # 3. 模型训练 k = 5
    estimator = KMeans(n_clusters=5, max_iter=100, random_state=23)
    estimator.fit(x)

    # 4. 模型预测
    y_pred = estimator.predict(x)
    # print(y_pred)

    # 5. 绘制5个簇的样本点 -> 散点图

    # 定义每个簇的颜色和标签（方便区分）
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    cluster_labels = ['簇0：中等收入+中等消费', '簇1：高收入+低消费', '簇2：低收入+低消费', '簇3：低收入+高消费',
                      '簇4：高收入+高消费']

    # 绘制每个簇的样本点（加label参数，用于生成图例）
    plt.scatter(x.values[y_pred == 0, 0], x.values[y_pred == 0, 1], c=colors[0], label=cluster_labels[0])  # 0号簇
    plt.scatter(x.values[y_pred == 1, 0], x.values[y_pred == 1, 1], c=colors[1], label=cluster_labels[1])  # 1号簇
    plt.scatter(x.values[y_pred == 2, 0], x.values[y_pred == 2, 1], c=colors[2], label=cluster_labels[2])  # 2号簇
    plt.scatter(x.values[y_pred == 3, 0], x.values[y_pred == 3, 1], c=colors[3], label=cluster_labels[3])  # 3号簇
    plt.scatter(x.values[y_pred == 4, 0], x.values[y_pred == 4, 1], c=colors[4], label=cluster_labels[4])  # 4号簇

    # 6. 绘制5个簇的质心 -> 散点图
    # print(estimator.cluster_centers_)       # 5个质心的坐标
    # 绘制质心（加图例标签）
    centroids = estimator.cluster_centers_
    plt.scatter([c[0] for c in centroids], [c[1] for c in centroids],
                s=200, c='black', marker='*', label='簇质心', edgecolors='white')

    # 7. 设置标题

    # 7. 设置标题，x轴，y轴标签
    plt.title('Clusters of Customers', fontsize=12)
    plt.xlabel('Annual Income(k$)', fontsize=11)
    plt.ylabel('Spending Score (1-100)', fontsize=11)
    plt.legend(loc='best', fontsize=9)  # 显示图例，loc='best'自动找最优位置
    plt.grid(True, alpha=0.3)  # 加网格，更清晰
    plt.show()


# 3. 测试
if __name__ == '__main__':
    # dm01_find_k()
    dm02_train_predict_evaluate()



