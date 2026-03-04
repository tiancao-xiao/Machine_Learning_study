# 导包
import sys
import os

# 核心：获取train.py所在目录（src）的父目录（即项目根目录load_predict_project）
# __file__ 指当前train.py文件的绝对路径
current_file_dir = os.path.dirname(os.path.abspath(__file__))  # 获取src文件夹路径
project_root = os.path.dirname(current_file_dir)  # 获取项目根目录（src的上一级）
sys.path.append(project_root)  # 将项目根目录添加到系统路径中，这样就可以导入项目中的模块了

import pandas as pd
import matplotlib.pyplot as plt
import datetime
from utils.log import Logger
from utils.common import data_preprocessing
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
import joblib

import matplotlib
# 切换到TkAgg后端（通用绘图后端，兼容所有环境）
matplotlib.use('TkAgg')

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 15

# 1. 定义电力负荷模型类, 配置日志, 获取数据源
class PowerLoadModel:
    # 1.1 初始化属性信息
    def __init__(self, file_path):
        # 1.2 拼接日志文件名
        logfile_name = 'train_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        # 1.3 创建日志对象
        # .get_logger()方法是Logger类中的一个方法，用于获取一个日志对象，这个对象可以用来记录日志信息。
        # 通过调用这个方法，我们可以在代码中使用这个日志对象来记录各种级别的日志（如info、debug、error等），以便在运行时跟踪程序的执行情况和调试问题。
        self.logfile = Logger('../', logfile_name).get_logger()
        # 测试写一条日志
        self.logfile.info('开始创建 电力负荷模型类的 对象了')
        # 1.4 获取数据源
        # data_preprocessing()函数是从utils.common模块中导入的一个函数，通常用于对原始数据进行预处理和清洗，以便后续的分析和建模工作。
        self.data_source = data_preprocessing(file_path)


# 2. 查看数据的整体分布情况
def ana_data(data):     # analysis: 分析
    """
    1.查看数据整体情况
    2.负荷整体的分布情况
    3.各个小时的平均负荷趋势，看一下负荷在一天中的变化情况
    4.各个月份的平均负荷趋势，看一下负荷在一年中的变化情况
    5.工作日与周末的平均负荷情况，看一下工作日的负荷与周末的负荷是否有区别
    :param data: 数据源
    :return:
    """
    # 2.0 为了防止会修改源数据, 我们做一次拷贝
    ana_data = data.copy()

    # 2.1 查看数据整体情况
    ana_data.info()

    # 2.2 负荷整体的分布情况, 直方图
    # 2.2.1 创建画布
    fig = plt.figure(figsize=(6, 8))
    # 设置子图垂直间距（hspace=0.5），避免标题/标签重合
    fig.subplots_adjust(hspace=0.5)
    # 2.2.2 添加子图
    ax1 = fig.add_subplot(411)
    # 参1：数据，参2：区间数量，参3：柱子颜色，参4：柱子透明度
    ax1.hist(ana_data['power_load'], bins=100, color='#1f77b4', alpha=0.7)
    # fontsize：字体大小，fontweight：字体粗细
    ax1.set_xlabel('负荷', fontsize=12)
    ax1.set_ylabel('频次', fontsize=12)  # 补充y轴标签
    ax1.set_title('负荷整体分布情况', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3) # 添加网格线，增强可读性

    # 2.3 各个小时的平均负荷趋势，看一下负荷在一天中的变化情况
    # 2.3.1 新增一列，充当小时（转为数值类型，避免字符串排序混乱）
    ana_data['hour'] = ana_data['time'].str[11:13].astype(int)
    # print(ana_data.head())
    # 2.3.2 根据小时分组，计算平均值
    # groupby()方法是pandas库中用于分组数据的函数。
    # 它允许我们根据一个或多个列的值将数据分成不同的组，然后对每个组进行聚合操作（如计算平均值、求和等）。
    # 在这里，ana_data.groupby('hour', as_index=False)['power_load'].mean()的作用是：
    # 1） 将ana_data数据框按照'hour'列的值进行分组。
    # 2） 对每个小时组内的'power_load'列计算平均值。
    # 3） as_index=False参数表示在结果中保留'hour'列作为普通列，而不是将其设置为索引。
    hour_load_mean = ana_data.groupby('hour', as_index=False)['power_load'].mean()
    # print(hour_load_mean)       # [列1 hour, 列2 power_load 当前小时的平均负荷]
    # 2.3.3 画出折线图（4行1列第二个位置：412）
    ax2 = fig.add_subplot(412)
    ax2.plot(hour_load_mean['hour'], hour_load_mean['power_load'], color='#ff7f0e', linewidth=2)
    ax2.set_title('各个小时的平均负荷趋势', fontsize=14, fontweight='bold')
    ax2.set_xlabel('小时', fontsize=12)
    ax2.set_ylabel('平均负荷', fontsize=12)
    ax2.set_xticks(range(0, 24, 2))  # 优化x轴刻度，避免拥挤
    ax2.grid(alpha=0.3) # 添加网格线，增强可读性

    # 2.4 各个月份的平均负荷趋势，看一下负荷在一年中的变化情况
    # 2.4.1 新增一列，充当月份（转为数值类型，避免字符串排序混乱）
    ana_data['month'] = ana_data['time'].str[5:7].astype(int)
    # 2.4.2 根据月份分组，计算平均值
    month_load_mean = ana_data.groupby('month', as_index=False)['power_load'].mean()
    # 2.4.3 画出折线图（4行1列第三个位置：413）
    ax3 = fig.add_subplot(413)
    # 参1：x轴数据，参2：y轴数据，color：线条颜色，linewidth：线条宽度
    ax3.plot(month_load_mean['month'], month_load_mean['power_load'], color='#2ca02c', linewidth=2)
    # 设置标题，字体大小14，字体加粗
    ax3.set_title('各个月份的平均负荷趋势', fontsize=14, fontweight='bold')
    ax3.set_xlabel('月份', fontsize=12)
    ax3.set_ylabel('平均负荷', fontsize=12)
    ax3.set_xticks(range(1, 13))  # 优化x轴刻度，避免拥挤
    ax3.grid(alpha=0.3) # 添加网格线，增强可读性

    # 2.5 工作日与周末的平均负荷情况，看一下工作日的负荷与周末的负荷是否有区别
    # 2.5.1 新增一列，充当工作日/周末（1：工作日，0：周末）
    # weekday()方法是Python中datetime模块中的一个函数，用于返回一个日期对象对应的星期几。
    # 返回值是一个整数，范围从0到6，其中0表示星期一，1表示星期二，以此类推，6表示星期日。
    ana_data['week_day'] = ana_data['time'].apply(lambda x: pd.to_datetime(x).weekday())
    ana_data['is_workday'] = ana_data['week_day'].apply(lambda x: 1 if x <= 4 else 0)
    # 2.5.2 分别计算工作日和周末的平均负荷
    power_load_workday_avg = ana_data[ana_data['is_workday'] == 1]['power_load'].mean()
    power_load_weekend_avg = ana_data[ana_data['is_workday'] == 0]['power_load'].mean()
    # 2.5.3 画出柱状图（4行1列第四个位置：414）
    ax4 = fig.add_subplot(414)
    # 参1：x轴数据，参2：y轴数据，color：柱子颜色，alpha：柱子透明度
    ax4.bar(x=['工作日平均负荷', '周末平均负荷'], height=[power_load_workday_avg, power_load_weekend_avg], color=['#d62728', '#9467bd'], alpha=0.7)
    ax4.set_title('工作日与周末的平均负荷对比', fontsize=14, fontweight='bold')
    ax4.set_ylabel('平均负荷', fontsize=12)
    ax4.grid(axis='y', alpha=0.3) # 添加水平网格线，增强可读性

    # tight_layout() 自动调整布局，兜底避免元素重叠
    plt.tight_layout()
    plt.savefig('../data/fig/负荷整体的分布情况.png')
    plt.show()


# 3. 特征工程（重点）
def feature_engineering(data, logger):
    """
    对给定的数据源，进行特征工程处理，提取出关键的特征
    1.提取出时间特征：月份、小时
    2.提取出相近时间窗口中的负荷特征：step大小窗口的负荷
    3.提取昨日同时刻负荷特征
    4.剔除出现空值的样本
    5.整理时间特征，并返回
    :param data: 数据源
    :param logger: 日志
    :return:
    """
    # 先拷贝源数据
    feature_data = data.copy(deep=True) # 深拷贝，完全独立于原数据，修改feature_data不会影响data

    # 3.1 提取出时间特征：月份、小时
    feature_data['hour'] = feature_data['time'].str[11:13]
    feature_data['month'] = feature_data['time'].str[5:7]
    # 热编码 one-hot 处理 hour 和 month 特征
    hour_month_data = pd.get_dummies(feature_data[['hour', 'month']])
    # print(hour_month_data.head(10))
    # print(hour_month_data.info())
    # 将热编码后的特征与原数据进行拼接
    feature_data = pd.concat([feature_data, hour_month_data], axis=1)
    # print(feature_data.head(10))
    # print(feature_data.info())

    # 3.2 提取出相近时间窗口中的负荷特征：step大小窗口的负荷
    # 3.2.1 获取上1个小时的负荷
    # shift()方法是pandas库中用于数据平移的函数。它可以将数据沿着指定的轴进行平移，常用于时间序列数据的特征工程中。
    # shift(1)表示将数据向下平移1行，即当前行的值变为上一行的值，当前行的值会被NaN填充。
    # shift(-1)表示将数据向上平移1行，即当前行的值变为下一行的值，当前行的值会被NaN填充。
    load_1h_data = feature_data['power_load'].shift(1)
    # 3.2.2 获取上2个小时的负荷
    load_2h_data = feature_data['power_load'].shift(2)
    load_3h_data = feature_data['power_load'].shift(3)
    # 3.2.3 将上1小时、上2小时、上3小时的负荷特征进行拼接
    load_shift_df = pd.concat([load_1h_data, load_2h_data, load_3h_data], axis=1)
    # 3.2.4 修改列名
    load_shift_df.columns = ['前1小时', '前2小时', '前3小时']
    # 3.2.5 将相近时间窗口中的负荷特征与原数据进行拼接
    feature_data = pd.concat([feature_data, load_shift_df], axis=1)
    # print(feature_data.info())

    # 3.3 提取昨日同时刻负荷特征
    # 3.3.1 给特征新增1列名，yesterday_time
    # apply()方法是pandas库中用于对DataFrame或Series中的每个元素应用一个函数的函数。
    # 它可以接受一个函数作为参数，并将该函数应用于DataFrame或Series中的每个元素，返回一个新的DataFrame或Series。
    # lambda x: (pd.to_datetime(x) - pd.to_timedelta('1d')).strftime('%Y-%m-%d %H:%M:%S')的作用是：
    # 1） 将输入的时间字符串x转换为datetime对象。
    # 2） 从该datetime对象中减去1天（pd.to_timedelta  ('1d')表示1天的时间差）。
    # 3） 将结果转换回字符串格式，格式为'年-月-日 时:分:秒'。
    feature_data['yesterday_time'] = feature_data['time'].apply(lambda x: (pd.to_datetime(x) - pd.to_timedelta('1d')).strftime('%Y-%m-%d %H:%M:%S'))
    # print(feature_data.head(30))
    # 3.3.2 把所有的日期和负荷拼接成字典，方便查找
    # set_index()方法是pandas库中用于设置DataFrame索引的函数。它可以将指定的列设置为DataFrame的索引，使得该列的值成为行标签。
    # to_dict()方法是pandas库中用于将DataFrame或Series转换为字典的函数。它可以将DataFrame或Series中的数据转换为字典格式，方便进行数据查找和操作。
    # 在下面的代码中，feature_data.set_index('time')['power_load'].to_dict()的作用是：
    # 1） 将feature_data DataFrame中的'time'列设置为索引
    # 2） 从设置了索引的DataFrame中选择'power_load'列，得到一个Series对象。
    # 3） 将这个Series对象转换为字典，其中键是'time'列的值，值是'power_load'列的值。这样就可以通过时间来查找对应的负荷值了。
    time_load_dict = feature_data.set_index('time')['power_load'].to_dict()
    # print(time_load_dict)
    # 3.3.3 新增1列，yesterday_load，表示：昨天相同时刻的负荷
    # 在下面的代码中，feature_data['yesterday_time'].apply(lambda x: time_load_dict.get(x))的作用是：
    # 1） 对feature_data DataFrame中的'yesterday_time'列的每个元素应用一个lambda函数。
    # 2） lambda函数接受一个输入x（即昨天相同时刻的时间字符串），并使用time_load_dict.get(x)来查找这个时间对应的负荷值。
    # 3） get()方法会返回time_load_dict中键为x的值，如果x不存在于字典中，则返回None（或者可以指定一个默认值）。
    # 3) 这样就可以得到昨天相同时刻的负荷值，并将其赋值给'yesterday_load'列了。
    feature_data['yesterday_load'] = feature_data['yesterday_time'].apply(lambda x: time_load_dict.get(x))
    # print(feature_data.head(30))
    # feature_data.info()

    # 3.4 剔除出现空值的样本
    # dropna()方法是pandas库中用于删除包含缺失值的行或列的函数。它可以根据指定的条件删除DataFrame中的行或列。
    # 默认axis=0表示删除包含缺失值的行，axis=1表示删除包含缺失值的列。默认情况下，dropna()会删除任何包含NaN值的行或列。
    feature_data = feature_data.dropna()

    # 3.5 整理时间特征，并返回
    feature_columns = list(hour_month_data.columns) + list(load_shift_df.columns) + ['yesterday_load']
    # print(f"特征列名是：{feature_columns}")
    """
    ['hour_00', 'hour_01', 'hour_02', 'hour_03', 'hour_04', 'hour_05', 'hour_06', 'hour_07', 
    'hour_08', 'hour_09', 'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15', 
    'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23', 
    'month_01', 'month_02', 'month_03', 'month_04', 'month_05', 'month_06', 'month_07', 'month_08', 
    'month_09', 'month_10', 'month_11', 'month_12', '前1小时', '前2小时', '前3小时', 'yesterday_load']
    """

    # 3.6 返回结果
    return feature_data, feature_columns


# 4. 模型训练，评估，保存
def model_train(data, features, logger):
    """
    1.数据集切分
    2.网格化搜索与交叉验证
    3.模型实例化
    4.模型训练
    5.模型评价
    6.模型保存
    :param data: 特征工程处理后的输入数据
    :param features: 特征名称
    :param logger: 日志对象
    :return:
    """
    # 4.1 数据集切分
    x = data[features]
    y = data['power_load']
    # print(x.shape, y.shape)
    # print(x.head(10))
    # print(y.head(10))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

    # # 4.2 网格化搜索与交叉验证
    # logger.info('=========开始网格化搜索+交叉验证 寻找最优超参=========')
    # logger.info(f'开始时间：{datetime.datetime.now()}')
    # # 4.2.1 定义参数字典
    # param_dict = {
    #     'n_estimators': [50, 100, 150, 200],
    #     'max_depth': [3, 5, 6, 7],
    #     'learning_rate': [0.01, 0.1]
    #
    # }
    # # 4.2.2 创建XGBoost模型对象
    # estimator = XGBRegressor()
    # # 4.2.3 创建网格搜索对象
    # gs = GridSearchCV(estimator=estimator, param_grid=param_dict, cv=5)
    # # 4.2.4 模型训练
    # gs.fit(x_train, y_train)
    # # 4.2.5 打印最优的超参数组合
    # logger.info(f'最优的超参数组合是：{gs.best_params_}')
    # logger.info(f'结束时间：{datetime.datetime.now()}')

    # 4.3 模型实例化
    estimator = XGBRegressor(n_estimators=100, max_depth=7, learning_rate=0.1)

    # 4.4 模型训练
    estimator.fit(x_train, y_train)
    y_pred = estimator.predict(x_test)

    # 4.5 模型评价
    print(f'均方误差（MSE）: {mean_squared_error(y_test, y_pred)}')
    print(f'均方根误差（RMSE）: {root_mean_squared_error(y_test, y_pred)}')
    print(f'平均绝对误差（MAE）: {mean_absolute_error(y_test, y_pred)}')
    print(f'平均绝对百分比误差（MAPE）: {mean_absolute_percentage_error(y_test, y_pred)}')

    # 4.6 模型保存
    joblib.dump(estimator, '../model/power_load_xgb_model.pkl')     # pickle文件 -> 后缀名一般是.pkl .pth .pickle
    logger.info(f'模型保存成功，保存路径：{os.path.abspath("../model/power_load_xgb_model.pkl")}')


# 5. 测试
if __name__ == '__main__':
    # 5.1 创建电力负荷模型类的对象
    pm = PowerLoadModel('../data/train.csv')

    # 5.2 打印数据源
    # print(pm.data_source)

    # 5.3 查看数据的整体分布情况
    # ana_data(pm.data_source)

    # 5.4 特征工程
    feature_data, feature_columns = feature_engineering(pm.data_source, pm.logfile)
    # print(feature_data.head(10))
    # print(feature_columns)

    # 5.5 模型训练，评估，保存
    # 参1：处理后的全部数据集； 参2：特征列名；参3：日志对象
    model_train(feature_data, feature_columns, pm.logfile)