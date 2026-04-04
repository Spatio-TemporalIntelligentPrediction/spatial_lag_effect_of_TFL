# config.py
import os

# ————————————————————————
# 路径配置
# ————————————————————————
NC_PATH = "causal/data/2m_temperature_hourly.nc"
FIGURE_PATH = "causal/paper_te/figure_te_avg_pairwise_1"
MATRIX_PATH = "causal/paper_te/matrix_te_avg_pairwise_1"

os.makedirs(FIGURE_PATH, exist_ok=True)
os.makedirs(MATRIX_PATH, exist_ok=True)

# ————————————————————————
# 数据参数
# ————————————————————————
VAR_NAME = "t2m"
MAX_LEVEL = 20        # 最大空间阶数
SAMPLE_LIMIT = None   # None 为跑全图
DAYS = 7              # 样本时间长度 (建议长一点，pairwise计算需要较多样本)
CPU_CORES = None

# ————————————————————————
# 关键 TE 参数 (修改部分)
# ————————————————————————
# WINDOW_SIZE (历史长度): 
# 表示我们要查看过去多少个小时的状态模式。
# 对应 JIDT 的 history_source 和 history_target
WINDOW_SIZE = 1       

# LAG (传输延迟):
# 表示源变量的过去要在多久之后才影响目标变量。
# 对应 JIDT 的 source_target_delay
# 循环范围: 1 到 LAG_MAX
LAG_MAX = 24          

INCLUDE_LOWER = False # 邻域形状

# ————————————————————————
# 算法选择
# ————————————————————————
# 'gaussian': 线性、速度快 (推荐用于全图扫描)
# 'kraskov': 非线性、速度慢 (但在做 Pairwise 时比 Collective 快一些)
ESTIMATOR_TYPE = 'gaussian' 

KRASKOV_SETTINGS = {
    'k': 4,
    'algorithm_num': 1
}