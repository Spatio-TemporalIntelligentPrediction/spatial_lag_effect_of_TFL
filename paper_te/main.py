# main.py
import time
import os
import numpy as np
from multiprocessing import Pool, cpu_count

import config as cfg
from data_utils import load_nc_data, make_offsets
from te_core import get_estimator
from plot_utils import plot_te_map

# 全局变量
global_temp = None
global_offsets = None
global_params = {} 

def init_worker(temp, offsets, params):
    global global_temp, global_offsets, global_params
    global_temp = temp
    global_offsets = offsets
    global_params = params
    # 预热
    get_estimator()

def sample_point_te_task(args):
    """
    单个点的 TE 计算任务
    """
    i, j, level, current_lag = args
    
    temp = global_temp
    offsets = global_offsets
    nlat = global_params['nlat']
    nlon = global_params['nlon']
    lon_shift = global_params['lon_shift']
    
    # 这里的 window 直接从 config 读取，或者也可以放入 args 传递
    window = cfg.WINDOW_SIZE
    
    try:
        # 1. 获取目标序列 (Target)
        # 直接取完整的列，不切片！JIDT 会自己处理
        target_series = temp[:, i, j]
        
        # 检查是否含 NaN
        if np.any(np.isnan(target_series)):
            return (i, j, np.nan)

        # 2. 获取所有邻居序列 (Sources)
        offs = offsets[level]
        neighbors_series = []
        
        for di, dj in offs:
            ni, nj = i + di, (j + dj) % nlon
            # 边界处理
            if ni < 0:
                ni = -ni
                nj = (nj + lon_shift) % nlon
            elif ni >= nlat:
                ni = 2 * (nlat - 1) - ni
                nj = (nj + lon_shift) % nlon
            
            # 取邻居完整序列
            n_ts = temp[:, ni, nj]
            
            if not np.any(np.isnan(n_ts)):
                neighbors_series.append(n_ts)
        
        if not neighbors_series:
            return (i, j, np.nan)

        # 3. 计算平均成对 TE (调用新核心)
        estimator = get_estimator()
        
        # 核心改变：传入列表、中心序列、窗口(history) 和 滞后(delay)
        # 所有的切片逻辑都在 JIDT 内部完成了
        avg_te_bits = estimator.compute_avg_pairwise_te(
            neighbors_series, 
            target_series, 
            window=window, 
            lag=current_lag
        )
        
        return (i, j, avg_te_bits)

    except Exception as e:
        # print(f"Error: {e}")
        return (i, j, np.nan)

def main():
    print(f"=== 启动平均成对 TE 分析 (JIDT Native) ===")
    print(f"算法: {cfg.ESTIMATOR_TYPE} | Window: {cfg.WINDOW_SIZE} | Max Lag: {cfg.LAG_MAX}")
    
    # 1. 加载数据
    # 这里加载完整的 DAYS 长度即可，JIDT 会根据 lag 自动消耗掉开头的数据
    # 为了安全，我们多加载一点点数据，或者就按 DAYS 加载
    # JIDT 如果发现 delay=20，history=5，它会自动从第 26 个点开始计算
    temp, lats, lons, lon_shift = load_nc_data(
        cfg.NC_PATH, cfg.VAR_NAME, cfg.DAYS, lag_max=cfg.LAG_MAX + cfg.WINDOW_SIZE
    )
    nlat, nlon = temp.shape[1:]
    
    params = {'nlat': nlat, 'nlon': nlon, 'lon_shift': lon_shift}
    offsets = make_offsets(cfg.MAX_LEVEL, cfg.INCLUDE_LOWER)
    
    # 结果汇总矩阵
    global_ei_summary = np.full((cfg.LAG_MAX + 1, cfg.MAX_LEVEL + 1), np.nan)

    # 2. 循环遍历 Lag 和 Level
    for lag in range(1, cfg.LAG_MAX + 1):
        print(f"\n=========== LAG (DELAY) = {lag} | WINDOW (HISTORY) = {cfg.WINDOW_SIZE} ===========")
        
        for level in range(1, cfg.MAX_LEVEL + 1):
            t0 = time.time()
            
            # 任务参数: (i, j, level, lag)
            coords = [(i, j, level, lag) for i in range(nlat) for j in range(nlon)]
            if cfg.SAMPLE_LIMIT:
                coords = coords[:cfg.SAMPLE_LIMIT]
            
            n_cores = cfg.CPU_CORES if cfg.CPU_CORES else cpu_count()
            
            with Pool(n_cores, initializer=init_worker, initargs=(temp, offsets, params)) as pool:
                results = pool.map(sample_point_te_task, coords)
            
            # 聚合
            mi_map = np.full((nlat, nlon), np.nan)
            for ii, jj, val in results:
                mi_map[ii, jj] = val
            
            # 保存
            save_name = f"te_map_level_{level}_shift_{lag}.npy"
            np.save(os.path.join(cfg.MATRIX_PATH, save_name), mi_map)
            
            EI = float(np.nanmean(mi_map))
            global_ei_summary[lag, level] = EI
            
            # 绘图 (这里需要稍微修改 plot_utils，因为入参变了)
            # 为了简单，我们这里假设 plot_te_map 还能用，或者传入假参数适配
            # 实际上我们不需要 min_lag/max_lag 了，只有一个 lag
            # 我们可以临时传 lag 作为 min, lag 作为 max
            plot_te_map(mi_map, lons, lats, level, lag, lag, EI, cfg.FIGURE_PATH)
            
            print(f"    Level {level}, Lag {lag}: Avg TE = {EI:.4f} bits, Time = {time.time()-t0:.1f}s")
    
    # 保存汇总
    np.save(os.path.join(cfg.MATRIX_PATH, "global_te_matrix_avg_pairwise.npy"), global_ei_summary)
    print("\n=== 计算完成 ===")

if __name__ == "__main__":
    main()