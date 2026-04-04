import time
import os
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

import config as cfg
from data_utils import load_nc_data, make_offsets, get_land_mask
from te_core import get_estimator

# ————————————————————————
# 配置
# ————————————————————————
TARGET_DAYS = 30         
TARGET_HOURS = range(24) 
FIXED_LEVEL = 1          
FIXED_LAG = 1            

HOURLY_FIGURE_PATH = "causal/paper_te/figure_hourly_analysis_1"
HOURLY_MATRIX_PATH = "causal/paper_te/matrix_hourly_analysis_1"
os.makedirs(HOURLY_FIGURE_PATH, exist_ok=True)
os.makedirs(HOURLY_MATRIX_PATH, exist_ok=True)

# 全局变量
global_temp = None
global_offsets = None
global_params = {} 
global_valid_indices = None 

def init_worker_hourly(temp, offsets, params, valid_indices):
    global global_temp, global_offsets, global_params, global_valid_indices
    global_temp = temp
    global_offsets = offsets
    global_params = params
    global_valid_indices = valid_indices 
    get_estimator() # 预热 JVM

def sample_point_hourly_pairwise_task(args):
    """
    计算单点的 Hourly Average Pairwise TE
    逻辑：遍历邻居 -> 手动切片 -> CMI -> 平均
    """
    i, j, level, lag = args
    
    temp = global_temp
    offsets = global_offsets
    valid_indices = global_valid_indices 
    
    nlat = global_params['nlat']
    nlon = global_params['nlon']
    lon_shift = global_params['lon_shift']
    window = cfg.WINDOW_SIZE
    
    try:
        if valid_indices is None or len(valid_indices) == 0:
            return (i, j, np.nan)

        # ————————————————————————————————————
        # 1. 准备中心点数据 (Y)
        # ————————————————————————————————————
        y_full_series = temp[:, i, j]
        
        # Y_now: 时刻 t 的值 (N_samples, 1)
        Y_input = y_full_series[valid_indices].reshape(-1, 1)
        
        # Y_past: 时刻 [t-1 ... t-window] (N_samples, window)
        # 广播减法构建索引: indices[:, None] - lags[None, :]
        hist_lags = np.arange(1, window + 1)
        y_hist_indices = valid_indices[:, None] - hist_lags[None, :]
        Z_input = y_full_series[y_hist_indices] 

        # 检查 Y 数据是否包含 NaN
        if np.isnan(Y_input).any() or np.isnan(Z_input).any():
            return (i, j, np.nan)

        # ————————————————————————————————————
        # 2. 遍历邻居计算 CMI (Pairwise)
        # ————————————————————————————————————
        offs = offsets[level]
        te_sum = 0.0
        valid_neighbor_count = 0
        
        estimator = get_estimator()
        
        # 预计算 X 的滞后索引 (对所有邻居通用)
        # X_past 时刻: [t-lag ... t-lag-window+1]
        x_lags = np.arange(lag, lag + window)
        x_hist_indices = valid_indices[:, None] - x_lags[None, :]
        
        for di, dj in offs:
            ni, nj = i + di, (j + dj) % nlon
            # 边界处理
            if ni < 0:
                ni = -ni
                nj = (nj + lon_shift) % nlon
            elif ni >= nlat:
                ni = 2 * (nlat - 1) - ni
                nj = (nj + lon_shift) % nlon
            
            # 获取邻居全序列
            x_full_series = temp[:, ni, nj]
            
            # 切片获取 X_past (N_samples, window)
            X_input = x_full_series[x_hist_indices]
            
            # 检查 X 数据
            if np.isnan(X_input).any():
                continue

            # 调用 CMI 接口
            # I(X_past; Y_now | Y_past)
            # 结果已在 te_core 中转为 bits 并剔除负值
            val = estimator.compute_cmi(X_input, Y_input, Z_input)
            
            if not np.isnan(val):
                te_sum += val
                valid_neighbor_count += 1
        
        # 3. 求平均
        if valid_neighbor_count > 0:
            avg_te = te_sum / valid_neighbor_count
            return (i, j, avg_te)
        else:
            return (i, j, np.nan)

    except Exception:
        return (i, j, np.nan)

def generate_hourly_indices(target_hour, total_hours, days, lag, window):
    indices = []
    # 确保最远回溯点 (lag + window) 不越界
    min_required_idx = lag + window
    for d in range(days):
        t = d * 24 + target_hour
        if t >= min_required_idx and t < total_hours:
            indices.append(t)
    return np.array(indices, dtype=int)

def main():
    print(f"=== 启动日变化 TE 分析 (Hourly Pairwise CMI) ===")
    
    # 1. 加载数据
    temp, lats, lons, lon_shift = load_nc_data(cfg.NC_PATH, cfg.VAR_NAME, days=None)
    nlat, nlon = temp.shape[1:]
    T_total = temp.shape[0]
    
    land_mask = get_land_mask(lats, lons)
    
    # 计算实际天数
    max_buffer = FIXED_LAG + cfg.WINDOW_SIZE
    valid_hours = T_total - max_buffer
    max_days = valid_hours // 24
    days_to_use = min(max_days, TARGET_DAYS)
    
    print(f"实际计算天数: {days_to_use} | 窗口: {cfg.WINDOW_SIZE} | 滞后: {FIXED_LAG}")

    params = {'nlat': nlat, 'nlon': nlon, 'lon_shift': lon_shift, 'T': T_total}
    offsets = make_offsets(cfg.MAX_LEVEL, cfg.INCLUDE_LOWER)
    n_cores = cfg.CPU_CORES if cfg.CPU_CORES else cpu_count()
    
    summary_data = []

    # 2. 循环 24 小时
    for hour in TARGET_HOURS:
        print(f"\n>>> 计算 {hour:02d}:00 ...")
        t0 = time.time()
        
        # A. 生成时间索引
        valid_indices = generate_hourly_indices(
            hour, T_total, days_to_use, FIXED_LAG, cfg.WINDOW_SIZE
        )
        
        if len(valid_indices) < 10:
            print("样本量不足，跳过。")
            continue

        # B. 任务坐标
        coords = [(i, j, FIXED_LEVEL, FIXED_LAG) for i in range(nlat) for j in range(nlon)]
        if cfg.SAMPLE_LIMIT:
            coords = coords[:cfg.SAMPLE_LIMIT]
        
        # C. 并行计算
        with Pool(n_cores, initializer=init_worker_hourly, initargs=(temp, offsets, params, valid_indices)) as pool:
            results = pool.map(sample_point_hourly_pairwise_task, coords)
        
        # D. 聚合
        mi_map = np.full((nlat, nlon), np.nan)
        for ii, jj, val in results:
            mi_map[ii, jj] = val
            
        # E. 统计
        global_mean = np.nanmean(mi_map)
        land_mean = np.nanmean(mi_map[land_mask])
        ocean_mean = np.nanmean(mi_map[~land_mask])
        
        # F. 保存
        npy_name = f"hourly_te_level{FIXED_LEVEL}_hour{hour:02d}.npy"
        np.save(os.path.join(HOURLY_MATRIX_PATH, npy_name), mi_map)
        
        summary_data.append({
            'hour': hour,
            'global_mean': global_mean,
            'land_mean': land_mean,
            'ocean_mean': ocean_mean
        })
        
        # 绘图
        plot_hourly_map(mi_map, lons, lats, hour, global_mean, HOURLY_FIGURE_PATH)
        
        print(f"    {hour:02d}:00 完成 | Global: {global_mean:.4f} | Time: {time.time()-t0:.1f}s")

    # 3. 保存 CSV
    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(HOURLY_MATRIX_PATH, "hourly_summary_land_ocean.csv"), index=False)
    print("\n=== 所有计算完成 ===")

def plot_hourly_map(mi_map, lons, lats, hour, ei_val, save_dir):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.feature.nightshade import Nightshade
    import datetime
    
    fig = plt.figure(figsize=(10, 6))
    proj = ccrs.Robinson(central_longitude=0)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    
    vmax = np.nanpercentile(mi_map, 99) if not np.all(np.isnan(mi_map)) else 1.0
    
    im = ax.pcolormesh(lons, lats, mi_map, 
                       transform=ccrs.PlateCarree(),
                       cmap='turbo', 
                       vmin=0, 
                       vmax=vmax)

    ax.coastlines(resolution='110m', linewidth=0.6, color='black')
    
    # 这里的日期仅用于模拟晨昏线位置，您可以改成您的实际数据月份
    current_time = datetime.datetime(2023, 1, 15, hour, 0, 0)
    ax.add_feature(Nightshade(current_time, alpha=0.2))
    
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.6, pad=0.05)
    cbar.set_label('Avg Pairwise TE (bits)', fontsize=10)

    plt.title(f"Global Information Flow at UTC {hour:02d}:00\n"
              f"Level={FIXED_LEVEL}, Lag={FIXED_LAG} | Global Avg={ei_val:.4f}", fontsize=12)
    
    save_name = f"hourly_te_hour{hour:02d}.png"
    plt.savefig(os.path.join(save_dir, save_name), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()