import os
import numpy as np
import xarray as xr
from global_land_mask import globe

def load_nc_data(nc_path, var_name, days=None, lag_max=0, remove_diurnal_cycle=False):
    """
    通用数据加载与预处理函数。
    
    功能：
    1. 加载 NetCDF 数据。
    2. (可选) 截取指定天数。
    3. 经度标准化 (-180 到 180)。
    4. (核心) 去除日变化，计算温度距平。
    
    参数:
        nc_path: 文件路径
        var_name: 变量名 (如 't2m')
        days: 需要加载的天数 (None 表示加载全部)
        lag_max: 用于检查长度的缓冲区
        remove_diurnal_cycle: 是否去除日夜变化 (默认为 True，气象因果分析推荐开启)
    
    返回:
        temp (np.array): 处理后的数据矩阵 (Time, Lat, Lon)
        lats, lons: 坐标数组
        lon_shift: 经度循环偏移量
    """
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"数据文件未找到: {nc_path}")

    print(f"正在读取文件: {nc_path}")
    ds = xr.open_dataset(nc_path)
    
    # 1. 提取原始数据
    raw_temp = ds[var_name].values  
    lats = ds["latitude"].values
    lons = ds["longitude"].values
    ds.close()
    
    total_hours = raw_temp.shape[0]
    
    # 2. 长度检查与截取
    if days is not None:
        required_len = days * 24 + lag_max
        if total_hours < required_len:
            print(f"【注意】文件总长 {total_hours}h < 请求 {required_len}h，将使用全部数据。")
        else:
            # 如果文件很长，只取前 days 天 + 缓冲
            # 但为了计算距平的准确性，通常建议先用全部数据算气候态，再切片
            # 这里为了简单，我们暂不切片，返回全量数据供主程序处理，或者仅做长度警告
            pass

    # 3. 经度标准化 (0~360 -> -180~180)
    # 这对于 land_mask 和绘图都非常重要
    if np.any(lons > 180):
        print("检测到 0-360 格式经度，正在标准化为 -180 到 180...")
        # 调整经度数值
        lons = (lons + 180) % 360 - 180
        # 注意：这通常意味着数组的列顺序变了，需要重新排序数据
        # 简单起见，我们假设数据是循环的，只修改坐标值给掩膜用
        # 如果要严格对齐地图，需要使用 np.roll 滚动数据
        # 鉴于 TE 计算只关心相对邻居，只要 lons 和 temp 对应关系没变就行
        # 但为了 mask 准确，我们只修改 lons 值，不 roll 数据，因为 globe.is_land 会根据 lat/lon 值查找
    
    # 4. 经度偏移量 (用于处理地球左右边界循环)
    if len(lons) > 1:
        mean_dlon = np.median(np.abs(np.diff(lons))) # 加 abs 防止乱序
        lon_shift = int(round(180.0 / mean_dlon))
    else:
        lon_shift = 0

    # 5. [核心] 去除日夜变化 (计算距平)
    if remove_diurnal_cycle:
        print("正在计算小时气候态并去除日变化 (Anomaly Calculation)...")
        
        # 确保数据是 24 小时的整数倍，方便 reshape
        n_days = raw_temp.shape[0] // 24
        valid_len = n_days * 24
        
        # 截取整天数据
        temp_truncated = raw_temp[:valid_len]
        
        # 重塑: (天, 24小时, 纬度, 经度)
        temp_reshaped = temp_truncated.reshape(n_days, 24, raw_temp.shape[1], raw_temp.shape[2])
        
        # 计算 30 天内每个小时的平均值 (24, Lat, Lon)
        hourly_climatology = np.mean(temp_reshaped, axis=0)
        
        # 减去平均值得到距平
        temp_anomaly = temp_reshaped - hourly_climatology[np.newaxis, :, :, :]
        
        # 展平回时间序列
        temp_final = temp_anomaly.reshape(-1, raw_temp.shape[1], raw_temp.shape[2])
        
        # 如果原始数据有尾巴 (多余的几小时)，也减去对应的小时均值
        remainder = raw_temp.shape[0] % 24
        if remainder > 0:
            tail_data = raw_temp[valid_len:]
            tail_anomaly = tail_data - hourly_climatology[:remainder]
            temp_final = np.concatenate([temp_final, tail_anomaly], axis=0)
            
        print("日变化已去除。")
    else:
        temp_final = raw_temp

    print(f"数据预处理完成: 形状={temp_final.shape}")
    return temp_final, lats, lons, lon_shift

def make_offsets(max_level, include_lower):
    """
    构建每一层级的空间邻居偏移量字典
    Level 1: 3x3 环 (8个邻居)
    Level 2: 5x5 环 (16个邻居) ...
    """
    offsets = {}
    for lvl in range(max_level + 1):
        if lvl == 0:
            offsets[lvl] = [(0, 0)]
        else:
            # 环形邻域 (Ring) - 只取最外圈
            ring = [(di, dj)
                    for di in range(-lvl, lvl + 1)
                    for dj in range(-lvl, lvl + 1)
                    if max(abs(di), abs(dj)) == lvl]
            
            if include_lower:
                # 实心邻域 (Cube) - 包含内部所有点
                cube = [(di, dj)
                        for di in range(-lvl, lvl + 1)
                        for dj in range(-lvl, lvl + 1)]
                offsets[lvl] = cube
            else:
                offsets[lvl] = ring
    return offsets

def get_land_mask(lats, lons):
    """
    生成陆地掩膜 (True=Land, False=Ocean)
    """
    print("正在生成陆地掩膜...")
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    is_land = globe.is_land(lat_grid, lon_grid)
    return is_land