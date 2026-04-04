# plot_utils.py
import os
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_te_map(mi_map, lons, lats, level, min_lag, max_lag, ei_val, figure_path):
    """
    绘制并保存全球 TE 热力图 (Robinson 投影)
    """
    fig = plt.figure(figsize=(10, 6))
    proj = ccrs.Robinson(central_longitude=0)
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    # 自动色标范围
    vmax = np.nanmax(mi_map)
    if np.isnan(vmax): vmax = 1.0
    
    im = ax.pcolormesh(lons, lats, mi_map, 
                       transform=ccrs.PlateCarree(),
                       cmap='turbo', 
                       vmin=0, 
                       vmax=vmax)

    ax.coastlines(resolution='110m', linewidth=0.8, color='black')
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, edgecolor='gray')
    
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.6, pad=0.05)
    cbar.set_label('Information Flow (bits)', fontsize=10)

    # 标题显示 Shift 和具体窗口范围
    # shift = min_lag
    shift = min_lag
    window_len = max_lag - min_lag + 1
    
    title_str = (f"Level {level} TE Map [Shift {shift} | Window {window_len}]\n"
                 f"Range: t-{max_lag} to t-{min_lag} | Global Avg = {ei_val:.4f} bits")
    plt.title(title_str, fontsize=12)
    
    # 文件名使用 shift 命名，保持简洁
    file_name = f"te_heatmap_level_{level}_shift_{shift}.png"
    save_path = os.path.join(figure_path, file_name)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()