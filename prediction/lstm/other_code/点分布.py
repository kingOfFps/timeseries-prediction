from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.datasets import make_blobs
import tkinter as tk
from tkinter import Text
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pysal.lib import weights
from pysal.explore import esda
import geopandas as gpd
from shapely.geometry import Point
from scipy.interpolate import griddata
import itertools
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.cm import get_cmap
from matplotlib.collections import PolyCollection
import matplotlib.patches as patches
import numpy as np
import matplotlib.gridspec as gridspec

def generate_clusters(num_clusters, min_distance):
    centers = np.random.uniform(0, 10, (1, 2))
    while len(centers) < num_clusters:
        new_center = np.random.uniform(0, 10, (1, 2))
        if np.all(np.min(cdist(new_center, centers), axis=1) > min_distance):
            centers = np.vstack([centers, new_center])
    return centers

def plot_cube(ax, lower_limits, upper_limits, base_height):
    lower_limits[2] += base_height
    upper_limits[2] += base_height
    vertices = np.array(list(itertools.product(*zip(lower_limits, upper_limits))))
    faces = [[vertices[j] for j in face] for face in
             [[0, 1, 5, 4], [4, 5, 7, 6], [6, 7, 3, 2], [2, 3, 1, 0], [1, 3, 7, 5], [0, 4, 6, 2]]]
    ax.add_collection3d(Poly3DCollection(faces, linewidths=1, edgecolors='r', alpha=0.2))


def calculate_cavity_volume(grid_x, grid_y, grid_z):
    min_height = np.nanmin(grid_z)
    volume = 0.0
    for i in range(grid_x.shape[0] - 1):
        for j in range(grid_x.shape[1] - 1):
            if not np.isnan(grid_z[i, j]) and not np.isnan(grid_z[i + 1, j]) and not np.isnan(
                    grid_z[i, j + 1]) and not np.isnan(grid_z[i + 1, j + 1]):
                height = (grid_z[i, j] + grid_z[i + 1, j] + grid_z[i, j + 1] + grid_z[i + 1, j + 1]) / 4 - min_height
                base_area = (grid_x[i + 1, j] - grid_x[i, j]) * (grid_y[i + 1, j + 1] - grid_y[i + 1, j])
                volume += base_area * height
    return volume


def update_volume_info(cavity_volume, water_above_volume, water_below_volume, water_volume):
    volume_info.config(state=tk.NORMAL)
    volume_info.delete(1.0, tk.END)
    volume_info.insert(tk.END, f"Total Cavity Volume: {cavity_volume}\n")
    volume_info.insert(tk.END, f"Water Above Volume: {water_above_volume}\n")
    volume_info.insert(tk.END, f"Water Below Volume: {water_below_volume}\n")
    volume_info.insert(tk.END, f"Water Volume: {water_volume}\n")
    volume_info.config(state=tk.DISABLED)


def calculate_water_above_volume_data(grid_x, grid_y, grid_z):
    min_z_value = np.nanmin(grid_z)
    max_z_value = np.nanmax(grid_z)
    water_levels = np.linspace(0, 1, 10)
    water_above_volume_data = []

    for water_level in water_levels:
        water_surface = (water_level * (max_z_value - min_z_value)) + min_z_value
        water_above_volume = calculate_cavity_volume(grid_x, grid_y, np.where(grid_z > water_surface, grid_z, np.nan))
        water_above_volume_data.append((water_level, water_above_volume))

    return water_above_volume_data


def plot_water_above_volume(ax6, water_above_volume_data):
    ax6.clear()
    ax6.set_title("Water Above Volume with Water Level")
    ax6.set_xlabel("Water Level")
    ax6.set_ylabel("Water Above Volume")

    # 将水面上升高度和Water Above Volume的数据分开
    water_above_volume_values = [data[1] for data in water_above_volume_data]
    water_level_values = [data[0] for data in water_above_volume_data]

    # 使用线性拟合曲线
    coefficients = np.polyfit(water_level_values, water_above_volume_values, 1)
    polynomial = np.poly1d(coefficients)
    fitted_values = polynomial(water_level_values)

    # 在ax6中绘制Water Above Volume随着水面上升高度的变化趋势图
    ax6.plot(water_level_values, water_above_volume_values, 'bo', label="Water Above Volume")
    ax6.plot(water_level_values, fitted_values, 'r-', label="Fitted Curve")
    ax6.set_xlabel("Water Level")
    ax6.set_ylabel("Water Above Volume")
    ax6.set_title("Water Above Volume with Water Level")
    ax6.legend()


def plot_k_function_and_distribution(fig, ax6, num_points, num_clusters, cluster_std, min_distance, water_level):
    global grid_x, grid_y, grid_z, water_above_volume_data
    # 获取噪声点的占比
    noise_ratio = slider_noise_ratio.get()

    # 计算噪声点和聚类点的数量
    num_noise_points = int(noise_ratio * num_points)
    num_cluster_points = num_points - num_noise_points

    # 生成聚类中心
    centers = generate_clusters(num_clusters, min_distance)

    # 生成噪声点和聚类点
    noise_points = np.random.uniform(0, 10, (num_noise_points, 2))
    cluster_points, _ = make_blobs(n_samples=num_cluster_points, centers=centers, cluster_std=cluster_std)

    # 合并噪声点和聚类点
    data = np.vstack([noise_points, cluster_points])

    x, y = data[:, 0], data[:, 1]
    x_centers, y_centers = centers[:, 0], centers[:, 1]

    radius = 1.0
    density = np.array([np.sum(np.sqrt(np.sum((data - point) ** 2, axis=1)) < radius) for point in data])

    distances = squareform(pdist(np.column_stack((x, y))))

    distances_to_centers = cdist(np.column_stack((x, y)), centers)
    labels = np.argmin(distances_to_centers, axis=1)

    gdf = gpd.GeoDataFrame({"X": x, "Y": y, "label": labels, "geometry": [Point(xy) for xy in zip(x, y)]})
    gdf["label"] = gdf["label"].astype(float)
    w = weights.Kernel.from_dataframe(gdf, fixed=False, k=15)
    morans_I = esda.moran.Moran(gdf["label"], w)
    morans_lag = weights.lag_spatial(w, gdf["label"])

    grid_x, grid_y = np.mgrid[0:10:100j, 0:10:100j]
    grid_z = griddata((x, y), density, (grid_x, grid_y), method='cubic')

    cavity_volume = calculate_cavity_volume(grid_x, grid_y, grid_z)

    min_z_value = np.nanmin(grid_z)
    max_z_value = np.nanmax(grid_z)
    water_surface = min_z_value + water_level * (max_z_value - min_z_value)

    # 计算Water Above Volume随着水面上升高度的变化数据
    water_above_volume_data = calculate_water_above_volume_data(grid_x, grid_y, grid_z)

    water_above_volume = calculate_cavity_volume(grid_x, grid_y, np.where(grid_z > water_surface, grid_z, np.nan))
    water_below_volume = cavity_volume - water_above_volume

    water_volume = (100 * water_level * (max_z_value - min_z_value)) - water_below_volume

    update_volume_info(cavity_volume, water_above_volume, water_below_volume, water_volume)

    print("Total Cavity Volume:", cavity_volume)
    print("Water Above Volume:", water_above_volume)
    print("Water Below Volume:", water_below_volume)
    print("Water Volume:", water_volume)

    # fig.clear()  # 这里不需要清除整个figure

    ax1 = fig.add_subplot(321)
    ax1.scatter(x, y, label="Points")
    ax1.scatter(x_centers, y_centers, color='red', label="Centers")
    ax1.set_xlim([0, 10])
    ax1.set_ylim([0, 10])
    ax1.set_aspect('equal', 'box')
    ax1.set_title("Point Distribution")
    ax1.legend()

    ax2 = fig.add_subplot(323, projection='3d')
    ax2.plot_surface(grid_x, grid_y, grid_z, cmap='viridis')
    ax2.set_title("3D IDW Interpolation")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    min_z_value = np.nanmin(grid_z)
    max_z_value = np.nanmax(grid_z)

    plot_cube(ax2, [0, 0, 0], [10, 10, max_z_value - min_z_value], min_z_value)

    water_surface = (water_level * (max_z_value - min_z_value)) + min_z_value

    ax2.plot_surface(grid_x, grid_y, np.full_like(grid_z, water_surface), color='blue', alpha=0.5)

    density_projection = np.full((grid_x.shape[0], grid_x.shape[1]), min_z_value)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            if not np.isnan(grid_z[i, j]):
                density_projection[i, j] = grid_z[i, j]

    ax2.plot_surface(grid_x, grid_y, density_projection, alpha=0.4, cmap='Blues', zorder=2)

    ax2.contourf(grid_x, grid_y, density_projection, zdir='z', offset=min_z_value, cmap='viridis', alpha=0.7)

    def ripley_k(distances, r, area):
        N = distances.shape[0]
        densities = np.sum(distances < r, axis=1) - 1
        return np.sum(densities) / (N * (N - 1) / 2) * area

    r_values = np.linspace(0, np.sqrt(2) * 10, 100)
    area = 10 * 10
    k_values = [ripley_k(distances, r, area) for r in r_values]

    np.random.seed(0)
    num_simulations = 1000
    k_values_simulations = np.empty((num_simulations, len(r_values)))
    for i in range(num_simulations):
        x_sim, y_sim = make_blobs(n_samples=num_points, centers=centers, n_features=2, cluster_std=cluster_std)
        distances_sim = squareform(pdist(np.column_stack((x_sim, y_sim))))
        k_values_simulations[i, :] = [ripley_k(distances_sim, r, area) for r in r_values]

    lower_bound = np.percentile(k_values_simulations, 2.5, axis=0)
    upper_bound = np.percentile(k_values_simulations, 97.5, axis=0)

    ax3 = fig.add_subplot(322)
    ax3.plot(r_values, k_values, label="Ripley's K-function")
    ax3.fill_between(r_values, lower_bound, upper_bound, color='gray', alpha=0.5, label="95% Confidence Interval")
    ax3.set_xlabel("r")
    ax3.set_ylabel("K(r)")
    ax3.legend()
    ax3.set_title("Ripley's K-function and 95% Confidence Interval")

    # Create ax5 based on the structure provided for ax4
    ax4 = fig.add_subplot(324)  # Assuming fig is the main figure you're working with
    # Generate the histogram for the x coordinates
    ax4.hist(x, bins=10, color='lightblue', edgecolor='black')
    # Set titles and labels for ax5
    ax4.set_title("Histogram of x-coordinate Distribution")
    ax4.set_xlabel("x")
    ax4.set_ylabel("Frequency")

    # Adding the top view of the water above part
    ax5 = fig.add_subplot(325)
    above_water = np.where(grid_z > water_surface, grid_z, np.nan)
    ax5.contourf(grid_x, grid_y, above_water, cmap='viridis')
    ax5.set_title("Top View of Water Above Part")
    ax5.set_xlabel("x")
    ax5.set_ylabel("y")

    # 初始化全局变量
    global root3, canvas2, fig2, ax7, ax8, ax9
    root3 = None
    canvas2 = None

    fig2 = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[2, 1])
    ax7 = fig2.add_subplot(gs[0, :2], projection='3d')
    ax8 = fig2.add_subplot(gs[1, 0], aspect='equal')
    ax9 = fig2.add_subplot(gs[1, 1], aspect='equal')

    def update_fig2(water_level):
        global root3, canvas2, fig2, ax7, ax8, ax9

        # 清除子图内容
        ax7.clear()
        ax8.clear()
        ax9.clear()

        # 定义中间剖面图的点数
        num_points = 100

        # 找到x=5所对应的索引
        middle_index_x = np.abs(grid_x[0, :] - 5).argmin()

        # 创建y的插值网格
        grid_y_middle = np.linspace(np.min(grid_y), np.max(grid_y), num_points)

        # 获取滑块的值
        slice_x = slider_slice_x.get()
        slice_y = slider_slice_y.get()

        # 获取x=slice_x这个平面上所有点的值
        middle_plane_z_x = griddata((grid_x.flatten(), grid_y.flatten()), grid_z.flatten(), (slice_x, grid_y_middle))

        # 找到y=6所对应的索引
        middle_index_y = np.abs(grid_y[:, 0] - 6).argmin()

        # 创建x的插值网格
        grid_x_middle = np.linspace(np.min(grid_x), np.max(grid_x), num_points)

        # 获取y=slice_y这个平面上所有点的值
        middle_plane_z_y = griddata((grid_x.flatten(), grid_y.flatten()), grid_z.flatten(), (grid_x_middle, slice_y))

        # 创建一个颜色映射对象
        cmap = get_cmap('viridis')

        # 获取z值的范围
        min_z_value = np.nanmin(grid_z)
        max_z_value = np.nanmax(grid_z)

        # 创建一个用于将数据值映射到颜色映射范围的对象
        norm = Normalize(vmin=min_z_value, vmax=max_z_value)

        # 创建一个新的图形窗口
        ax7.plot_surface(grid_x, grid_y, grid_z, cmap='viridis')
        ax7.set_title("3D IDW Interpolation")
        ax7.set_xlabel("x")
        ax7.set_ylabel("y")
        ax7.set_zlabel("z")

        plot_cube(ax7, [0, 0, 0], [10, 10, max_z_value - min_z_value], min_z_value)
        water_surface = (water_level * (max_z_value - min_z_value)) + min_z_value
        ax7.plot_surface(grid_x, grid_y, np.full_like(grid_z, water_surface), color='blue', alpha=0.5)

        density_projection = np.full((grid_x.shape[0], grid_x.shape[1]), min_z_value)
        for i in range(grid_x.shape[0]):
            for j in range(grid_x.shape[1]):
                if not np.isnan(grid_z[i, j]):
                    density_projection[i, j] = grid_z[i, j]

        ax7.plot_surface(grid_x, grid_y, density_projection, alpha=0.4, cmap='Blues', zorder=2)
        ax7.contourf(grid_x, grid_y, density_projection, zdir='z', offset=min_z_value, cmap='viridis', alpha=0.7)

        # 在新的图形窗口中创建第一个子图来展示x=5的剖面图
        ax8.plot(grid_y_middle, middle_plane_z_x, color='red')
        for i in range(len(grid_y_middle) - 1):
            ax8.fill_between(grid_y_middle[i:i + 2], middle_plane_z_x[i:i + 2], min_z_value,
                             color=cmap(norm(middle_plane_z_x[i])))
        ax8.axhline(water_level, color='blue', linestyle='--')
        ax8.set_title(f"Cross Section at x = {slice_x}")
        ax8.set_xlabel("y")
        ax8.set_ylabel("z")
        ax8.set_ylim([min_z_value, max_z_value])

        # 在新的图形窗口中创建第二个子图来展示y=6的剖面图
        ax9.plot(grid_x_middle, middle_plane_z_y, color='red')
        for i in range(len(grid_x_middle) - 1):
            ax9.fill_between(grid_x_middle[i:i + 2], middle_plane_z_y[i:i + 2], min_z_value,
                             color=cmap(norm(middle_plane_z_y[i])))
        ax9.axhline(water_level, color='blue', linestyle='--')
        ax9.set_title(f"Cross Section at y = {slice_y}")
        ax9.set_xlabel("x")
        ax9.set_ylabel("z")
        ax9.set_ylim([min_z_value, max_z_value])

        fig2.tight_layout()

        if root3 is None:
            root3 = tk.Toplevel(root2)
            canvas2 = FigureCanvasTkAgg(fig2, master=root3)
            canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            canvas2.draw()
        else:
            canvas2.draw()
    update_fig2(water_level)


    # 在此处调用新的绘制函数绘制Water Above Volume随着水面上升高度的变化趋势
    plot_water_above_volume(ax6, water_above_volume_data)
    canvas.draw()  # 在绘制完成后进行画布的更新



def animate_water():
    water_levels = np.linspace(0, 1, 10)  # 将水面高度从0到1分为10个时间步
    ani = FuncAnimation(fig, update_water_surface, fargs=(ax6,), frames=water_levels, interval=100, repeat=False)
    canvas.draw()


def update_water_surface(water_level, ax6):
    global grid_x, grid_y, grid_z
    global water_above_volume_data
    global canvas3
    # 在这里更新水面高度，并重新绘制3D插值图
    plot_k_function_and_distribution(fig, ax6, slider_points.get(), slider_clusters.get(), slider_std.get(),
                                     slider_min_distance.get(), water_level)
    min_z_value = np.nanmin(grid_z)
    max_z_value = np.nanmax(grid_z)
    water_surface = (water_level * (max_z_value - min_z_value)) + min_z_value
    water_above_volume = calculate_cavity_volume(grid_x, grid_y, np.where(grid_z > water_surface, grid_z, np.nan))
    # Append this data to water_above_volume_data
    water_above_volume_data.append((water_level, water_above_volume))
    # Create or update ax6 based on the new data
    ax6 = fig4.add_subplot(111)
    water_above_volume_values = [data[1] for data in water_above_volume_data]
    water_level_values = [data[0] for data in water_above_volume_data]

    # Linear regression
    coefficients = np.polyfit(water_level_values, water_above_volume_values, 1)
    polynomial = np.poly1d(coefficients)
    fitted_values = polynomial(water_level_values)

    ax6.clear()
    ax6.plot(water_level_values, water_above_volume_values, 'bo', label="Water Above Volume")
    ax6.plot(water_level_values, fitted_values, 'r-', label="Fitted Curve")
    ax6.set_xlabel("Water Level")
    ax6.set_ylabel("Water Above Volume")
    ax6.set_title("Water Above Volume with Water Level")
    ax6.legend()

    canvas4.draw()

root2 = tk.Tk()
root2.title("绘图")
fig = plt.figure(figsize=(10, 10))
fig.subplots_adjust(hspace=0.4)  # 增加子图之间的上下间距
canvas = FigureCanvasTkAgg(fig, master=root2)
canvas.get_tk_widget().pack()

root1 = tk.Tk()
root1.title("滑块")

root4 = tk.Tk()
root4.title("淹水模型")
fig4 = plt.figure(figsize=(10, 10))
canvas4 = FigureCanvasTkAgg(fig4, master=root4)
canvas4.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

# 创建点的数量滑块
slider_points = tk.Scale(root1, from_=10, to=500, resolution=10, orient="horizontal", label="Number of Points")
slider_points.pack()
slider_points.set(100)

# 创建噪声比例滑块
slider_noise_ratio = tk.Scale(root1, from_=0, to=1, resolution=0.01, orient="horizontal", label="Noise Ratio")
slider_noise_ratio.pack()
slider_noise_ratio.set(0.5)

# 创建聚类数量滑块
slider_clusters = tk.Scale(root1, from_=1, to=10, resolution=1, orient="horizontal", label="Number of Clusters")
slider_clusters.pack()
slider_clusters.set(3)

slider_std = tk.Scale(root1, from_=0.1, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, label="Cluster Standard Deviation")
slider_min_distance = tk.Scale(root1, from_=0, to=10, resolution=0.1, orient=tk.HORIZONTAL, label="Minimum Distance")

slider_water_level = tk.Scale(root1, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="Water Level",
                              sliderlength=30)
slider_water_level.set(0)

slider_water_level_button = tk.Button(root1, text="Animate Water", command=animate_water)
slider_water_level_button.pack()

button = tk.Button(root1, text="Update",
                   command=lambda: plot_k_function_and_distribution(fig, ax6, slider_points.get(),
                                                                    slider_clusters.get(), slider_std.get(),
                                                                    slider_min_distance.get(),
                                                                    slider_water_level.get()))

# 然后定义滑块
slider_slice_x = tk.Scale(root1, from_=0.0, to=10.0, orient=tk.HORIZONTAL, resolution=0.01, label="X Slice")
slider_slice_x.set(5)

slider_slice_y = tk.Scale(root1, from_=0.0, to=10.0, orient=tk.HORIZONTAL, resolution=0.01, label="Y Slice")
slider_slice_y.set(5)


slider_points.pack()
slider_clusters.pack()
slider_std.pack()
slider_min_distance.pack()
slider_water_level.pack()
slider_slice_x.pack()
slider_slice_y.pack()
button.pack()

ax6 = fig.add_subplot(326)

volume_info = Text(root1, wrap=tk.WORD, width=30, height=10)
volume_info.pack()

root1.mainloop()
