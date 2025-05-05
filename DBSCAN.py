import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from kneed import KneeLocator

# Define diet group mapping for readable labels
diet_mapping = {
    'vegan': 'Vegans',
    'veggie': 'Vegetarians',
    'fish': 'Fish-eaters',
    'meat50': 'Meat-eaters (<50g/day)',
    'meat': 'Meat-eaters (50-99g/day)',
    'meat100': 'Meat-eaters (100+g/day)'
}

# 加载数据集
data = pd.read_csv("Results_21Mar2022.csv")

# 获取所有唯一饮食组
diet_groups = data['diet_group'].unique()

# 初始化异常值数据的列表，用于汇总
all_outliers = []

# 设置子图布局（例如 2 行 2 列，调整根据饮食组数量）
n_groups = len(diet_groups)
n_cols = 3
n_rows = 2  # 向上取整以适应行数

# 初始化小多重图（仅创建一次）
plt.figure(figsize=(20, 5 * n_rows))

# 循环分析每个饮食组
for idx, diet_group in enumerate(diet_groups):
    # 按饮食组过滤数据
    filtered_data = data[data['diet_group'] == diet_group]

    if filtered_data.empty:
        print(f"No data found for diet group: {diet_group}")
        continue

    readable_diet = diet_mapping.get(diet_group, diet_group)
    print(f"Analyzing data for diet group: {readable_diet}")
    print(f"Number of records: {len(filtered_data)}")

    # 选择相关特征
    features = ['mean_ghgs', 'mean_land', 'mean_watscar', 'mean_eut']
    X = filtered_data[features]

    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 计算 k-距离（k = min_samples - 1）
    min_samples = 5
    k = min_samples - 1
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
    distances, indices = nbrs.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, k-1])
    kneedle = KneeLocator(range(len(k_distances)), k_distances, curve='convex', direction='increasing')
    suggested_eps = k_distances[kneedle.knee] if kneedle.knee is not None else 0.5

    # 绘制 k-距离图
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(k_distances)), k_distances, marker='o')
    plt.axvline(x=kneedle.knee, color='r', linestyle='--', label=f'Suggested eps = {suggested_eps:.2f}')
    plt.title(f'k-Distance Plot for {readable_diet} ')
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'k-Distance (k={k})')
    plt.legend()
    plt.savefig(f'k_distance_plot_{diet_group}.png')
    plt.close()

    print(f"Suggested eps for {readable_diet}: {suggested_eps:.2f}")

    # 应用DBSCAN使用建议的 eps
    dbscan = DBSCAN(eps=suggested_eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_scaled)

    # 使用PCA降维到2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 绘制子图
    plt.subplot(n_rows, n_cols, idx + 1)
    plt.scatter(X_pca[clusters != -1, 0], X_pca[clusters != -1, 1],
                c=clusters[clusters != -1], cmap='viridis', label='In-cluster Points')
    plt.scatter(X_pca[clusters == -1, 0], X_pca[clusters == -1, 1],
                c='red', label='Outliers', marker='x')
    plt.title(f'DBSCAN Clustering for {readable_diet} Diet (eps={suggested_eps:.2f})')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()

    # 检查异常值的数据
    outliers = filtered_data[clusters == -1]
    print(f"Number of outliers found for {readable_diet}: {len(outliers)}")
    if len(outliers) > 0:
        print(f"Outlier details for {readable_diet}:")
        print(outliers[['diet_group', 'sex', 'age_group', 'mean_ghgs', 'mean_land', 'mean_watscar', 'mean_eut']])
        # 添加异常值到汇总列表
        all_outliers.append(outliers)

# 调整布局并保存小多重图
plt.tight_layout()
plt.savefig('dbscan_outliers_small_multiples.png')
plt.close()

# 汇总所有异常值并保存到CSV
if all_outliers:
    all_outliers_df = pd.concat(all_outliers, ignore_index=True)
    all_outliers_df.to_csv('all_outliers_multi_diet.csv', index=False)
    print("All outliers saved to 'all_outliers_multi_diet.csv'")
else:
    print("No outliers detected across all diet groups.")