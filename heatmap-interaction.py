import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

# 加载数据集
data = pd.read_csv('Results_21Mar2022.csv')
data = data.copy()  # 创建副本以避免原始数据警告

# 更新饮食类型映射（不合并肉食类别）
diet_mapping = {
    'vegan': 'Vegans',
    'veggie': 'Vegetarians',
    'fish': 'Fish-eaters',
    'meat50': 'Meat-eaters (<50g/day)',
    'meat': 'Meat-eaters (50-99g/day)',
    'meat100': 'Meat-eaters (100+g/day)'
}
data['diet_group'] = data['diet_group'].map(diet_mapping)

# 定义字典并选择列
metrics = {
    'mean_ghgs': 'Mean-GHG-emissions',
    'mean_land': 'Mean-Land-use',
    'mean_watscar': 'Mean-Water-scarcity',
    'mean_eut': 'Mean-Eutrophication',
    'mean_ghgs_ch4': 'Mean-GHG-CH4',
    'mean_ghgs_n2o': 'Mean-GHG-N2O',
    'mean_bio': 'Mean-Biodiversity',
    'mean_watuse': 'Mean-Water-usage',
    'mean_acid': 'Mean-Acidification'
}

# 从 metrics 字典中提取旧列名（键）
original_metrics = list(metrics.keys())

# 选择列并创建副本
plot_data = data[['mc_run_id', 'diet_group'] + original_metrics].copy()

# 重命名列为新名称（字典的值）
plot_data = plot_data.rename(columns=metrics)

# 获取新列名列表
new_metrics = list(metrics.values())

# 缩放数据
scaler = MinMaxScaler()
plot_data.loc[:, new_metrics] = scaler.fit_transform(plot_data[new_metrics])

# 计算每个饮食类型的指标中位数
# 按饮食类型分组，计算每个指标的中位数
heatmap_data = plot_data.groupby('diet_group')[new_metrics].median().reset_index()

# 创建交互式热力图
# 将数据转换为长格式（便于 Plotly 绘图）
heatmap_melted = heatmap_data.melt(id_vars=['diet_group'], value_vars=new_metrics,
                                   var_name='Metric', value_name='Scaled Value')

# 使用 Plotly Express 创建热力图
fig = px.imshow(
    heatmap_data.set_index('diet_group')[new_metrics],  # 设置 diet_group 为索引，new_metrics 为列
    labels=dict(x="Environmental Metric", y="Diet Type", color="Scaled Value"),
    x=new_metrics,
    y=heatmap_data['diet_group'],
    color_continuous_scale='Viridis',  # 颜色范围，从低（浅）到高（深）
    title="Interactive Heatmap of Environmental Impacts by Diet Type"
)

# 更新布局，增强交互性
fig.update_layout(
    height=600,
    width=1000,
    title_text='Interactive Heatmap of Environmental Impacts by Diet Type',
    xaxis_title="Environmental Metric",
    yaxis_title="Diet Type",
    coloraxis_colorbar_title="Scaled Value",
    xaxis={'tickangle': 45, 'tickfont': dict(size=10)},  # 旋转 X 轴标签以提高可读性
)

# 优化悬停信息
fig.update_traces(
    hovertemplate='Diet: %{y}<br>Metric: %{x}<br>Scaled Value: %{z:.2f}<extra></extra>'
)

# 保存为交互式 HTML 文件
fig.write_html("interactive_heatmap.html")

# 可选：保存为静态图像（如果需要）
fig.write_image("interactive_heatmap.png")