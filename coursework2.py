import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from plotly.subplots import make_subplots

# Step 1: 加载数据集
data = pd.read_csv('Results_21Mar2022.csv')
data = data.copy()  # 创建副本以避免原始数据警告

# Step 2: 更新饮食类型映射（不合并肉食类别）
diet_mapping = {
    'vegan': 'Vegans',
    'veggie': 'Vegetarians',
    'fish': 'Fish-eaters',
    'meat50': 'Meat-eaters (<50g/day)',
    'meat': 'Meat-eaters (50-99g/day)',
    'meat100': 'Meat-eaters (>=100g/day)'
}
data['diet_group'] = data['diet_group'].map(diet_mapping)

# Step 3: 选择相关列
metrics = ['mean_ghgs', 'mean_land', 'mean_watscar', 'mean_eut', 'mean_ghgs_ch4',
           'mean_ghgs_n2o', 'mean_bio', 'mean_watuse', 'mean_acid']
plot_data = data[['mc_run_id', 'diet_group'] + metrics].copy()  # 明确创建副本

# 定义新列名映射
metrics_rename = {
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

# 重命名列
plot_data = plot_data.rename(columns=metrics_rename)

# 更新 metrics 列表以使用新列名
metrics = list(metrics_rename.values())

# Step 4: 缩放数据
scaler = MinMaxScaler()
plot_data.loc[:, metrics] = scaler.fit_transform(plot_data[metrics])  # 使用 .loc 安全赋值

# Step 5: 准备小 multiples 数据
diet_types = list(diet_mapping.values())  # 获取所有饮食类型
fig = make_subplots(rows=2, cols=3, subplot_titles=diet_types)  # 调整为 2x3 布局

# Step 6: 为每个饮食类型创建平行坐标图
for i, diet in enumerate(diet_types, 1):
    diet_data = plot_data[plot_data['diet_group'] == diet].copy()  # 明确创建副本
    row, col = (i - 1) // 3 + 1, (i - 1) % 3 + 1  # 调整行列计算

    # 添加每条 mc_run_id 的线
    for run_id in diet_data['mc_run_id'].unique():
        run_data = diet_data[diet_data['mc_run_id'] == run_id]
        fig.add_trace(go.Scatter(x=metrics, y=run_data[metrics].values.flatten(),
                                 mode='lines', line=dict(color='grey', width=0.5),
                                 showlegend=False), row=row, col=col)

    # 计算并添加 IQR 包络
    grouped = diet_data.groupby('mc_run_id')[metrics].mean()
    median = grouped.median()
    q25 = grouped.quantile(0.25)
    q75 = grouped.quantile(0.75)

    fig.add_trace(go.Scatter(x=metrics, y=median, mode='lines',
                             line=dict(color='blue', width=2), name=diet), row=row, col=col)
    fig.add_trace(go.Scatter(x=metrics, y=q75, mode='lines', line=dict(color='rgba(0,0,0,0)'),
                             showlegend=False), row=row, col=col)
    fig.add_trace(go.Scatter(x=metrics, y=q25, mode='lines', fill='tonexty',
                             fillcolor='rgba(0,0,255,0.2)', line=dict(color='rgba(0,0,0,0)'),
                             showlegend=False), row=row, col=col)

# Step 7: 更新布局
fig.update_layout(height=800, width=1200, title_text='Environmental Impacts by Diet Type',
                  showlegend=True)
for i in range(1, 7):  # 调整为 6 个图表
    row, col = (i - 1) // 3 + 1, (i - 1) % 3 + 1
    fig.update_xaxes(tickangle=45, tickfont=dict(size=10), row=row, col=col)
    fig.update_yaxes(range=[0, 1], title_text='Scaled Value', row=row, col=col)

# Step 8: 保存图像
fig.write_image("small_multiples_parallel_coords_detailed.png")  # 需要 kaleido 包