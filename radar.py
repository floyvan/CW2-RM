import pandas as pd
import plotly.graph_objects as go

# 加载数据集
df = pd.read_csv("Results_21Mar2022.csv")

# 按饮食组、性别和年龄组聚合数据，计算蒙特卡洛运行的平均值
grouped_df = df.groupby(['diet_group', 'sex', 'age_group']).agg({
    'mean_ghgs': 'mean',
    'mean_land': 'mean',
    'mean_watscar': 'mean',
    'mean_eut': 'mean',
    'mean_bio': 'mean',
    'n_participants': 'first'  # 每个组的参与者数量相同
}).reset_index()

# 计算每个饮食组的加权平均值
metrics = ['mean_ghgs', 'mean_land', 'mean_watscar', 'mean_eut', 'mean_bio']

# 步骤 1：计算每行的加权指标
for metric in metrics:
    grouped_df[f'weighted_{metric}'] = grouped_df[metric] * grouped_df['n_participants']

# 步骤 2：按 'diet_group' 分组并计算加权指标的总和
weighted_columns = [f'weighted_{metric}' for metric in metrics]
weighted_sums = grouped_df.groupby('diet_group')[weighted_columns].sum()

# 步骤 3：计算每个饮食组的总参与者数量
total_participants = grouped_df.groupby('diet_group')['n_participants'].sum()

# 步骤 4：计算加权平均值
weighted_averages = weighted_sums.div(total_participants, axis=0)
weighted_averages = weighted_averages.reset_index()

# 将加权列名改回原始指标名以保持一致性
weighted_averages.columns = ['diet_group'] + metrics

# 应用饮食组映射以获得可读标签
diet_mapping = {
    'vegan': 'Vegans',
    'veggie': 'Vegetarians',
    'fish': 'Fish-eaters',
    'meat50': 'Meat-eaters (<50g/day)',
    'meat': 'Meat-eaters (50-99g/day)',
    'meat100': 'Meat-eaters (100+g/day)'
}
weighted_averages['diet_group'] = weighted_averages['diet_group'].map(diet_mapping)

# 将指标归一化到 [0,1]
mins = weighted_averages[metrics].min()
maxs = weighted_averages[metrics].max()
normalized_df = (weighted_averages[metrics] - mins) / (maxs - mins)
normalized_df['diet_group'] = weighted_averages['diet_group']

# 定义绘图顺序（从环境影响小的组到大的组）
plot_order = [
    'High-Meat-eaters (100+g/day)',
    'Medium-Meat-eaters (50-99g/day)',
    'Low-Meat-eaters (<50g/day)',
    'Fish-eaters',
    'Vegetarians',
    'Vegans'
]

# 根据 plot_order 重新排序 normalized_df
normalized_df['diet_group'] = pd.Categorical(normalized_df['diet_group'], categories=plot_order, ordered=True)
normalized_df = normalized_df.sort_values('diet_group')

# 定义指标的可读标签
labels = {
    'mean_ghgs': 'GHG Emissions',
    'mean_land': 'Land Use',
    'mean_watscar': 'Water Scarcity',
    'mean_eut': 'Eutrophication',
    'mean_bio': 'Biodiversity Impact'
}

# 创建雷达图
fig = go.Figure()

for index, row in normalized_df.iterrows():
    fig.add_trace(go.Scatterpolar(
        r=row[metrics].values,
        theta=list(labels.values()),
        fill='toself',
        name=row['diet_group'],
        hovertemplate=
        '<b>%{theta}</b><br>' +
        'Value: %{r:.3f}<br>' +
        'Diet Group: %{customdata}<br>' +
        '<extra></extra>',
        customdata=[row['diet_group']] * len(metrics)
    ))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
    showlegend=True,
    title='Environmental Impact of Different Diet Groups',
    hovermode='closest'  # 确保悬停时目标是最接近的点
)

# 将图表保存为图片
fig.write_image("radar_chart.png")

# 可选：显示交互式图表以进行测试
fig.show()

# Optionally, display the interactive plot (for development purposes)
fig.write_html("interactive_radar_chart.html")