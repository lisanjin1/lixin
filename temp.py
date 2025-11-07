import plotly.graph_objects as go
import numpy as np

# 示例数据
residual = y_true[:, 0] - y_pred[:, 0]   # 第一个输出的残差
x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]

fig = go.Figure(data=[
    go.Scatter3d(
        x=x1, y=x2, z=residual,
        mode='markers',
        marker=dict(
            size=5,
            color=residual,      # 颜色映射残差值
            colorscale='RdBu',
            colorbar=dict(title='Residual'),
            opacity=0.7
        ),
        text=[f"y_true={yt:.3f}<br>y_pred={yp:.3f}" for yt, yp in zip(y_true[:,0], y_pred[:,0])],
        hovertemplate="x1=%{x:.2f}<br>x2=%{y:.2f}<br>res=%{z:.3f}<br>%{text}"
    )
])

fig.update_layout(
    title="3D Residual Cloud (Output 1)",
    scene=dict(
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        zaxis_title="Residual"
    ),
    template="plotly_dark",
    height=700
)

fig.show()
