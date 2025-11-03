import glob
import os

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 必须导入以启用3D
from sklearn.svm import SVC


def basic_features(curve):
    return [
        np.mean(curve),  # 均值
        np.std(curve),  # 标准差
        np.min(curve), np.max(curve),  # 极值
        np.median(curve),  # 中位数
        np.percentile(curve, 25), np.percentile(curve, 75),  # 分位数
        np.mean(np.diff(curve)),  # 一阶差分均值
        np.std(np.diff(curve)),  # 一阶差分标准差
    ]


def signal_features(curve):
    curve = curve - np.mean(curve)
    fft_vals = np.abs(rfft(curve))
    freqs = rfftfreq(len(curve))

    peaks, _ = find_peaks(curve)
    num_peaks = len(peaks)
    mean_peak_height = np.mean(curve[peaks]) if num_peaks > 0 else 0

    return [
        np.sum(fft_vals[1:5]),  # 低频能量
        np.sum(fft_vals[5:20]),  # 高频能量
        np.argmax(fft_vals[1:]),  # 主频索引
        num_peaks,  # 峰数
        mean_peak_height,  # 峰高平均值
    ]


def geometric_features(curve):
    x = np.arange(len(curve))
    diff = np.max(curve) - np.min(curve)
    deriv = np.gradient(curve)
    curvature = np.mean(np.abs(np.gradient(deriv)))
    area = np.trapz(curve, x)
    roughness = np.mean(np.abs(np.diff(curve)))

    # return [area, curvature, roughness]
    return [np.max(curve)]


def extract_features(curve):
    f1 = basic_features(curve)
    f2 = signal_features(curve)
    f3 = geometric_features(curve)
    return np.hstack([f1, f2, f3])


def extract_features_from_file(filepath, ranges=[(-0.2, 0.2), (0.6, 1)]):
    # ranges = [(-np.inf, np.inf)]

    """从一个xlsx文件提取特征向量"""
    df = pd.read_excel(filepath)

    # 去掉表头中的空格等
    df.columns = [str(c).strip() for c in df.columns]

    # 横坐标 (Voltage)
    voltage = df.iloc[:, 0].values.ravel()  # 保证一维
    currents = df.iloc[:, 1:].values  # 其他列为电流矩阵 (n点 × m曲线)

    # print(currents.shape)

    features = []

    for i in range(currents.shape[1]):
        current = currents[:, i]
        cnt = 0
        feat = []
        for vmin, vmax in ranges:
            cnt += 1
            mask = (voltage >= vmin) & (voltage <= vmax)
            v_sub = voltage[mask]
            c_sub = current[mask]  # 取所有曲线在这个电压区间的点

            # 把这些都作为特征
            if cnt == 1:
                feat_1 = np.max(c_sub)
                feat.append(feat_1)
                # print(feat)
            elif cnt == 2:
                feat_2 = np.std(np.diff(c_sub))
                feat.append(feat_2)
            elif cnt == 3:
                feat_3 = np.std(np.diff(c_sub))
                feat.append(feat_3)
        # 子区间循环结束
        features.append(feat)

    # print(f"{filepath} : {np.round(features, 2)}")

    return np.array(features)


def load_dataset(base_dir, ranges):
    """读取所有文件，构建 (X,y)"""
    X, y = [], []
    class_labels = os.listdir(base_dir)

    for label in class_labels:
        folder = os.path.join(base_dir, label)
        for filepath in glob.glob(os.path.join(folder, "*.xlsx")):
            feats = extract_features_from_file(filepath, ranges)
            for feat in feats:
                X.append(feat)
                y.append(label)

    return np.array(X), np.array(y)


def train_and_evaluate(X, y, method="svm", seed=1):
    if method == "svm":
        """训练分类器并评估"""
        # ✅ SVM 必须配合标准化
        clf = make_pipeline(
            StandardScaler(),
            SVC(kernel='rbf', probability=True, C=1, gamma='scale', random_state=seed)
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    elif method == "rf":
        clf = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=400, random_state=seed)
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # 交叉验证预测
    y_pred = cross_val_predict(clf, X, y, cv=cv)

    report = classification_report(y, y_pred)
    print("分类报告：")
    print(report)

    c_matrix = confusion_matrix(y, y_pred)
    print("混淆矩阵：")
    print(c_matrix)

    return report, c_matrix


def preprocess_outliers(X, y, method="iqr", threshold=0.5):
    """
    对于每个标签分组的数据，逐列检测离群值并替换为正常值均值。

    参数:
        X : numpy.ndarray, shape (n_samples, n_features)
        y : numpy.ndarray, shape (n_samples,)
        method : str, "iqr" 或 "zscore"
        threshold : float, IQR系数 (默认1.5) 或 z-score阈值 (默认3)

    返回:
        X_new : numpy.ndarray (处理后的矩阵)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    X_new = X.copy()

    for label in np.unique(y):  # 多个标签都会单独处理
        mask = (y == label)
        X_group = X[mask]  # 当前标签对应的子矩阵

        for col in range(X.shape[1]):
            col_data = X_group[:, col]

            # 1. 离群值检测
            if method == "iqr":
                Q1, Q3 = np.percentile(col_data, [25, 75])
                IQR = Q3 - Q1
                lower, upper = Q1 - threshold * IQR, Q3 + threshold * IQR
                outliers = (col_data < lower) | (col_data > upper)
            elif method == "zscore":
                mean, std = np.mean(col_data), np.std(col_data)
                z = (col_data - mean) / (std + 1e-8)
                outliers = np.abs(z) > threshold
            else:
                raise ValueError("Unknown method, choose 'iqr' or 'zscore'")

            # 2. 替换离群值
            normal_vals = col_data[~outliers]
            if len(normal_vals) > 0:
                replacement = np.mean(normal_vals)
                col_data[outliers] = replacement

            # 写回原矩阵
            X_new[mask, col] = col_data

    return X_new


if __name__ == "__main__":
    base_dir = "data"  # 三个子文件夹所在目录
    method = "svm"
    save_file = "classified_results.txt"
    # 指定提取特征的电压区间
    ranges = [(-0.5, 0), (0.75, 0.85)]
    ranges = [(1.2, 1.5), (-0.5, 0.5), (0.75, 0.85)]
    # ranges = [(-np.inf, np.inf)]

    X, y = load_dataset(base_dir, ranges)
    # print(X, y)
    A = np.hstack([X, np.transpose([y])])

    # 指定每一列的倍数（长度 = 特征列数）
    scales = np.array([1e4, 1e6, 1e6])  # 每列乘以不同的系数

    n_features = A.shape[1] - 1  # 特征列数
    valid_scales = scales[:n_features]  # 自动截取匹配长度

    # 对特征部分按列缩放
    A_scaled = A.copy().astype(object)  # 保留不同类型列的兼容性
    A_scaled[:, :-1] = np.round(A[:, :-1].astype(float) * valid_scales, 1)

    A = A_scaled

    for label in np.unique(A[:, -1]):
        subset = A[A[:, -1] == label]  # 取出该标签对应的行
        print(f"\nLabel {label}:")
        print(subset[:10])  # 打印前10个（若不足10个就打印全部）
    # print(A)

    report, c_matrix = train_and_evaluate(X, y, method=method)

    # # 处理离群值！
    X = preprocess_outliers(X, y)
    A = np.hstack([X, np.transpose([y])])
    # 指定每一列的倍数（长度 = 特征列数）
    scales = np.array([1e4, 1e6, 1e8])  # 每列乘以不同的系数

    n_features = A.shape[1] - 1  # 特征列数
    valid_scales = scales[:n_features]  # 自动截取匹配长度

    # 对特征部分按列缩放
    A_scaled = A.copy().astype(object)  # 保留不同类型列的兼容性
    A_scaled[:, :-1] = np.round(A[:, :-1].astype(float) * valid_scales, 1)
    A = A_scaled
    for label in np.unique(A[:, -1]):
        subset = A[A[:, -1] == label]  # 取出该标签对应的行
        print(f"\nLabel {label}:")
        print(subset[:10])  # 打印前10个（若不足10个就打印全部）
    # print(A)

    report, c_matrix = train_and_evaluate(X, y, method=method)
    # # 保存分类结果
    with open(save_file, "w", encoding="utf-8") as f:
        f.write(f"分类报告：\n{report}")
        f.write(f"混淆矩阵：\n{c_matrix}")
        print(f"file {save_file} saved.")

        # A.shape = (N, 4)，前3列是坐标，最后1列是标签
        # 示例：
        # A = np.random.rand(100, 4)
        # A[:, -1] = np.random.randint(0, 3, size=100)

        # 1️⃣ 创建3D绘图
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # 颜色列表（可根据类别数量调整）
        colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(A[:, -1]))))

        # 2️⃣ 遍历标签分组
        for i, label in enumerate(np.unique(A[:, -1])):
            subset = A[A[:, -1] == label]
            x, y, z = subset[:, 0], subset[:, 1], subset[:, 2]
            ax.scatter(x, y, z, color=colors[i], label=f'Label {label}', s=30, alpha=0.8)

        # 3️⃣ 美化图形
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend(title="Classes")
        ax.set_title("3D Scatter Plot by Label")
        plt.tight_layout()

        # 4️⃣ 保存图像
        plt.savefig("3d_scatter_by_label.png", dpi=300)
        plt.show()

        # 5️⃣ 保存数据为 CSV（方便外部绘图）
        df = pd.DataFrame(A, columns=['x', 'y', 'z', 'label'])
        df.to_csv("3d_points_with_label.csv", index=False)
        print("✅ 图像已保存为 3d_scatter_by_label.png")
        print("✅ 数据已保存为 3d_points_with_label.csv")
