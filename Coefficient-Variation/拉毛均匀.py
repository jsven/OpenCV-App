import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_bush_hammered_uniformity(image_path, threshold=60, grid_size=60, cv_threshold=0.2):
    """
    分析拉毛混凝土图像的均匀性，并可视化结果。

    Args:
        image_path: 拉毛混凝土图像的路径。
        threshold: 用于二值化图像的阈值。
        grid_size: 用于划分图像的网格大小。

    Returns:
        一个字典，包含以下信息：
        - "uniformity":  拉毛均匀性的布尔值 (True 或 False)。
        - "grid_deviations": 一个包含每个网格标准差的列表。
    """
    print(cv_threshold)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("无法读取图像。请检查文件路径。")

    # 使用自适应阈值进行二值化
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    # 可视化二值化后的图像
    cv2.imshow("Binary Image", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # 计算每个网格的标准差
    height, width = img.shape
    grid_deviations = []
    for y in range(0, height, grid_size):
        for x in range(0, width, grid_size):
            grid = thresh[y:min(y + grid_size, height), x:min(x + grid_size, width)]
            if grid.size > 0:  # 避免空网格
                std_dev = np.std(grid)
                grid_deviations.append(std_dev)

    # 通过标准差的变异系数来评估均匀性
    if len(grid_deviations) > 0:
        coefficient_of_variation = np.std(grid_deviations) / np.mean(grid_deviations)
        print(f"Coefficient of Variation: {coefficient_of_variation}")
        uniformity = coefficient_of_variation < cv_threshold  #  根据变异系数设置阈值，可调整
    else:
        uniformity = False

    # 绘制 grid_deviations 的直方图
    plt.hist(grid_deviations, bins=20)
    plt.xlabel("Standard Deviation")
    plt.ylabel("Frequency")
    plt.title("Distribution of Grid Standard Deviations")
    plt.show()

    return {"uniformity": uniformity, "grid_deviations": grid_deviations}


# 示例用法
image_path = "img_1.png"  # 请替换为你的图像路径
result = analyze_bush_hammered_uniformity(image_path, cv_threshold=0.3)  # 根据需要调整参数
print(f"拉毛均匀性: {result['uniformity']}")