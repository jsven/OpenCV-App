import cv2  # 导入 OpenCV 库
import numpy as np  # 导入 NumPy 库
import matplotlib.pyplot as plt
def analyze_texture(img_path):
    """
    分析图像的纹理特征，例如均匀性、一致性和边缘长度。

    Args:
        img_path: 图像文件的路径。

    Returns:
        一个包含分析结果的字典。
    """
    # 读取图像，并转换为灰度图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_original = cv2.imread(img_path)
    # 预处理：使用中值滤波去除噪声
    img = cv2.medianBlur(img, 5)  # 5 是内核大小，可以根据需要调整

    # Blob 检测：使用 SimpleBlobDetector 检测拉毛的凸起部分
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True  # 根据面积过滤 Blob
    params.minArea = 10  # 最小面积，需要根据实际情况调整
    params.maxArea = 100 # 最大面积，需要根据实际情况调整
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)  # 检测到的 Blob

    # 计算 Blob 特征
    centers = [kp.pt for kp in keypoints]  # 获取 Blob 的中心点坐标
    areas = [kp.size for kp in keypoints]  # 获取 Blob 的面积（直径）

    # 均匀性：计算中心点坐标的方差，值越小越均匀
    if centers:  # 检查是否检测到 Blob
        centers = np.array(centers)  # 将列表转换为 NumPy 数组
        center_variance = np.var(centers, axis=0) # 计算 x 和 y 坐标的方差
        uniformity = np.mean(center_variance)  # 计算平均方差
    else:
        uniformity = float('inf')  # 没有检测到 Blob，均匀性设为无穷大


    # 一致性：计算 Blob 面积的标准差，值越小越一致
    if areas: # 检查是否检测到 Blob
        areas = np.array(areas) # 将列表转换为 NumPy 数组
        area_std = np.std(areas) # 计算面积的标准差
        consistency = area_std
    else:
        consistency = float('inf')  # 没有检测到 Blob，一致性设为无穷大


    # 边缘检测：使用 Canny 算子检测边缘
    edges = cv2.Canny(img, 50, 150) # 50 和 150 是 Canny 算子的阈值，需要根据实际情况调整
    total_edge_length = np.sum(edges > 0) # 计算边缘像素的总数，可以粗略代表边缘长度

    # 将结果存储在字典中
    results = {
        "uniformity": uniformity,
        "consistency": consistency,
        "total_edge_length": total_edge_length,
        "num_blobs": len(keypoints)  # 检测到的 Blob 数量
    }

    return results, img_original, img, edges # 返回更多用于显示的图像


# 分析两张图片
img1_results, img1_original, img1_gray, img1_edges = analyze_texture("img.png")
img2_results, img2_original, img2_gray, img2_edges  = analyze_texture("img_1.png")

# 打印分析结果
print("Image 1:", img1_results)
print("Image 2:", img2_results)

# --- 绘制图表 ---
#
# # 创建图表
# fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2x2 的子图
# fig.suptitle("Texture Analysis Results")  # 主标题
#
# # 绘制均匀性
# axes[0, 0].bar(["Image 1", "Image 2"], [img1_results["uniformity"], img2_results["uniformity"]])
# axes[0, 0].set_title("Uniformity (Lower is better)")
# axes[0, 0].set_ylabel("Variance")
#
# # 绘制一致性
# axes[0, 1].bar(["Image 1", "Image 2"], [img1_results["consistency"], img2_results["consistency"]])
# axes[0, 1].set_title("Consistency (Lower is better)")
# axes[0, 1].set_ylabel("Standard Deviation")
#
# # 绘制边缘长度
# axes[1, 0].bar(["Image 1", "Image 2"], [img1_results["total_edge_length"], img2_results["total_edge_length"]])
# axes[1, 0].set_title("Total Edge Length")
# axes[1, 0].set_ylabel("Pixel Count")
#
# # 绘制 Blob 数量
# axes[1, 1].bar(["Image 1", "Image 2"], [img1_results["num_blobs"], img2_results["num_blobs"]])
# axes[1, 1].set_title("Number of Blobs")
# axes[1, 1].set_ylabel("Count")
#
# plt.tight_layout()  # 调整子图布局，避免重叠
# plt.show()  # 显示图表



# --- 显示图像 ---

# 创建一个新的 Figure 和 Axes 用于显示图像
fig_images, axes_images = plt.subplots(2, 3, figsize=(12, 8))
fig_images.suptitle("Image Processing Steps")

# 显示第一张图片
axes_images[0, 0].imshow(cv2.cvtColor(img1_original, cv2.COLOR_BGR2RGB)) # 转换颜色通道用于显示
axes_images[0, 0].set_title("Image 1 (Original)")
axes_images[0, 0].axis('off') # 关闭坐标轴

axes_images[0, 1].imshow(img1_gray, cmap='gray')
axes_images[0, 1].set_title("Image 1 (Grayscale)")
axes_images[0, 1].axis('off')

axes_images[0, 2].imshow(img1_edges, cmap='gray')
axes_images[0, 2].set_title("Image 1 (Edges)")
axes_images[0, 2].axis('off')


# 显示第二张图片
axes_images[1, 0].imshow(cv2.cvtColor(img2_original, cv2.COLOR_BGR2RGB))
axes_images[1, 0].set_title("Image 2 (Original)")
axes_images[1, 0].axis('off')

axes_images[1, 1].imshow(img2_gray, cmap='gray')
axes_images[1, 1].set_title("Image 2 (Grayscale)")
axes_images[1, 1].axis('off')

axes_images[1, 2].imshow(img2_edges, cmap='gray')
axes_images[1, 2].set_title("Image 2 (Edges)")
axes_images[1, 2].axis('off')


plt.tight_layout()
plt.show()

# 根据分析结果进行判断，需要根据实际情况设定阈值
if img1_results["uniformity"] < img2_results["uniformity"] and img1_results["consistency"] < img2_results["consistency"]:
    print("Image 1 has better texture.")
elif img2_results["uniformity"] < img1_results["uniformity"] and img2_results["consistency"] < img1_results["consistency"]:
    print("Image 2 has better texture.")
else:
    print("Texture quality is similar or inconclusive.")