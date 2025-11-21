import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


class PixelArtGenerator:
    def __init__(self):
        """
        初始化像素艺术生成器
        """
        self.progress_callback = None

    def set_progress_callback(self, callback):
        """
        设置进度回调函数
        
        Args:
            callback: 回调函数，接受两个参数：(progress_type, value)
                     progress_type: 进度类型 ('load', 'process', 'match', 'total')
                     value: 进度值 (0-100)
        """
        self.progress_callback = callback

    def resize_image_to_grid(self, image, grid_size):
        """
        将图像调整为指定网格大小
        
        Args:
            image: 输入图像 (numpy array)
            grid_size: 网格大小 (width, height)
            
        Returns:
            调整大小后的图像
        """
        width, height = grid_size
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        return resized

    def get_dominant_color(self, image_section):
        """
        获取图像区域的主要颜色（使用K-means聚类）
        
        Args:
            image_section: 图像的一部分
            
        Returns:
            主要RGB颜色值
        """
        # 将图像数据重塑为像素列表
        pixels = image_section.reshape((-1, 3))
        pixels = np.float32(pixels)
        
        # 使用K-means聚类找出主要颜色
        kmeans = KMeans(n_clusters=1, random_state=42, n_init=1)
        kmeans.fit(pixels)
        
        # 获取聚类中心（主要颜色）
        dominant_color = kmeans.cluster_centers_[0]
        
        # 转换为整数
        dominant_color = np.uint8(dominant_color)
        
        return tuple(dominant_color)

    def get_color_features(self, image_section):
        """
        获取图像区域的颜色特征（包括平均颜色和主要颜色）
        
        Args:
            image_section: 图像的一部分 (BGR格式)
            
        Returns:
            颜色特征元组 (avg_color, dominant_color)
        """
        # 转换为RGB
        rgb_section = cv2.cvtColor(image_section, cv2.COLOR_BGR2RGB)
        
        # 计算平均颜色
        avg_color = np.mean(rgb_section.reshape(-1, 3), axis=0)
        avg_color = tuple(avg_color.astype(int))
        
        # 计算主要颜色
        dominant_color = self.get_dominant_color(rgb_section)
        
        return (avg_color, dominant_color)

    def create_pixelated_image(self, image, pixel_size):
        """
        创建像素化图像
        
        Args:
            image: 原始图像
            pixel_size: 像素块大小
            
        Returns:
            像素化图像
        """
        height, width = image.shape[:2]
        
        # 计算网格尺寸
        grid_width = width // pixel_size
        grid_height = height // pixel_size
        
        # 调整图像到网格大小然后放大回原始尺寸
        small_image = cv2.resize(image, (grid_width, grid_height), interpolation=cv2.INTER_LINEAR)
        pixelated_image = cv2.resize(small_image, (width, height), interpolation=cv2.INTER_NEAREST)
        
        return pixelated_image

    def extract_dominant_colors(self, image, num_colors=5):
        """
        使用K-means算法提取图像中的主要颜色
        
        Args:
            image: 输入图像
            num_colors: 主要颜色数量
            
        Returns:
            主要颜色列表
        """
        # 将图像数据重塑为像素列表
        pixels = image.reshape((-1, 3))
        pixels = np.float32(pixels)
        
        # 使用K-means聚类找出主要颜色
        kmeans = KMeans(n_clusters=num_colors, random_state=42)
        kmeans.fit(pixels)
        
        # 获取聚类中心（主要颜色）
        dominant_colors = kmeans.cluster_centers_
        
        # 转换为整数
        dominant_colors = np.uint8(dominant_colors)
        
        return dominant_colors

    def calculate_color_distance(self, color_features1, color_features2):
        """
        计算两个颜色特征之间的距离
        
        Args:
            color_features1: 第一个颜色特征 (avg_color, dominant_color)
            color_features2: 第二个颜色特征 (avg_color, dominant_color)
            
        Returns:
            颜色距离值
        """
        avg_color1, dominant_color1 = color_features1
        avg_color2, dominant_color2 = color_features2
        
        # 计算平均颜色距离
        avg_distance = np.sqrt(np.sum((np.array(avg_color1) - np.array(avg_color2)) ** 2))
        
        # 计算主要颜色距离
        dominant_distance = np.sqrt(np.sum((np.array(dominant_color1) - np.array(dominant_color2)) ** 2))
        
        # 综合距离（主要颜色权重更高）
        combined_distance = 0.3 * avg_distance + 0.7 * dominant_distance
        
        return combined_distance

    def find_closest_image_by_features(self, target_features, image_library_features):
        """
        根据颜色特征在图像库中找到最接近的图像
        
        Args:
            target_features: 目标颜色特征
            image_library_features: 图像库颜色特征字典 {path: features}
            
        Returns:
            最接近的图像路径
        """
        closest_image = None
        min_distance = float('inf')
        
        # 遍历图像库
        for img_path, features in image_library_features.items():
            # 计算颜色特征距离
            distance = self.calculate_color_distance(target_features, features)
            
            if distance < min_distance:
                min_distance = distance
                closest_image = img_path
                
        return closest_image

    def _process_grid_cell(self, args):
        """
        处理单个网格单元的内部函数
        
        Args:
            args: 包含(i, j, target_image, pixel_size, image_library_features)的元组
            
        Returns:
            (i, j, replacement_data) 处理结果
        """
        i, j, target_image, pixel_size, image_library_features = args
        height, width = target_image.shape[:2]
        
        # 计算网格单元坐标
        y_start = i * pixel_size
        y_end = min((i + 1) * pixel_size, height)
        x_start = j * pixel_size
        x_end = min((j + 1) * pixel_size, width)
        
        # 获取网格单元内的颜色特征
        grid_section = target_image[y_start:y_end, x_start:x_end]
        target_features = self.get_color_features(grid_section)
        
        # 在图像库中找到最接近的图像
        closest_img_path = self.find_closest_image_by_features(target_features, image_library_features)
        
        if closest_img_path:
            # 加载并调整匹配图像的大小
            replacement_img = cv2.imread(closest_img_path)
            if replacement_img is not None:
                replacement_img = cv2.resize(replacement_img, (x_end - x_start, y_end - y_start))
                return (i, j, (y_start, y_end, x_start, x_end, replacement_img))
        
        # 如果没有找到匹配图像，则使用纯色填充
        avg_color, dominant_color = target_features
        color_block = np.full((y_end - y_start, x_end - x_start, 3), 
                             [int(c) for c in avg_color[::-1]],  # RGB to BGR
                             dtype=np.uint8)
        return (i, j, (y_start, y_end, x_start, x_end, color_block))

    def create_mosaic_image(self, target_image, pixel_size, image_library):
        """
        创建马赛克图像（并行版本）
        
        Args:
            target_image: 目标图像
            pixel_size: 像素块大小
            image_library: 图像库路径列表
            
        Returns:
            马赛克图像
        """
        height, width = target_image.shape[:2]
        
        # 计算网格尺寸
        grid_width = width // pixel_size
        grid_height = height // pixel_size
        
        # 创建输出图像
        mosaic_image = np.zeros_like(target_image)
        
        # 更新总进度
        if self.progress_callback:
            self.progress_callback('total', 20)
        
        # 预先加载图像库的颜色特征
        image_library_features = {}
        for idx, img_path in enumerate(image_library):
            try:
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        # 获取图像的颜色特征
                        features = self.get_color_features(img)
                        image_library_features[img_path] = features
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
            
            # 更新匹配进度
            if self.progress_callback:
                self.progress_callback('match', int((idx+1) / len(image_library) * 100))
        
        # 更新总进度
        if self.progress_callback:
            self.progress_callback('total', 40)
        
        total_cells = grid_height * grid_width
        processed_cells = 0
        
        # 准备并行处理任务
        tasks = []
        for i in range(grid_height):
            for j in range(grid_width):
                tasks.append((i, j, target_image, pixel_size, image_library_features))
        
        # 使用线程池并行处理网格单元
        with ThreadPoolExecutor(max_workers=8) as executor:
            # 提交所有任务
            future_to_task = {executor.submit(self._process_grid_cell, task): task for task in tasks}
            
            # 收集结果
            for future in as_completed(future_to_task):
                try:
                    i, j, replacement_data = future.result()
                    y_start, y_end, x_start, x_end, replacement_img = replacement_data
                    
                    # 将替换图像放入马赛克图像中
                    mosaic_image[y_start:y_end, x_start:x_end] = replacement_img
                    
                    # 更新处理进度
                    processed_cells += 1
                    if self.progress_callback:
                        # 对于处理图像进度，我们传递当前处理的像素片索引和总像素片数
                        self.progress_callback('process_detail', (processed_cells, total_cells))
                        self.progress_callback('total', 40 + int(processed_cells / total_cells * 60))
                        
                except Exception as e:
                    print(f"Error processing grid cell ({i}, {j}): {e}")
                    processed_cells += 1
                    if self.progress_callback:
                        self.progress_callback('process_detail', (processed_cells, total_cells))
                        self.progress_callback('total', 40 + int(processed_cells / total_cells * 60))
        
        return mosaic_image