import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import os
from pixel_art_generator import PixelArtGenerator
import threading
import queue
from datetime import datetime
import webbrowser


class PixelArtApp:
    def __init__(self, root):
        self.root = root
        self.root.title("像素艺术生成器")
        self.root.geometry("1000x700")
        
        self.generator = PixelArtGenerator()
        # 设置进度回调
        self.generator.set_progress_callback(self.update_progress)
        
        self.target_image_path = None
        self.image_library_path = None
        self.output_image_path = None
        self.pixel_size = 20
        
        # 用于后台处理的队列和线程
        self.result_queue = queue.Queue()
        self.process_thread = None
        self.is_processing = False
        
        # 图像显示
        self.target_image = None
        self.result_image = None
        self.tk_target_image = None
        self.tk_result_image = None
        
        self.setup_ui()
    
    def setup_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="像素艺术生成器", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 选择目标图像
        target_image_btn = ttk.Button(main_frame, text="选择目标图像", command=self.select_target_image)
        target_image_btn.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.target_image_label = ttk.Label(main_frame, text="未选择图像")
        self.target_image_label.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # 选择图像库
        image_library_btn = ttk.Button(main_frame, text="选择图像库目录", command=self.select_image_library)
        image_library_btn.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.image_library_label = ttk.Label(main_frame, text="未选择图像库")
        self.image_library_label.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        # 像素大小设置
        pixel_size_label = ttk.Label(main_frame, text="像素大小:")
        pixel_size_label.grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.pixel_size_var = tk.StringVar(value=str(self.pixel_size))
        pixel_size_entry = ttk.Entry(main_frame, textvariable=self.pixel_size_var, width=10)
        pixel_size_entry.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
        
        # 生成按钮
        self.generate_btn = ttk.Button(main_frame, text="生成像素艺术", command=self.generate_pixel_art)
        self.generate_btn.grid(row=4, column=0, columnspan=2, padx=5, pady=20, sticky=tk.W)
        
        # 步骤进度条框架
        progress_frame = ttk.LabelFrame(main_frame, text="处理进度", padding="10")
        progress_frame.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        progress_frame.columnconfigure(1, weight=1)
        
        # 加载图像进度
        self.load_image_progress_label = ttk.Label(progress_frame, text="加载目标图像:")
        self.load_image_progress_label.grid(row=0, column=0, sticky=tk.W, pady=2)
        
        self.load_image_progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.load_image_progress.grid(row=0, column=1, padx=(5, 0), sticky=(tk.W, tk.E), pady=2)
        
        # 处理图像进度
        self.process_image_progress_label = ttk.Label(progress_frame, text="处理图像:")
        self.process_image_progress_label.grid(row=1, column=0, sticky=tk.W, pady=2)
        
        self.process_image_progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.process_image_progress.grid(row=1, column=1, padx=(5, 0), sticky=(tk.W, tk.E), pady=2)
        
        # 处理图像详细进度标签
        self.process_detail_label = ttk.Label(progress_frame, text="")
        self.process_detail_label.grid(row=2, column=1, padx=(5, 0), sticky=tk.W, pady=2)
        
        # 匹配图像进度
        self.match_image_progress_label = ttk.Label(progress_frame, text="匹配图像:")
        self.match_image_progress_label.grid(row=3, column=0, sticky=tk.W, pady=2)
        
        self.match_image_progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.match_image_progress.grid(row=3, column=1, padx=(5, 0), sticky=(tk.W, tk.E), pady=2)
        
        # 总体进度
        self.total_progress_label = ttk.Label(progress_frame, text="总体进度:")
        self.total_progress_label.grid(row=4, column=0, sticky=tk.W, pady=2)
        
        self.total_progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.total_progress.grid(row=4, column=1, padx=(5, 0), sticky=(tk.W, tk.E), pady=2)
        
        # 图像显示区域
        images_frame = ttk.Frame(main_frame)
        images_frame.grid(row=6, column=0, columnspan=2, padx=5, pady=10, sticky=(tk.W, tk.E, tk.N, tk.S))
        images_frame.columnconfigure(0, weight=1)
        images_frame.columnconfigure(1, weight=1)
        images_frame.rowconfigure(0, weight=1)
        
        # 目标图像显示区域
        target_frame = ttk.LabelFrame(images_frame, text="目标图像", padding="5")
        target_frame.grid(row=0, column=0, padx=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        target_frame.columnconfigure(0, weight=1)
        target_frame.rowconfigure(0, weight=1)
        
        self.target_canvas = tk.Canvas(target_frame, bg="white")
        self.target_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 结果显示区域
        result_frame = ttk.LabelFrame(images_frame, text="结果预览", padding="5")
        result_frame.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
        self.result_canvas = tk.Canvas(result_frame, bg="white")
        self.result_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 打开结果按钮
        self.open_result_btn = ttk.Button(main_frame, text="打开结果文件", command=self.open_result_file, state=tk.DISABLED)
        self.open_result_btn.grid(row=7, column=0, padx=5, pady=10, sticky=tk.W)
        
        # 配置行权重
        main_frame.rowconfigure(6, weight=1)
        
        # 定期检查结果队列
        self.check_result_queue()
    
    def select_target_image(self):
        """选择目标图像"""
        file_path = filedialog.askopenfilename(
            title="选择目标图像",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        
        if file_path:
            self.target_image_path = file_path
            self.target_image_label.config(text=os.path.basename(file_path))
            self.load_target_image()
    
    def load_target_image(self):
        """加载并显示目标图像"""
        if self.target_image_path and os.path.exists(self.target_image_path):
            try:
                # 使用OpenCV加载图像
                target_image_bgr = cv2.imread(self.target_image_path)
                if target_image_bgr is not None:
                    # 转换颜色格式 BGR to RGB
                    target_image_rgb = cv2.cvtColor(target_image_bgr, cv2.COLOR_BGR2RGB)
                    # 转换为PIL Image
                    self.target_image = Image.fromarray(target_image_rgb)
                    # 显示图像
                    self.display_target_image()
            except Exception as e:
                messagebox.showerror("错误", f"无法加载目标图像: {str(e)}")
    
    def display_target_image(self):
        """显示目标图像"""
        if self.target_image is not None:
            # 调整图像大小以适应显示区域
            canvas_width = self.target_canvas.winfo_width()
            canvas_height = self.target_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:  # 确保canvas已初始化
                # 计算缩放比例
                scale_w = canvas_width / self.target_image.width
                scale_h = canvas_height / self.target_image.height
                scale = min(scale_w, scale_h, 1.0)  # 不放大图像
                
                if scale < 1.0:
                    new_width = int(self.target_image.width * scale)
                    new_height = int(self.target_image.height * scale)
                    resized_image = self.target_image.resize((new_width, new_height), Image.LANCZOS)
                else:
                    resized_image = self.target_image
                
                # 转换为tkinter PhotoImage
                self.tk_target_image = ImageTk.PhotoImage(resized_image)
                
                # 清除画布并显示新图像
                self.target_canvas.delete("all")
                x = (self.target_canvas.winfo_width() - resized_image.width) // 2
                y = (self.target_canvas.winfo_height() - resized_image.height) // 2
                self.target_canvas.create_image(x, y, anchor=tk.NW, image=self.tk_target_image)
    
    def select_image_library(self):
        """选择图像库目录"""
        folder_path = filedialog.askdirectory(title="选择图像库目录")
        
        if folder_path:
            self.image_library_path = folder_path
            self.image_library_label.config(text=os.path.basename(folder_path))
    
    def get_image_files_from_directory(self, directory):
        """从目录中获取所有图像文件"""
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        image_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_files.append(os.path.join(root, file))
        
        return image_files
    
    def update_progress(self, progress_type, value):
        """更新进度条"""
        # 将进度更新请求放入队列，在主线程中处理
        self.result_queue.put(('progress', progress_type, value))
    
    def check_result_queue(self):
        """定期检查结果队列并处理其中的消息"""
        try:
            while True:
                item = self.result_queue.get_nowait()
                if item[0] == 'progress':
                    _, progress_type, value = item
                    if progress_type == 'load':
                        self.load_image_progress['value'] = value
                    elif progress_type == 'process':
                        self.process_image_progress['value'] = value
                    elif progress_type == 'process_detail':
                        # 处理详细进度 (current, total)
                        current, total = value
                        self.process_image_progress['value'] = int(current / total * 100)
                        self.process_detail_label.config(text=f"已完成: {current}/{total} 像素片")
                    elif progress_type == 'match':
                        self.match_image_progress['value'] = value
                    elif progress_type == 'total':
                        self.total_progress['value'] = value
                elif item[0] == 'result':
                    _, result_image = item
                    self.result_image = result_image
                    self.total_progress['value'] = 100
                    self.generate_btn.config(state=tk.NORMAL, text="生成像素艺术")
                    self.open_result_btn.config(state=tk.NORMAL)
                    self.is_processing = False
                    self.display_result_image()
                    
                    # 自动保存图片
                    self.auto_save_result()
                    
                    messagebox.showinfo("成功", "像素艺术生成并自动保存完成")
                elif item[0] == 'error':
                    _, error_msg = item
                    self.generate_btn.config(state=tk.NORMAL, text="生成像素艺术")
                    self.is_processing = False
                    messagebox.showerror("错误", f"生成过程中出现错误: {error_msg}")
                elif item[0] == 'warning':
                    _, warning_msg = item
                    messagebox.showwarning("警告", warning_msg)
        except queue.Empty:
            pass
        
        # 继续定期检查队列
        self.root.after(100, self.check_result_queue)
    
    def generate_pixel_art(self):
        """生成像素艺术"""
        if self.is_processing:
            # 如果已经在处理中，取消操作
            return
        
        # 验证输入
        if not self.target_image_path:
            messagebox.showerror("错误", "请选择目标图像")
            return
        
        if not self.image_library_path:
            messagebox.showerror("错误", "请选择图像库目录")
            return
        
        try:
            pixel_size = int(self.pixel_size_var.get())
            if pixel_size <= 0:
                raise ValueError("像素大小必须是正整数")
            self.pixel_size = pixel_size
        except ValueError as e:
            messagebox.showerror("错误", f"无效的像素大小: {e}")
            return
        
        # 启动后台线程进行处理
        self.is_processing = True
        self.generate_btn.config(state=tk.DISABLED, text="处理中...")
        self.open_result_btn.config(state=tk.DISABLED)
        
        # 重置所有进度条
        self.load_image_progress['value'] = 0
        self.process_image_progress['value'] = 0
        self.process_detail_label.config(text="")
        self.match_image_progress['value'] = 0
        self.total_progress['value'] = 0
        
        # 启动处理线程
        self.process_thread = threading.Thread(target=self._process_image_in_background)
        self.process_thread.daemon = True
        self.process_thread.start()
    
    def _process_image_in_background(self):
        """在后台线程中处理图像"""
        try:
            # 获取图像库中的所有图像文件
            self.result_queue.put(('progress', 'match', 0))
            image_library = self.get_image_files_from_directory(self.image_library_path)
            if not image_library:
                self.result_queue.put(('warning', "图像库中没有找到图像文件"))
                self.result_queue.put(('progress', 'total', 100))
                return
            
            # 加载目标图像
            self.result_queue.put(('progress', 'load', 50))
            target_image = cv2.imread(self.target_image_path)
            if target_image is None:
                raise ValueError("无法加载目标图像")
            
            self.result_queue.put(('progress', 'load', 100))
            self.result_queue.put(('progress', 'total', 25))
            
            # 计算总像素片数并更新标签
            height, width = target_image.shape[:2]
            grid_width = width // self.pixel_size
            grid_height = height // self.pixel_size
            total_cells = grid_height * grid_width
            self.result_queue.put(('label_update', f"处理图像 ({total_cells} 像素片):"))
            
            # 生成马赛克图像
            result_image = self.generator.create_mosaic_image(
                target_image, self.pixel_size, image_library
            )
            
            # 发送结果到主线程
            self.result_queue.put(('result', result_image))
        except Exception as e:
            # 发送错误消息到主线程
            self.result_queue.put(('error', str(e)))
    
    def display_result_image(self):
        """显示结果图像"""
        if self.result_image is not None:
            # 转换颜色格式
            rgb_image = cv2.cvtColor(self.result_image, cv2.COLOR_BGR2RGB)
            
            # 转换为PIL图像
            self.result_pil_image = Image.fromarray(rgb_image)
            
            # 调整图像大小以适应显示区域
            canvas_width = self.result_canvas.winfo_width()
            canvas_height = self.result_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:  # 确保canvas已初始化
                # 计算缩放比例
                scale_w = canvas_width / self.result_pil_image.width
                scale_h = canvas_height / self.result_pil_image.height
                scale = min(scale_w, scale_h, 1.0)  # 不放大图像
                
                if scale < 1.0:
                    new_width = int(self.result_pil_image.width * scale)
                    new_height = int(self.result_pil_image.height * scale)
                    resized_image = self.result_pil_image.resize((new_width, new_height), Image.LANCZOS)
                else:
                    resized_image = self.result_pil_image
                
                # 转换为tkinter PhotoImage
                self.tk_result_image = ImageTk.PhotoImage(resized_image)
                
                # 清除画布并显示新图像
                self.result_canvas.delete("all")
                x = (self.result_canvas.winfo_width() - resized_image.width) // 2
                y = (self.result_canvas.winfo_height() - resized_image.height) // 2
                self.result_canvas.create_image(x, y, anchor=tk.NW, image=self.tk_result_image)
    
    def auto_save_result(self):
        """自动保存结果图像"""
        if self.result_image is None:
            messagebox.showwarning("警告", "没有可保存的结果")
            return
        
        try:
            # 创建输出目录（如果不存在）
            output_dir = "output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 生成带时间戳的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pixel_art_{timestamp}.png"
            file_path = os.path.join(output_dir, filename)
            
            # 保存图像
            cv2.imwrite(file_path, self.result_image)
            
            # 更新输出路径
            self.output_image_path = file_path
            
            print(f"图像已自动保存到: {file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"自动保存图像时出错: {str(e)}")
    
    def open_result_file(self):
        """打开结果文件"""
        if self.output_image_path and os.path.exists(self.output_image_path):
            try:
                # 使用系统默认方式打开文件
                webbrowser.open(f'file://{os.path.abspath(self.output_image_path)}')
            except Exception as e:
                messagebox.showerror("错误", f"无法打开文件: {str(e)}")
        else:
            messagebox.showwarning("警告", "结果文件不存在")


if __name__ == "__main__":
    root = tk.Tk()
    app = PixelArtApp(root)
    root.mainloop()