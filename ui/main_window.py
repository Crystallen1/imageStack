from PyQt6.QtWidgets import (
    QMainWindow, 
    QWidget, 
    QVBoxLayout, 
    QHBoxLayout,
    QPushButton,
    QListWidget,
    QLabel,
    QFileDialog,
    QProgressBar,
    QMessageBox,
    QSpinBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from core.image_stack import ImageStack
import cv2
import numpy as np
from core.stack_strategies import (
    MeanStackStrategy, 
    MedianStackStrategy, 
    MaxStackStrategy,
    HDRStackStrategy,
    DenoiseStackStrategy
)
from PyQt6.QtWidgets import QApplication

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像堆栈工具")
        self.setMinimumSize(800, 600)
        
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧面板 - 图像列表
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        self.image_list = QListWidget()
        add_image_btn = QPushButton("添加图像")
        remove_image_btn = QPushButton("移除图像")
        
        left_layout.addWidget(QLabel("图像列表"))
        left_layout.addWidget(self.image_list)
        left_layout.addWidget(add_image_btn)
        left_layout.addWidget(remove_image_btn)
        
        # 右侧面板 - 预览和控制
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 创建预览区域的分割布局
        preview_widget = QWidget()
        preview_layout = QHBoxLayout(preview_widget)
        
        # 原始图像预览
        original_preview_group = QWidget()
        original_preview_layout = QVBoxLayout(original_preview_group)
        original_preview_layout.addWidget(QLabel("原始图像"))
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(300, 300)
        self.preview_label.setStyleSheet("border: 1px solid black")
        original_preview_layout.addWidget(self.preview_label)
        
        # 结果图像预览
        result_preview_group = QWidget()
        result_preview_layout = QVBoxLayout(result_preview_group)
        result_preview_layout.addWidget(QLabel("处理结果"))
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setMinimumSize(300, 300)
        self.result_label.setStyleSheet("border: 1px solid black")
        result_preview_layout.addWidget(self.result_label)
        
        # 添加两个预览区域到预览布局
        preview_layout.addWidget(original_preview_group)
        preview_layout.addWidget(result_preview_group)
        
        # 将预览区域添加到右侧布局
        right_layout.addWidget(preview_widget)
        
        # 堆叠方法选择按钮
        stack_methods_group = QWidget()
        stack_methods_layout = QHBoxLayout(stack_methods_group)
        
        
        mean_stack_btn = QPushButton("均值堆叠")
        median_stack_btn = QPushButton("中值堆叠")
        max_stack_btn = QPushButton("最大值堆叠")
        hdr_stack_btn = QPushButton("HDR堆叠")
        denoise_stack_btn = QPushButton("去噪堆叠")
        
        stack_methods_layout.addWidget(mean_stack_btn)
        stack_methods_layout.addWidget(median_stack_btn)
        stack_methods_layout.addWidget(max_stack_btn)
        stack_methods_layout.addWidget(hdr_stack_btn)
        stack_methods_layout.addWidget(denoise_stack_btn)
        
        # 添加batch size控制
        batch_control = QWidget()
        batch_layout = QHBoxLayout(batch_control)
        
        batch_layout.addWidget(QLabel("批处理大小:"))
        self.batch_size_input = QSpinBox()
        self.batch_size_input.setMinimum(1)
        self.batch_size_input.setMaximum(100)
        self.batch_size_input.setValue(2)  # 默认值
        self.batch_size_input.setToolTip("设置每批处理的图片数量，数值越大内存占用越多")
        batch_layout.addWidget(self.batch_size_input)
        
        # 添加进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # 调整布局顺序
        right_layout.addWidget(stack_methods_group)
        right_layout.addWidget(batch_control)
        right_layout.addWidget(self.progress_bar)
        
        # 添加左右面板到主布局
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
        # 连接信号
        add_image_btn.clicked.connect(self.add_image)
        remove_image_btn.clicked.connect(self.remove_image)
        mean_stack_btn.clicked.connect(lambda: self.start_stack(MeanStackStrategy()))
        median_stack_btn.clicked.connect(lambda: self.start_stack(MedianStackStrategy()))
        max_stack_btn.clicked.connect(lambda: self.start_stack(MaxStackStrategy()))
        hdr_stack_btn.clicked.connect(lambda: self.start_stack(HDRStackStrategy()))
        denoise_stack_btn.clicked.connect(lambda: self.start_stack(DenoiseStackStrategy()))
        
        self.image_stack = ImageStack()
    
    def add_image(self):
        """添加图像到列表"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择图像文件",
            "",
            "图像文件 (*.jpg *.jpeg *.png *.tif *.tiff)"
        )
        
        for path in file_paths:
            if self.image_stack.add_image(path):
                self.image_list.addItem(path)
                # 预览最新添加的图片
                self.preview_image(path)
    
    def preview_image(self, image_path):
        """在预览区域显示图片"""
        img = cv2.imread(image_path)
        if img is not None:
            # 保持纵横比缩放图片以适应预览区域
            h, w = img.shape[:2]
            preview_size = self.preview_label.size()
            scale = min(preview_size.width() / w, preview_size.height() / h)
            new_size = (int(w * scale), int(h * scale))
            resized = cv2.resize(img, new_size)
            pixmap = self.convert_cv_to_pixmap(resized)
            self.preview_label.setPixmap(pixmap)
            # 清空结果显示
            self.result_label.clear()
            self.result_label.setText("等待处理...")
    
    def remove_image(self):
        """从列表中移除选中的图像"""
        current_row = self.image_list.currentRow()
        if current_row >= 0:
            self.image_list.takeItem(current_row)
            self.image_stack.remove_image(current_row)
    
    def start_stack(self, strategy):
        """开始堆叠处理"""
        if not self.image_stack.image_paths:
            QMessageBox.warning(self, "警告", "请先添加图像")
            return
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.result_label.setText("处理中...")
        
        try:
            self.image_stack.strategy = strategy
            result = None
            
            batch_size = self.batch_size_input.value()
            print(f"开始处理，批次大小: {batch_size}")  # 调试信息
            
            generator = self.image_stack.strategy.stack(
                self.image_stack.image_paths, 
                batch_size=batch_size
            )
            
            for progress_or_result in generator:
                print(f"收到返回值: {type(progress_or_result)}")  # 调试信息
                if isinstance(progress_or_result, np.ndarray):
                    result = progress_or_result
                    print("获得最终结果")  # 调试信息
                    break
                else:
                    self.progress_bar.setValue(int(progress_or_result))
                    QApplication.processEvents()
            
            if result is not None:
                print(f"结果形状: {result.shape}")  # 调试信息
                self._display_and_save_result(result)
            else:
                print("未获得结果")  # 调试信息
                self.result_label.setText("处理失败")
            
        except Exception as e:
            print(f"发生错误: {str(e)}")  # 调试信息
            QMessageBox.critical(self, "错误", f"处理失败: {str(e)}")
            self.result_label.setText("处理失败")
        finally:
            self.progress_bar.setVisible(False)
    
    def _display_and_save_result(self, result):
        """显示和保存结果"""
        # 显示堆叠结果
        h, w = result.shape[:2]
        preview_size = self.result_label.size()
        scale = min(preview_size.width() / w, preview_size.height() / h)
        new_size = (int(w * scale), int(h * scale))
        resized = cv2.resize(result, new_size)
        pixmap = self.convert_cv_to_pixmap(resized)
        self.result_label.setPixmap(pixmap)
        
        # 保存结果对话框
        response = QMessageBox.question(
            self,
            "保存结果",
            "处理完成，是否保存结果？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if response == QMessageBox.StandardButton.Yes:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "保存堆叠结果",
                "",
                "图像文件 (*.jpg *.png *.tiff)"
            )
            if file_path:
                cv2.imwrite(file_path, result)
                QMessageBox.information(self, "成功", "结果已保存")
    
    def convert_cv_to_pixmap(self, cv_img):
        """将OpenCV图像转换为QPixmap"""
        height, width, channel = cv_img.shape
        bytes_per_line = 3 * width
        # OpenCV使用BGR顺序，需要转换为RGB
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qt_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qt_image) 