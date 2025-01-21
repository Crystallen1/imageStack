import cv2
import numpy as np
from core.stack_strategies import MeanStackStrategy

class ImageStack:
    def __init__(self):
        self.image_paths = []  # 存储路径而不是图片
        self._strategy = MeanStackStrategy()  # 默认使用均值堆叠
    
    @property
    def strategy(self):
        return self._strategy
    
    @strategy.setter
    def strategy(self, strategy):
        self._strategy = strategy
    
    def add_image(self, image_path):
        """添加图像路径到堆叠列表"""
        try:
            # 只验证图片是否可读，不保存在内存中
            img = cv2.imread(image_path)
            if img is not None:
                self.image_paths.append(image_path)
                return True
            return False
        except Exception as e:
            print(f"添加图像失败: {str(e)}")
            return False
    
    def remove_image(self, index):
        """移除指定索引的图像路径"""
        if 0 <= index < len(self.image_paths):
            self.image_paths.pop(index)
    
    def stack_images(self):
        """执行图像堆叠操作"""
        return self._strategy.stack(self.image_paths) 