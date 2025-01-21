from abc import ABC, abstractmethod
import cv2
import numpy as np

class StackStrategy(ABC):
    @abstractmethod
    def stack(self, image_paths, batch_size=2):
        pass
    
    def _get_target_size(self, image_paths):
        """确定目标尺寸（使用第一张图片的尺寸）"""
        if not image_paths:
            return None
        first_img = cv2.imread(image_paths[0])
        if first_img is not None:
            return first_img.shape[:2]
        return None

    def _align_images(self, images, target_size):
        """统一图像尺寸"""
        if not images or target_size is None:
            return []
        h, w = target_size
        
        aligned = []
        for img in images:
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            aligned.append(img.astype(np.float32))
        return aligned
    
    def _read_batch(self, image_paths, start_idx, batch_size):
        """读取一批图像"""
        batch_paths = image_paths[start_idx:start_idx + batch_size]
        batch_images = []
        
        for path in batch_paths:
            img = cv2.imread(path)
            if img is not None:
                batch_images.append(img)
                
        return batch_images
    
    def _process_in_batches(self, image_paths, batch_size, process_func):
        """通用批处理方法"""
        if not image_paths:
            return None
            
        # 获取目标尺寸
        target_size = self._get_target_size(image_paths)
        if target_size is None:
            return None
            
        result = None
        total_count = len(image_paths)
        processed_count = 0
        
        for i in range(0, total_count, batch_size):
            # 读取批次图像
            batch_images = self._read_batch(image_paths, i, batch_size)
            
            if batch_images:
                # 对齐图像到目标尺寸
                aligned_batch = self._align_images(batch_images, target_size)
                # 处理当前批次
                batch_result = process_func(aligned_batch)
                
                # 更新结果
                if result is None:
                    result = batch_result
                else:
                    result = self._combine_results(result, batch_result)
                
                processed_count += len(batch_images)
                yield processed_count / total_count * 100
                
                # 释放内存
                del batch_images
                del aligned_batch
        
        if result is not None:
            final_result = self._finalize_result(result, processed_count)
            yield final_result
    
    def _combine_results(self, current_result, new_result):
        """合并两个批次的结果"""
        return current_result
    
    def _finalize_result(self, result, total_count):
        """完成最终结果处理"""
        return result

    def _tonemap(self, hdr_image):
        """
        使用Tonemapping算法将堆栈后的图像映射到普通显示设备上。
        :param hdr_image: 输入的HDR图像
        :return: 经过tonemap处理后的图像
        """
        tonemap = cv2.createTonemapReinhard(2.2, 0, 0, 0)
        ldr = tonemap.process(hdr_image)
        ldr = np.clip(ldr * 255, 0, 255).astype('uint8')
        
        return ldr

    def _process_images_in_batches(self, image_paths, batch_size=2):

        """分批处理图像"""
        total_images = len(image_paths)
        accumulated_sum = None
        processed_count = 0

        for i in range(0, total_images, batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            # 读取当前批次的图像
            for path in batch_paths:
                img = cv2.imread(path)
                if img is not None:
                    batch_images.append(img)
            
            # 处理当前批次
            if batch_images:
                aligned_batch = self._align_images(batch_images, self._get_target_size(image_paths))
                batch_sum = np.sum(aligned_batch, axis=0)
                
                if accumulated_sum is None:
                    accumulated_sum = batch_sum
                else:
                    accumulated_sum += batch_sum
                
                processed_count += len(batch_images)
                
                # 释放内存
                del batch_images
                del aligned_batch
            
            # 报告进度
            yield processed_count / total_images * 100

        # 计算最终结果
        if accumulated_sum is not None and processed_count > 0:
            result = accumulated_sum / processed_count
            return np.clip(result, 0, 255).astype(np.uint8)
        return None

class MeanStackStrategy(StackStrategy):
    def stack(self, image_paths, batch_size=2):
        """均值堆叠"""
        def process_batch(batch):
            return np.sum(batch, axis=0)
            
        return self._process_in_batches(image_paths, batch_size, process_batch)
    
    def _combine_results(self, current_result, new_result):
        return current_result + new_result
    
    def _finalize_result(self, result, total_count):
        return np.clip(result / total_count, 0, 255).astype(np.uint8)

class MedianStackStrategy(StackStrategy):
    def stack(self, image_paths, batch_size=2):
        """中值堆叠"""
        def process_batch(batch):
            return batch
            
        def combine_results(current, new):
            if isinstance(current, list):
                current.extend(new)
            else:
                current = list(current)
                current.extend(new)
            return current
            
        def finalize(result, _):
            stacked = np.median(result, axis=0)
            return np.clip(stacked, 0, 255).astype(np.uint8)
            
        self._combine_results = combine_results
        self._finalize_result = finalize
        
        return self._process_in_batches(image_paths, batch_size, process_batch)

class MaxStackStrategy(StackStrategy):
    def stack(self, image_paths, batch_size=2):
        """最大值堆叠"""
        def process_batch(batch):
            return np.max(batch, axis=0)
            
        def combine_results(current, new):
            return np.maximum(current, new)
            
        def finalize(result, _):
            return np.clip(result, 0, 255).astype(np.uint8)
            
        self._combine_results = combine_results
        self._finalize_result = finalize
        
        return self._process_in_batches(image_paths, batch_size, process_batch)

class DenoiseStackStrategy(StackStrategy):
    def stack(self, image_paths, batch_size=2):
        """去噪堆叠"""
        def process_batch(batch):
            denoised_sum = None
            for img in batch:
                denoised = cv2.fastNlMeansDenoisingColored(
                    np.uint8(img), None, 10, 10, 7, 21
                )
                if denoised_sum is None:
                    denoised_sum = denoised.astype(np.float32)
                else:
                    denoised_sum += denoised.astype(np.float32)
            return denoised_sum
            
        return self._process_in_batches(image_paths, batch_size, process_batch)
    
    def _combine_results(self, current_result, new_result):
        return current_result + new_result
    
    def _finalize_result(self, result, total_count):
        return np.clip(result / total_count, 0, 255).astype(np.uint8)

class HDRStackStrategy(StackStrategy):
    def stack(self, images, batch_size=2):
        """HDR堆叠"""
        if not images:
            return None
        
        aligned = self._align_images(images, self._get_target_size(images))
        # 计算每张图像的平均亮度    
        brightness_values = [self.calculate_average_brightness(img) for img in aligned]
        # 根据亮度排序
        sorted_images = [img for _, img in sorted(zip(brightness_values, images))]
        # 创建HDR图像
        hdr_image = self.create_hdr(sorted_images)
        # 将HDR图像映射到普通显示设备
        return self._tonemap(hdr_image)
        
    def calculate_average_brightness(self, image):
        """
        计算图像的平均亮度。
        :param image: 输入的图像
        :return: 图像的平均亮度值
        """
        # 转换为灰度图像
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 计算并返回灰度图像的平均值
        return np.mean(gray_image)
    
    def create_hdr(self, images):
        """
        使用加权平均的方法来合成 HDR 图像。
        :param images: 一个包含多张曝光不同的图像的列表
        :return: 合成后的 HDR 图像
        """
        # 将图像转换为浮动数据类型，以便进行加权操作
        images_floats = [np.float32(img) for img in images]

        # 使用OpenCV的createMergeDebevec()来生成HDR图像
        merge_debevec = cv2.createMergeDebevec()

        # 合成HDR图像
        hdr = merge_debevec.process(images_floats)

        # 返回合成后的HDR图像
        return hdr

