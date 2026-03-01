import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class TensorboardLogger:
    def __init__(self, log_dir='./logs'):
        # 创建日志目录
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 创建唯一的日志子目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = os.path.join(log_dir, f'train_{timestamp}')
        
        # 创建SummaryWriter
        self.writer = SummaryWriter(self.log_dir)
        print(f"TensorBoard日志已保存到: {self.log_dir}")
    
    def add_scalar(self, tag, scalar_value, global_step=None):
        """添加标量数据"""
        self.writer.add_scalar(tag, scalar_value, global_step)
    
    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None):
        """添加多个标量数据"""
        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step)
    
    def add_image(self, tag, img_tensor, global_step=None, dataformats='CHW'):
        """添加图像数据"""
        self.writer.add_image(tag, img_tensor, global_step, dataformats=dataformats)
    
    def add_images(self, tag, img_tensor, global_step=None, dataformats='NCHW'):
        """添加多张图像数据"""
        self.writer.add_images(tag, img_tensor, global_step, dataformats=dataformats)
    
    def add_graph(self, model, input_to_model=None):
        """添加模型结构"""
        self.writer.add_graph(model, input_to_model)
    
    def add_histogram(self, tag, values, global_step=None, bins='tensorflow'):
        """添加直方图数据"""
        self.writer.add_histogram(tag, values, global_step, bins)
    
    def add_confusion_matrix(self, tag, y_true, y_pred, global_step=None, class_names=None):
        """添加混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        if class_names is not None:
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=class_names,
                   yticklabels=class_names,
                   title='Confusion matrix',
                   ylabel='True label',
                   xlabel='Predicted label')
            
            # 旋转x轴标签
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")
        
        # 添加文本注释
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        self.writer.add_figure(tag, fig, global_step)
    
    def close(self):
        """关闭SummaryWriter"""
        self.writer.close()
    
    def get_log_dir(self):
        """获取日志目录"""
        return self.log_dir