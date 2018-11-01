# 神经网络分类器模型
A structured classification code, and core codes are adapted from [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN) and [keras](https://github.com/keras-team/keras).


## 数据的组织方式
源码对数据集进行了比较好的组织，形成了具有通用性的数据组织结构。其中，数据集的基础类是utils.Dataset，其具体结构如下：

```java
def __init__(self, class_map=None):
    self._image_ids = []
    self.image_info = []
    # Background is always the first class
    self.class_info = [{"source": "", "id": 0, "name": "BG"}]
    self.source_class_ids = {}
```
从这个初始化函数可以看出，所有的数据都被组织成：图片索引的列表、有序图片的信息、所有的类等信息。

通过继承Dataset类，并重写方法来创建自己的数据集类：
```python
"""
The base class for dataset classes.
To use it, create a new class that 
adds functions specific to the dataset
you want to use. For example:
"""
class CatsAndDogsDataset(Dataset):
	def load_cats_and_dogs(self):
```
