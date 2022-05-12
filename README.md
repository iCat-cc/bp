# BP算法简介
## 基于BP算法的多层感知器模型
采用BP算法的多层感知器是至今为止应用最广泛的神经网络，在多层感知器的应用中，以图3-15所示的单隐层网络的应用最为普遍。一般习惯将单隐层前馈网称为三层感知器，所谓三层包括了输入层、隐层和输出层。
![image](https://user-images.githubusercontent.com/61224939/167989236-fafa8f7a-e91c-49c5-8d88-69139d0a6ab9.png)
![image](https://user-images.githubusercontent.com/61224939/167989277-65c76142-a8c3-487c-ae03-0035c4986e1d.png)
算法最终结果采用梯度下降法，具体详细过程此处就省略了！
## BP算法的程序实现流程
![image](https://user-images.githubusercontent.com/61224939/167989335-7ef0968b-3bdc-4324-abb6-1eb81fb941ca.png)


## Python实现BP神经网络及其学习算法
这里为了运用算法，简要的举了一个例子（不需归一化或标准化的例子）

输入 X=-1:0.1:1;
输出 D=.....（具体查看代码里面的数据）

为了便于查看结果我们输出把结果绘制为图形，如下：

![image](https://user-images.githubusercontent.com/61224939/167989119-d0ef3211-27fc-4cea-afad-d7d14ef16221.png)

其中黄线和蓝线代表着训练完成后的输出与输入
