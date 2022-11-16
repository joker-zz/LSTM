models里面保存的是双层LSTM模型的参数文件





实现的代码都在main.py中


TextLSTM类为手写的LSTM层，重写了构造函数与前向传递，详细的结构解析见报告


MyModuleDoubleLSTM类为双层LSTM模型，可以直接用于训练


MyModuleLSTM类为普通LSTM模型，同样也能直接用于训练



![N T%SA~~}8Z O3RTI C T65](https://user-images.githubusercontent.com/74516126/202191802-4b10593f-74fd-48cc-a8b6-798c594910e3.png)

直接修改train()函数中的该参数即可切换训练模型
