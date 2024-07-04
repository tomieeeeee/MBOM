 
 
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
#做平滑处理，我觉得应该可以理解为减弱毛刺，，吧  能够更好地看数据走向
def tensorboard_smoothing(x,smooth=0.99):
    x = x.copy()
    weight = smooth
    for i in range(1,len(x)):
        x[i] = (x[i-1] * weight + x[i]) / (weight + 1)
        weight = (weight + 1) * smooth
    return x
 
if __name__ == '__main__':
    
    fig, ax1 = plt.subplots(1, 1)    # a figure with a 1x1 grid of Axes
    
    #设置上方和右方无框
    ax1.spines['top'].set_visible(False)                   # 不显示图表框的上边框
    ax1.spines['right'].set_visible(False)  
 
    len_mean = pd.read_csv("D:/document/MBAM/data/show_data/60ava.csv")
    my_font = font_manager.FontProperties(family='SimHei', size=18)
 
    #设置折线颜色，折线标签
    #使用平滑处理
    ax1.plot(len_mean['Step'], tensorboard_smoothing(len_mean['Value'], smooth=0.8), color="#005BAC",label='ours')
    ax1.plot(len_mean['Step'], tensorboard_smoothing(len_mean['Value1'], smooth=0.8), color="#C00000",label='PPO')
    #不使用平滑处理
    #ax1.plot(len_mean['Step'], len_mean['Value'], color="#C00000",label='set1')
    #ax1.plot(len_mean['Step'], len_mean['Value1'], color="#005BAC",label='set2')#FCAF98
    #ax1.plot(len_mean['Step'], len_mean['Value2'], color="#FFC000",label='set3')#D67545
 
 
    #s设置标签位置，lower upper left right，上下和左右组合
    plt.legend(loc = 'lower right')
    y=range(40,140,20)
    label=[40,60,80,100,120,140]
    plt.yticks(y,label)
   
    #ax1.set_xlabel("epoch")
    #ax1.set_ylabel("score")
    plt.xlabel("对抗轮次", fontproperties=my_font)
    plt.ylabel("每轮得分", fontproperties=my_font)
    #标题ax1.set_title("test Accuracy")
    plt.show()
    #保存图片，也可以是其他格式，如pdf
    fig.savefig(fname='D:/document/MBAM/data/show_data/score'+'.png', format='png')
 