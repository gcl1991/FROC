# coding:utf-8
import numpy as np
from froc import computeFROC, plotFROC
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    np.random.seed(1)    
    
    #parameters
    nbr_img = 20  # 照片数量
    size_img = [10,10]  # 照片尺寸
    save_path = 'FROC_example1.pdf'
    nbr_of_thresholds = 40  # 要计算以绘制FROC的阈值数
    range_threshold = [.1,.5]  # 开始和结束用于绘制FROC的阈值范围
    allowedDistance = 2  # 检测半径
    
    #create artificial data
    ground_truth = np.random.randint(2, size=[nbr_img]+size_img)  # [0,2)
    proba_map = np.random.randint(100, size=[nbr_img]+size_img)*1./100  # [0,99)
    # plt.imshow(ground_truth[0], 'gray')
    # plt.savefig(save_path)

    # compute FROC
    sensitivity_list, FPavg_list, _ = computeFROC(proba_map,ground_truth, allowedDistance, nbr_of_thresholds, range_threshold)
    print 'computed FROC'
    print np.array(sensitivity_list).shape
    print np.array(FPavg_list).shape
    # #plot FROC
    plotFROC(FPavg_list,sensitivity_list,save_path)
    print 'plotted FROC'