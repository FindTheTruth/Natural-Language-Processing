

import matplotlib.pyplot as plt
plt.style.use('ggplot')
from DataUtils import *
from N_Gram import *
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    datapath = 'data'
    utils = DataUtils(datapath)
    whole_text, whole_label = utils.data_preprocessing()

    # 数据划分，80%训练，20%验证
    XTrain, XTest, YTrain, YTest = train_test_split(whole_text, whole_label, train_size=0.8, random_state=3)

    print("train len", len(XTrain), "test len", len(XTest))

    # 针对构造的N_gram对测试集进行概率预测，预测邮件究竟属于常规邮件的概率更大还是垃圾邮件的概率更大。
    n_grams_list = [1, 2, 3, 4, 5, 6, 7]
    smooth_k_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    for n_gram in n_grams_list:
        for smooth_k in smooth_k_list:
            model = NGramModel(XTrain, YTrain, n_grams=n_gram, smooth_k=smooth_k,islog=False)
            testRes = model.predict(XTest)
            print("grams:", n_gram, "smooth:", smooth_k, "acc:", sum(np.array(testRes) == np.array(YTest)) / (len(testRes) + 0.0))

