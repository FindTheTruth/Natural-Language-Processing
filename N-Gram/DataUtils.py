import codecs
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

class DataUtils:
    def __init__(self,path):
        self.path = path

    # 读取邮件信息
    def read_dataset(self):
        with open(self.path + "/ham_5000.utf8", encoding='utf-8') as f:
            ham_txt_list = f.readlines()
        with open(self.path + "/spam_5000.utf8", encoding='utf-8') as f:
            spam_txt_list = f.readlines()
        stopwords_list = codecs.open(self.path + "/stopword.txt", 'r', 'utf8').read().split('\r\n')
        return ham_txt_list, spam_txt_list, stopwords_list

    # 去除停用词
    def wipe_stop_words(self, mails, stop_words):
        new_mails = []
        for mail in mails:
            seg_words = jieba.cut(mail)
            new_mail_list = []
            for words in seg_words:
                if words != '\n' and words not in stop_words:
                    new_mail_list.append(words)
            new_mail = " ".join(new_mail_list)

            new_mail = "<s> " + new_mail + " </s>"
            new_mails.append(new_mail)
        return new_mails

    def transformerTextToMatrix(self, texts):
        vectorizers = CountVectorizer(binary=False)
        vectorizers.fit(texts)

        # countVector 默认排序表是按照key的大小给出
        vector = vectorizers.transform(texts)
        result = pd.DataFrame(vector.toarray())
        result.columns = vectorizers.get_feature_names()
        return result

    def split_dataset(self,data, label):
        """送进去什么类型，返回的还是什么类型，dataFrame返回DataFrame,list返回list"""
        X_train, X_test, y_train, y_test = train_test_split(data, label, train_size=0.8, random_state=3)
        return X_train, X_test, y_train, y_test

    def data_preprocessing(self):
        hams, spams, stop_words = self.read_dataset()
        preprocess_hams = self.wipe_stop_words(hams, stop_words)
        preprocess_spams = self.wipe_stop_words(spams, stop_words)
        whole_text = preprocess_hams.copy()
        for spam in preprocess_spams:
            whole_text.append(spam)
        whole_label = [0] * len(preprocess_hams) + [1] * len(preprocess_spams)
        return whole_text, whole_label

    def extracted_count_feature(self,pdFeatures, count, islog=True):
        """
             去除特征出现次数较少的数据
        :param pd: type:Dataframe:包含feature
        :param count: type:int: feature出现的频率，大于该值才会作为用于训练特征
        :return:type:DataFrame with extracted_features
        """
        # Note:dataFrame默认求和
        sumFeatures = pd.DataFrame(pdFeatures.apply(sum, axis=0))
        print(sumFeatures.iloc[0],"--",sumFeatures.iloc[0,0])
        extractedFeature_cols = [sumFeatures.index[i] for i in range(len(pdFeatures.columns))
                                 if sumFeatures.iloc[i, 0] > count]

        pdExtractedFeature = pdFeatures[extractedFeature_cols]

        if islog:
            print(len(pdExtractedFeature.columns), len(pdFeatures.columns))
        return pdExtractedFeature

    def extracted_correlation_feature(self,pdFeatures, label, lower_cor=1e-2, islog=False):
        """
             去除和标签关联度较低的特征

        :param pdFeatures: type:Dataframe:包含feature
        :param label: type:List:包含label
        :param lower_cor: type:float: feature和label的关联程度
        :param islog: print log
        :return:type:DataFrame with extracted_features
        """
        selected_feature = []
        for i in pdFeatures.columns:
            x = np.array(pdFeatures[i])
            y = np.abs(np.corrcoef(x, np.array(label))[0][1])
            if islog:
                print("y cor with ", i, " ", y)
            if y > lower_cor:
                selected_feature.append(i)
        if islog:
            print("select features before", len(pdFeatures.columns))
            print("select features by corr", len(selected_feature))
        return pdFeatures[selected_feature]
