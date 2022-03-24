import numpy as np


class NGram:
    def __init__(self, n_grams=2, smooth_k=1):
        self.n_grams = n_grams
        self.gram_map = {}
        self.term_map ={}
        self.smooth_k = smooth_k
        self.v = 0

    def generate_for_sentence(self, sentence):
        """

        :param sentence: type:list data for sentence
        :param n: n-gram,default:2
        :return: None
        """

        length = len(sentence)

        for i in range(length):
            # 构建 term gram统计表
            if i<length + 1 - self.n_grams:

                word = sentence[i:i + self.n_grams - 1]
                if word not in self.term_map.keys():
                    self.term_map[word] = 1
                else:
                    self.term_map[word] = self.term_map[word] + 1

                gram = sentence[i:i+self.n_grams]
                if gram not in self.gram_map.keys():
                    self.gram_map[gram] = 1
                else:
                    self.gram_map[gram] = self.gram_map[gram] + 1

    def generate_for_doc(self, documents):
        for document in documents:
            self.generate_for_sentence(document)
        self.v = len(self.term_map)

    def cal_posterior_prob(self, wordA, wordCombine):
        word_freq = self.smooth_k * self.v
        gram_freq = self.smooth_k

        if wordA in self.term_map.keys():
            word_freq += self.term_map[wordA]

        if wordCombine in self.gram_map.keys():
            gram_freq += self.gram_map[wordCombine]

        return gram_freq/word_freq + 1e-5


class NGramModel:
    def __init__(self, data, label, n_grams=2, smooth_k=1,islog = True):
        """

        :param data:  type:list, two dims: input data,each row is a document
        :param label: type:list label, responding to label info
        :param n_grams: type:int
        :param smooth_k: type:int
        """
        self.data = data
        self.label = label
        self.n_grams = n_grams
        self.smooth_k = smooth_k
        self.NGram_list = []
        self.uniqueLabel = np.unique(label).tolist()
        for i in self.uniqueLabel:
            tGram = NGram(self.n_grams, self.smooth_k)
            tGram.generate_for_doc(np.array(data)[np.array(label) == i])
            if islog:
                print("总数据长度:", len(data), "label为",i ,"长度",len(np.array(data)[np.array(label) == i]))
                print("词汇表长度:",tGram.v)
            self.NGram_list.append(tGram)

    def predict(self, documents):
        """
        :param documents: type:list
        :return label: type:list label info
        """
        documentProb = []
        documentsProb = []
        for document in documents:
            for label in self.uniqueLabel:
                probL = self.predictLog(document, label)
                documentProb.append(probL)
            documentsProb.append(documentProb)
            documentProb = []

        return np.array(self.uniqueLabel)[np.argmax(documentsProb, axis=1)].tolist()

    def predictLog(self, document, label):
        documentLen = len(document)
        logProb = 0.0
        for i in range(documentLen - self.n_grams + 1):
            prob = self.NGram_list[self.uniqueLabel.index(label)].cal_posterior_prob(document[i:i+self.n_grams-1], document[i:i+self.n_grams])
            logProb = logProb + np.log(prob)
        return logProb