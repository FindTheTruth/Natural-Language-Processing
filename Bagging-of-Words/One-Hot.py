# one-hot demo:
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()

fit_data = [
    ['male', 0, 3],
    ['female', 0, 1],
    ['male', 1, 2]
]
enc.fit(fit_data)
# 稀疏表示和数字表示
sparedata = enc.transform([['male', 1, 3]])
arraydata = sparedata.toarray()
print("sparse:", sparedata)
print("array:",arraydata)
print(enc.categories_)
print("----------------end----------------------")
# # 除了自主决定外，也可以手动指定one-hot的类别数
# 如果fit中不存在data里出现的数字，可以利用参数 handle_unknown='ignore'忽略报错信息编码时，不编写
# inverse_transform遇到未知的类型时，转换结果为None
pointEnc = OneHotEncoder(categories=[['male', 'female'], [0, 1], [0, 1, 2, 3]], handle_unknown='ignore')
pointEnc.fit(fit_data)
# 稀疏表示和数字表示
sparedata = pointEnc.transform([['male', 1, 0]])
arraydata = sparedata.toarray()
print("sparse:", sparedata)
print("array:", arraydata)
print(pointEnc.inverse_transform([[0., 1., 0., 1., 0., 0., 1.,0], [1., 0., 0., 1., 0., 1., 0,0.]]))