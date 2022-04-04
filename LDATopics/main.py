import ctypes
import sys

class Utils:

    @staticmethod
    def loadLDAso(sopath):
        so = ctypes.CDLL(sopath)
        return so

class parserArg:

    def __init__(self,argv):
        self.argList = []
        length = len(argv)
        for i in range(1,length):
            print(argv[i])
            self.addArg(argv[i])

    def addArg(self,name):
        namePointer = ctypes.create_string_buffer(name)
        print("name",namePointer)
        self.argList.append(namePointer)   

if __name__ == '__main__':
    print("hello")
    so = Utils().loadLDAso("./gibbs-lda-master/src/liblda.so")
    print(sys.argv,len(sys.argv))
    arg = parserArg(sys.argv)
    # arg.addArg("-est")
    # arg.addArg("-alpha")
    # arg.addArg("0.5")
    # arg.addArg("-beta")
    # arg.addArg("0.1")
    # arg.addArg("-ntopics")
    # arg.addArg("100")
    # arg.addArg("-niters")
    # arg.addArg("1000")
    # arg.addArg("-savestep")
    # arg.addArg("100")
    # arg.addArg("-twords")
    # arg.addArg("20")
    # arg.addArg("-dfile")
    # arg.addArg("/home/yao/LDATopics/Data/trndocs.dat")
    lenArg = len(arg.argList)
    pointer = (ctypes.c_char_p*lenArg)(*(map(ctypes.addressof,arg.argList)))
    so.main(lenArg,pointer)