from exr import *


class Data:
    def __init__(self, train_path, test_path):

        data_list = sorted(os.listdir(path=train_path))
        print(data_list)
        for data in data_list:
            read_all(train_path + data)


if __name__ == '__main__':

    train_path = './data/'
    test_path = ''
    data = Data(train_path, test_path)
