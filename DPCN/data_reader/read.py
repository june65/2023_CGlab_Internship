from exr import *


class Data:
    def __init__(self, train_path, test_path):

        data_list = sorted(os.listdir(path=train_path))
        
        for data in data_list:
            read_data = read_all(train_path + data)
            print(read_data)


if __name__ == '__main__':

    train_path = './data/'
    test_path = ''
    data = Data(train_path, test_path)
