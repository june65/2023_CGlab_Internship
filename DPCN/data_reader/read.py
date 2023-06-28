from exr import *
from PIL import Image


class Data:
    def __init__(self, train_path, test_path):

        data_list = sorted(os.listdir(path=train_path))

        for data in data_list:
            read_data = read_all(train_path + data)

            for feature in read_data.keys():

                # print(read_data[feature])
                if not os.path.exists(train_path + feature):

                    os.makedirs(train_path + feature)

                new_image = read_data[feature].astype(np.uint8)

                if len(new_image.shape) == 3 and new_image.shape[2] == 1:

                    new_image = read_data[feature]

                    image = Image.fromarray(new_image.squeeze(), mode="L")

                    image.save(train_path+feature+'/' +
                               data.replace('exr', 'png'))

                if len(new_image.shape) == 3 and new_image.shape[2] == 3:

                    new_image = read_data[feature].astype(np.uint8)

                    image = Image.fromarray(new_image)

                    image.save(train_path+feature+'/' +
                               data.replace('exr', 'png'))


if __name__ == '__main__':

    train_path = './data/'
    test_path = './data/'
    data = Data(train_path, test_path)
