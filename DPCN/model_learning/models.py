from exr import *
import os


if __name__ == '__main__':

    image_path = '../data/example/'
    model_path = './model/'
    model_version = open('./model_learning/model_version.txt', 'r').read()

    data_list = sorted(os.listdir(path=image_path))

    # model learning

    open('./model_learning/model_version.txt',
         'w').write(str(int(model_version)+1))

    model_text = open('./model_learning/model_info.txt', 'r').read()
    model_text = model_text + '\nversion_{} : train_AC:{}% , test_AC:{}% , change:{}'.format(
        model_version, model_version, model_version, model_version)
    open('./model_learning/model_info.txt', 'w').write(model_text)
