from exr import *
from PIL import Image
import torch


eps = 0.00316


def preprocess_diffuse(diffuse, albedo):
    return diffuse / (albedo + eps)


def preprocess_specular(specular):
    return np.log(specular + 1)


class Data:
    def __init__(self, input_path, GT_path):

        data_list = sorted(os.listdir(path=input_path))

        for data in data_list:

            patch_size = 80

            read_input = read_all(input_path + data)
            read_GT = read_all(GT_path + data.replace('0128', '8192'))

            read_input['diffuse'] = preprocess_diffuse(
                read_input['diffuse'], read_input['albedo'])
            read_input['specular'] = preprocess_specular(
                read_input['specular'])

            for i in range(int(read_input['diffuse'].shape[1]/patch_size)):

                for j in range(int(read_input['diffuse'].shape[0]/patch_size)):

                    sliced_train = {}
                    sliced_input = {}
                    sliced_GT = {}

                    for key in read_input.keys():

                        sliced_input[key] = read_input[key][80 *
                                                            j:80*(j+1), 80*i:80*(i+1), :]
                        sliced_GT[key] = read_GT[key][80 *
                                                      j:80*(j+1), 80*i:80*(i+1), :]

                    sliced_train['input'] = sliced_input
                    sliced_train['GT'] = sliced_GT
                    torch.save(
                        sliced_train, '../data/sample_KPCN/sample_'+str(j+1)+'_'+str(i+1)+'.pt')


if __name__ == '__main__':

    input_path = '../data/example/'
    GT_path = '../data/example_GT/'
    data = Data(input_path, GT_path)
