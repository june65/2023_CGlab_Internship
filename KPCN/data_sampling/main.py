from exr import *
from PIL import Image
import torch


eps = 0.00316


def preprocess_diffuse(diffuse, albedo):
    return diffuse / (albedo + eps)


def preprocess_specular(specular):
    return np.log(specular + 1)


def preprocess_diff_var(variance, albedo):
    return variance / (albedo + eps)**2


def preprocess_spec_var(variance, specular):
    return variance / (specular+1e-5)**2


def gradients(data):
    h, w, c = data.shape
    dX = data[:, 1:, :] - data[:, :w - 1, :]
    dY = data[1:, :, :] - data[:h - 1, :, :]
    # padding with zeros
    dX = np.concatenate((np.zeros([h, 1, c], dtype=np.float32), dX), axis=1)
    dY = np.concatenate((np.zeros([1, w, c], dtype=np.float32), dY), axis=0)

    return np.concatenate((dX, dY), axis=2)


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

            read_input['diffuseVariance'] = preprocess_diff_var(
                read_input['diffuseVariance'], read_input['albedo'])

            read_input['specularVariance'] = preprocess_spec_var(
                read_input['specularVariance'], read_input['specular'])

            read_input['gradAlbedo'] = gradients(
                read_input['albedo'][:, :, :3].copy())

            read_input['gradNormal'] = gradients(
                read_input['normal'][:, :, :3].copy())

            read_input['gradDepth'] = gradients(
                read_input['depth'][:, :, :3].copy())

            read_input['gradDiffuse'] = gradients(
                read_input['diffuse'][:, :, :3].copy())

            read_input['gradSpecular'] = gradients(
                read_input['specular'][:, :, :3].copy())

            for i in range(int(read_input['diffuse'].shape[1]/patch_size)):

                for j in range(int(read_input['diffuse'].shape[0]/patch_size)):

                    sliced_train = {}

                    for key in read_input.keys():

                        sliced_train[key] = read_input[key][80 *
                                                            j:80*(j+1), 80*i:80*(i+1), :]

                    sliced_train['diffuse_GT'] = read_GT['diffuse'][80 *
                                                                    j:80*(j+1), 80*i:80*(i+1), :]

                    sliced_train['specular_GT'] = read_GT['specular'][80 *
                                                                      j:80*(j+1), 80*i:80*(i+1), :]

                    X_diff = np.concatenate((sliced_train['diffuse'],
                                             sliced_train['gradDiffuse'],
                                             sliced_train['diffuseVariance'],
                                             sliced_train['gradNormal'],
                                             sliced_train['normalVariance'],
                                             sliced_train['gradDepth'],
                                             sliced_train['depthVariance'],
                                             sliced_train['gradAlbedo'],
                                             sliced_train['albedoVariance']
                                             ), axis=2)

                    X_spec = np.concatenate((sliced_train['specular'],
                                             sliced_train['gradSpecular'],
                                             sliced_train['specularVariance'],
                                             sliced_train['gradNormal'],
                                             sliced_train['normalVariance'],
                                             sliced_train['gradDepth'],
                                             sliced_train['depthVariance'],
                                             sliced_train['gradAlbedo'],
                                             sliced_train['albedoVariance']
                                             ), axis=2)

                    sliced_train['X_diff'] = X_diff
                    sliced_train['X_spec'] = X_spec

                    torch.save(
                        sliced_train, '../data/sample_KPCN/sample_'+str(j+1)+'_'+str(i+1)+'.pt')


if __name__ == '__main__':

    input_path = '../data/example/'
    GT_path = '../data/example_GT/'
    data = Data(input_path, GT_path)
