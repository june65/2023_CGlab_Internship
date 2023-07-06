from exr import *
from PIL import Image
import torch


eps = 0.00316


def preprocess_diffuse(diffuse, albedo):
    return diffuse / (albedo + eps)


def preprocess_specular(specular):
    specular = np.where(specular < 0, 0, specular)
    return np.log(specular + 1) + 1e-6


def preprocess_diff_var(variance, albedo):
    return variance / (albedo + eps)**2


def preprocess_spec_var(variance, specular):
    return variance / (specular+1e-5)**2


def preprocess_albe_var(variance):
    variance = np.where(np.isnan(variance) | (
        variance == -np.nan), 0, variance)
    return variance


def preprocess_depth_var(variance):
    variance = np.where(np.isnan(variance) | (
        variance == -np.nan), 0, variance)
    return variance


def preprocess_norm_var(variance):
    variance = np.where(np.isnan(variance) | (
        variance == -np.nan), 0, variance)
    return variance


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

            read_input['albedoVariance'] = preprocess_albe_var(
                read_input['albedoVariance'])

            read_input['depthVariance'] = preprocess_depth_var(
                read_input['depthVariance'])

            read_input['normalVariance'] = preprocess_norm_var(
                read_input['normalVariance'])

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

            sliced_train = {}

            for key in read_input.keys():

                sliced_train[key] = read_input[key]

            sliced_train['diffuse_GT'] = read_GT['diffuse']

            sliced_train['specular_GT'] = read_GT['specular']

            sliced_train['finalInput'] = read_input['default']

            sliced_train['finalGt'] = read_GT['default']

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
                sliced_train, '../data/test_data/test_torch/'+data.replace('.exr', '.pt'))


if __name__ == '__main__':

    input_path = '../data/test_data/test_img/example/'
    GT_path = '../data/test_data/test_img/example_GT/'
    data = Data(input_path, GT_path)
