from exr import *
from diffuse import diff_model
from spacular import spac_model
import torch
import os


if __name__ == '__main__':

    image_path = '../data/example/'
    model_path = './model_learning/model/'
    model_version = open('./model_learning/model_version.txt', 'r').read()

    data_list = sorted(os.listdir(path=image_path))

    diff_N, diff_AC_L = diff_model()
    spac_N, spac_AC_L = spac_model()

    open('./model_learning/model_version.txt',
         'w').write(str(int(model_version)+1))

    torch.save(diff_N.state_dict(), model_path +
               'diffuse/DPCN_diff_'+model_version+'.pth')
    torch.save(spac_N.state_dict(), model_path +
               'spacular/DPCN_spac_'+model_version+'.pth')

    model_text = open('./model_learning/model_info.txt', 'r').read()
    model_text = model_text + '\nversion_{} : diff_LOSS:{}, spac_LOSS:{}, change:{}'.format(
        model_version, diff_AC_L.pop(), spac_AC_L.pop(), model_version)
    open('./model_learning/model_info.txt', 'w').write(model_text)
