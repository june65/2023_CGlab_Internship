from exr import *
from diffuse import diff_model
from specular import spec_model
import torch
import os


if __name__ == '__main__':

    image_path = '../data/example/'
    model_path = './model/'
    model_version = open('./model/model_version.txt', 'r').read()

    data_list = sorted(os.listdir(path=image_path))

    diff_net, diff_AC_loss = diff_model()
    spec_net, spec_AC_loss = spec_model()

    open('./model/model_version.txt',
         'w').write(str(int(model_version)+1))

    torch.save(diff_net.state_dict(), model_path +
               'diffuse/KPCN_diff_'+model_version+'.pth')
    torch.save(spec_net.state_dict(), model_path +
               'specular/KPCN_spec_'+model_version+'.pth')

    model_text = open('./model_learning/model_info.txt', 'r').read()
    model_text = model_text + '\nversion_{} : diff_LOSS:{}, spec_LOSS:{}, change:{}'.format(
        model_version, diff_AC_loss.pop(), spec_AC_loss.pop(), model_version)
    open('./model/model_info.txt', 'w').write(model_text)
