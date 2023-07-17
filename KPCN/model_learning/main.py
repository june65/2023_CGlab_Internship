from exr import *
from diffuse import diff_model
from specular import spec_model
import matplotlib.pyplot as plt
import torch
import os


def draw_result(epochs, diff_AC_loss, val_diff_AC_loss, spec_AC_loss, val_spec_AC_loss):

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    ax1.plot(diff_AC_loss, label='Diffuse train')
    ax1.plot(val_diff_AC_loss, label='Diffuse val')
    ax1.set_title('Diffuse Loss')
    ax1.set_xlim([0, epochs])
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('loss')
    ax1.legend()

    ax2.plot(spec_AC_loss, label='Specular train')
    ax2.plot(val_spec_AC_loss, label='Specular val')
    ax2.set_title('Specular Loss')
    ax2.set_xlim([0, epochs])
    ax2.set_xlabel('Epochs')
    ax2.legend()

    plt.show()


if __name__ == '__main__':

    model_path = './model/'
    model_version = open('./model/model_version.txt', 'r').read()

    epochs = 100
    
    spec_net, spec_AC_loss, val_spec_AC_loss = spec_model(epochs)
    diff_net, diff_AC_loss, val_diff_AC_loss = diff_model(epochs)
    

    open('./model/model_version.txt',
         'w').write(str(int(model_version)+1))

    torch.save(diff_net.state_dict(), model_path +
               'diffuse/KPCN_diff_'+model_version+'.pth')
    torch.save(spec_net.state_dict(), model_path +
               'specular/KPCN_spec_'+model_version+'.pth')

    draw_result(epochs, diff_AC_loss, val_diff_AC_loss,
                spec_AC_loss, val_spec_AC_loss)

    model_text = open('./model/model_info.txt', 'r').read()

    model_text = model_text + '\nversion_{} : diff_LOSS:{},val_diff_LOSS:{}, spec_LOSS:{}, val_spec_LOSS:{}, change:{}'.format(
        model_version, diff_AC_loss.pop(), val_diff_AC_loss.pop(), spec_AC_loss.pop(), val_spec_AC_loss.pop(), spec_AC_loss.pop(), model_version)
    open('./model/model_info.txt', 'w').write(model_text)
