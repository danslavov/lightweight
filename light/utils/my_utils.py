from cv2 import cv2 as cv2
import numpy as np


def freeze_encoder(model, start_module=0, end_module=1):
    module_list = [module for module in model.children()][start_module:end_module]
    for module in module_list:
        freeze_parameters_recursively(module)


def freeze_parameters_recursively(module):
    # freeze all parameters of current module
    for parameter in module.parameters():
        parameter.requires_grad = False
    # call the same function on current module's children
    for submodule in module.children():
        freeze_parameters_recursively(submodule)


def print_trainable_state(model):
    module_list = [module for module in model.named_children()]
    for name, module in module_list:
        print(name)
        for parameter in module.parameters():
            print(parameter.requires_grad)
        print_trainable_state(module)


def print_modules_with_index_and_name(model):
    module_list = [module for module in model.named_children()]
    index = 0
    for name, module in module_list:
        print('{}. {}'.format(index, name))
        index += 1


def save_output(output):

    # open image and mask
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    # convert pred tensor to ndarray
    pred = np.array(pred)
    # pred = np.swapaxes(pred, 0, 2)

    # argmax and color


    cv2.imwrite('C:/Users/Admin/Desktop/tmp/img.png', img)
    cv2.imwrite('C:/Users/Admin/Desktop/tmp/target.png', target)
    # save_image(img, 'C:/Users/Admin/Desktop/tmp/img.png')
    # save_image(target, 'C:/Users/Admin/Desktop/tmp/target.png')