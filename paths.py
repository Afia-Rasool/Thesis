def path(model):
    if model=='FPN':
        filepath = "/home/afia/PycharmProjects/Thesis/Segmentation_Models/FPN/results"
    elif model=='FPN_ASPP':
        filepath = "/home/afia/PycharmProjects/Thesis/Segmentation_Models/FPN_ASPP/results"
    elif model == 'SegNet':
        filepath = "/home/afia/PycharmProjects/Thesis/Segmentation_Models/SegNet/results"
    elif model == 'UNet':
        filepath = "/home/afia/PycharmProjects/Thesis/Segmentation_Models/UNet/results"
    elif model == 'FCN':
        filepath = "/home/afia/PycharmProjects/Thesis/Segmentation_Models/FCN/results"
    return filepath