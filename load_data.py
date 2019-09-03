import cv2, os
import numpy as np
import matplotlib.pyplot as plt
def load_DATA():
    #Path to Training and Validation Data
    dir_data = "/home/afia/FPN/GIAS/Preprocessed/"
    dir_seg = dir_data + "/Annotations/"
    dir_img = dir_data + "/Images/"


    #Path to Testing Data
    dirr_data = "/home/afia/FPN/GIAS/test/testA"
    dirr_seg = dirr_data + "/annotation/"
    dirr_img = dirr_data + "/image/"
    return  dir_data, dir_seg, dir_img, dirr_data, dirr_seg, dirr_img

def pre_process(dir_data, dir_seg, dir_img, dirr_data, dirr_seg, dirr_img ):
    ldseg = np.array(os.listdir(dir_seg))
    fnm = ldseg[2]
    print(fnm)

    width = 224
    height = 224

    seg = cv2.imread(dir_seg + fnm, 0)
    img = cv2.imread(dir_img + fnm, 1)
    ret, thresh = cv2.threshold(seg, 0, 255, cv2.THRESH_BINARY)
    ret, reverse = cv2.threshold(seg, 0, 255, cv2.THRESH_BINARY_INV)

    img1 = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1

    thresh1 = cv2.resize(thresh, (width, height)) / 255
    reverse1 = cv2.resize(reverse, (width, height))

    print(img1.shape, thresh1.shape, reverse1.shape)
    fig = plt.figure(figsize=(10, 40))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(img)
    ax.set_title("original")

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(seg)
    ax.set_title("original")

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(thresh)
    ax.set_title("original")
    cv2.imwrite('binary_image1.png', thresh)
    print(thresh.shape)

    fig2 = plt.figure(figsize=(10, 40))
    ax = fig2.add_subplot(1, 3, 1)
    ax.imshow(reverse)
    # fig2.imshow(reverse)
    ax.set_title("original")
    print(reverse.shape)
    cv2.imwrite('binary_image_inverse1.png', reverse)

    # In[5]:

    def getImageArr(path, width, height):
        img = cv2.imread(path, 1)
        img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
        return img

    def getSegmentationArr(path, nClasses, width, height):
        seg = cv2.imread(path, 0)
        seg = cv2.resize(seg, (width, height))
        ret, thresh = cv2.threshold(seg, 0, 255, cv2.THRESH_BINARY)
        ret, reverse = cv2.threshold(seg, 0, 255, cv2.THRESH_BINARY_INV)
        seg = thresh.reshape((width, height, -1))
        return seg

    def getSegID(path, nClasses, width, height):
        seg_labels = np.zeros((height, width, nClasses))
        img = cv2.imread(path, 1)
        img = cv2.resize(img, (width, height))
        img = img[:, :, 0]
        # print(img)
        # for c in range(nClasses):
        seg_labels[:, :, 0] = (img == 0).astype(int)
        seg_labels[:, :, 1] = (img != 0).astype(int)
        # print(seg_labels[: , : , c ] )
        ##seg_labels = np.reshape(seg_labels, ( width*height,nClasses  ))
        # print(seg_labels[:,:,1])
        return seg_labels

    # In[6]:

    input_width = 224
    input_height = 224
    n_classes = 2

    # In[7]:

    images = os.listdir(dir_img)
    images.sort()
    segmentations = os.listdir(dir_seg)
    segmentations.sort()

    X = []
    Y = []

    for im, seg in zip(images, segmentations):
        X.append(getImageArr(dir_img + im, input_width, input_height))
        # Y.append( getSegmentationArr( dir_seg + im , n_classes ,  input_width , input_height )  )
        Y.append(getSegID(dir_seg + im, n_classes, input_width, input_height))

    X, Y = np.array(X), np.array(Y)

    # X, Y = np.array(X, dtype=np.float16) , np.array(Y, dtype=np.float16)

    print(X.shape, Y.shape)

    # In[8]:

    from sklearn.utils import shuffle

    X, Y = shuffle(X, Y)

    print(X.shape, Y.shape)


    dirr_data = "/home/afia/FPN/GIAS/test/testA"
    dirr_seg = dirr_data + "/annotation/"
    dirr_img = dirr_data + "/image/"

    # In[10]:

    test_images = os.listdir(dirr_img)
    test_images.sort()
    test_segmentations = os.listdir(dirr_seg)
    test_segmentations.sort()

    Xtest = []
    Ytest = []

    def getSegmentationArr1(path, nClasses, width, height):
        t_seg = cv2.imread(path, 0)
        t_seg = cv2.resize(t_seg, (width, height))
        ret, thresh_seg = cv2.threshold(t_seg, 0, 255, cv2.THRESH_BINARY)
        ret, reverse_seg = cv2.threshold(t_seg, 0, 255, cv2.THRESH_BINARY_INV)
        # seg = thresh.reshape((width , height,-1 ))
        return thresh_seg

    for imm, segg in zip(test_images, test_segmentations):
        Xtest.append(getImageArr(dirr_img + imm, input_width, input_height))
        Ytest.append(getSegID(dirr_seg + imm, n_classes, input_width, input_height))

    X_test, y_test = np.array(Xtest), np.array(Ytest)
    print(X_test.shape, y_test.shape)
    return X, Y , X_test, y_test
