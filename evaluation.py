import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from Segmentation_Models import path


def performance_measures(Yi, y_predi, Model):

    IoUs = []
    pres = []
    rec = []
    ff = []
    sp = []
    PA = []
    Dic = []
    sen = []
    Nclass = 2
    for c in range(Nclass):
        TP = np.sum((Yi == c) & (y_predi == c))
        FP = np.sum((Yi != c) & (y_predi == c))
        FN = np.sum((Yi == c) & (y_predi != c))
        TN = np.sum((Yi != c) & (y_predi != c))

        IoU = TP / float(TP + FP + FN)
        IoUs.append(IoU)

        pixel_accuracy = float(TP + TN) / float(TP + FP + FN + TN)
        PA.append(pixel_accuracy)

        precision = TP / float(TP + FP)
        pres.append(precision)

        recall = TP / float(TP + FN)
        rec.append(recall)

        specificity = TN / float(TN + FP)
        sp.append(specificity)

        sensitivity = TP / float(TP + FN)
        sen.append(sensitivity)

        f1 = 2 * float(precision * recall) / float(precision + recall)
        ff.append(f1)

        dice = 2 * TP / float(2 * TP + FP + FN)
        Dic.append(dice)

        print("class {:02.0f}: #TP={:6.0f}, #FP={:6.0f}, #FN={:5.0f}, IoU={:4.3f}".format(c, TP, FP, FN, IoU))






    mIoU = np.mean(IoUs)
    print("_________________")
    print("Mean IoU: {:4.3f}".format(mIoU))

    mpixel_accuracy= np.mean(PA)
    print("_________________")
    print("Mean Pixel Accuracy: {:4.3f}".format(mpixel_accuracy))

    mprecision = np.mean(pres)
    print("_________________")
    print("Mean Precision: {:4.3f}".format(mprecision))

    mrecall = np.mean(rec)
    print("_________________")
    print("Mean Recall: {:4.3f}".format(mrecall))

    mf1 = np.mean(ff)
    print("_________________")
    print("Mean F1 Score: {:4.3f}".format(mf1))

    mdice=np.mean(Dic)
    print("_________________")
    print("Mean Dice: {:4.3f}".format(mdice))

    mspecificity = np.mean(sp)
    print("_________________")
    print("Mean Specificity: {:4.3f}".format(mspecificity))

    msensitivity = np.mean(sen)
    print("_________________")
    print("Mean Sensitivity: {:4.3f}".format(msensitivity))

    with open(path(Model) + '/Performance_Measures.txt', mode='w') as f:
        f = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        f.writerow(['Mean IoU:', mIoU])
        f.writerow(['Mean Pixel Accuracy:', mpixel_accuracy])
        f.writerow(['Mean Precision:', mprecision])
        f.writerow(['Mean Recall:', mrecall])
        f.writerow(['Mean F1 Score:'  , mf1])
        f.writerow(['Mean Dice:' , mdice])
        f.writerow(['Mean Specificity: ' , mspecificity])
        f.writerow(['Mean Sensitivity: ', msensitivity])



    df = pd.read_csv(path(Model) + '/Performance_Measures.txt')
    print(df)




def visualize_predictions(X_test,y_predi ,y_testi):
    for i in range(40):
        img_is = (X_test[i] + 1) * (255.0 / 2)
        seg = (y_predi[i] + 1) * (255.0 / 2)
        segtest = (y_testi[i] + 1) * (255.0 / 2)

        fig = plt.figure(figsize=(10, 30))
        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(img_is / 255.0)
        ax.set_title("original")

        ax = fig.add_subplot(1, 3, 2)
        ax.imshow(seg)
        ax.set_title("predicted class")

        ax = fig.add_subplot(1, 3, 3)
        ax.imshow(segtest)
        ax.set_title("true class")
        plt.show()

def save_history(results, Model):
    hist_df = pd.DataFrame(results.history)
    # save to json:
    hist_json_file = path(Model) + '/Training_History.json'
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)
    # or save to csv:
    hist_csv_file = path(Model) + '/Training_History.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)


def Plot_history(Model):
    df = pd.read_csv(path(Model) + '/Training_History.csv')
    print(df)

    for key in ['loss', 'val_loss']:
        plt.plot(df[key], label=key)
    plt.legend()
    plt.show()

    for key in ['acc', 'val_acc']:
        plt.plot(df[key], label=key)
    plt.legend()
    plt.show()
