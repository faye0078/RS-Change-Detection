import fnmatch
import os
import pandas as pd
import numpy as np
import  glob

def changeFile():
    file = open("E:/wangyu_file/rs_Nas/src/data/lists/rs_test.lst", "r", encoding='UTF-8')
    file_list = file.readlines()
    file_name = []
    for i in range(file_list.__len__()):
        a = str(file_list[i].replace('test', 'train')).replace('\n', '')
        file_name.append(a)
    df = pd.DataFrame(file_name, columns=['one'])
    df.to_csv('E:/wangyu_file/rs_Nas/src/data/lists/rs_test1.lst', columns=['one'], index=False, header=False)

    file.close()

def mergeFile():
    file1 = open("E:/wangyu_file/1.lst", "r", encoding='UTF-8')
    file2 = open("E:/wangyu_file/2.lst", "r", encoding='UTF-8')
    file_list1 = file1.readlines()  # 将所有变量读入列表file_list1
    file_list2 = file2.readlines()  # 将所有变量读入列表file_list2
    file_list = []
    for i in range(file_list1.__len__()):
        a = str(file_list1[i])
        a = a.replace('\n', '').replace('\\', '/')
        b = str(file_list2[i])
        b = b.replace('\n', '').replace('\\', '/').replace('goundTruth', 'groundTruth')
        file_list.append(a + '\t'+ b)
    df = pd.DataFrame(file_list, columns=['one'])
    df.to_csv('E:/wangyu_file/rs_Nas/src/data/lists/rs_test.lst', columns=['one'], index=False, header=False)
    # file = open("train_pair.lst", "w")
    # file.writelines(file_list)
    file1.close()
    file2.close()
    # file.close()


def ReadSaveAddr(Stra, Strb):
    df = pd.DataFrame(np.arange(0).reshape(0, 1), columns=['Addr'])
    print(df)
    path = Stra
    for dirpath, dirnames, filenames in os.walk(path):
        filenames_len = filenames.__len__()
        # a_list = fnmatch.filter(os.listdir(dirpath),Strb)
        if filenames_len:
            dft = pd.DataFrame(np.arange(filenames_len).reshape((filenames_len, 1)), columns=['Addr'])
            # for i in range(len(filenames)):
            #     filenames[i] = filenames[i].replace('.tif', '.png')
            dft.Addr = filenames
            dft.Addr = dirpath.replace('E:/wangyu_file/rs_Nas/src/data/datasets/VOCdevkit/', '') + '/' + dft.Addr  # 输出绝对路径
            frames = [df, dft]
            df = pd.concat(frames)
            print(df.shape)
    df.to_csv('E:/wangyu_file/2.lst', columns=['Addr'], index=False, header=False)  # ***.lst即为最终保存的文件名，可修改
    print("Write To Get.lst !")

def ReadFileBytxt(file_path):
    file = []
    imgs = glob.glob('{}*.jpg'.format(file_path))

    for img in imgs:
        img1 = img.replace('D:/DPCode/software-cup/RS-Change-Detection/Datasets/', '')
        img2 = img1.replace('t1', 't2')
        label = img1.replace('t1', 'mask').replace('.jpg', '.png')
        file.append(img1 + '\t' + img2 + '\t' + label)

    df = pd.DataFrame(file, columns=['one'])
    df.to_csv("D:/DPCode/software-cup/RS-Change-Detection/Datasets/DSIFN-Dataset/val.lst", columns=['one'], index=False, header=False)

if __name__ == '__main__':
    # InputStra="E:/wangyu_file/rs_Nas/src/data/datasets/VOCdevkit/512/test/label"#数据存在的路径
    # InputStra = "E:/wangyu_file/rs_Nas/src/data/datasets/VOCdevkit/512/train/label"
    # InputStrb = "*.png"
    # ReadSaveAddr(InputStra, InputStrb)
    # mergeFile()
    # changeFile()
    path = "D:/DPCode/software-cup/RS-Change-Detection/Datasets/DSIFN-Dataset/val/t1/"
    ReadFileBytxt(path)