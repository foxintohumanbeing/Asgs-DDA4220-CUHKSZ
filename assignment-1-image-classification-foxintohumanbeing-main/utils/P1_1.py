import os
import shutil
import random

def save(filename, data):  # filename为写入txt文件的路径，data为要写入数据列表.
        file = open(filename, 'a')
        s = str(data).replace(
                '[', '').replace(']', '') 
        s = s + '\n'
        file.write(s)
        file.close()

def save_train_val(filename, data,flower_name,flower_index):  # filename为写入txt文件的路径，data为要写入数据列表.
        file = open(filename, 'a')
        s = flower_name+'/'+str(data)+' '+str(flower_index)+'\n'
        file.write(s)
        file.close()

def write_name_into_class(check = False):
    tmp_path = '/Users/yanghuihan/Desktop/assignment-1-image-classification-foxintohumanbeing'
    data_path = './flower_dataset/'
    flower_kind = os.listdir(data_path)
    for i in range(len(flower_kind)):
        flower_i_name = flower_kind[i]
        save(os.path.join(data_path,'classes.txt'),flower_i_name)
    if check == True:
        with open('/flower_dataset/classes.txt','r') as f:
            a = f.readline()
            return len(a)

def move_seperate_and_record():
    data_path = './flower_dataset/'
    flower_kind = os.listdir(data_path)
    print(flower_kind)
    if not os.path.exists('./flower_dataset/train'):
        os.makedirs('./flower_dataset/train')
    if not os.path.exists('./flower_dataset/val'):
        os.makedirs('./flower_dataset/val')
    val_path = os.path.join(data_path,'val')
    train_path = os.path.join(data_path,'train')

    for j in range(len(flower_kind)):
        flower = flower_kind[j]
        flower_path = os.path.join(data_path,flower)
        picture_name = os.listdir(flower_path)
        data_folder_length = len(picture_name)
        val_path_j = os.path.join(val_path,flower)
        train_path_j = os.path.join(train_path,flower)
        if not os.path.exists(val_path_j):
            os.makedirs(val_path_j)
        if not os.path.exists(train_path_j):
            os.makedirs(train_path_j)    
        val_num = int(data_folder_length*0.2)
        val_list = random.sample(range(0,data_folder_length),val_num)
        for i in range(len(picture_name)):
            path_i = os.path.join(flower_path,picture_name[i])
            if i in val_list:
                save_train_val('./flower_dataset/val.txt',picture_name[i],flower,j)
                target_i = os.path.join(val_path_j,picture_name[i])
            else:
                save_train_val('./flower_dataset/train.txt',picture_name[i],flower,j)
                target_i = os.path.join(train_path_j,picture_name[i])
            shutil.move(path_i,target_i)