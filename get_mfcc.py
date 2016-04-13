#coding:gbk
'''
Created on 2016年4月10日

@author: wumeng
'''
from mfcc import  calcMFCC
from mfcc import  log_fbank
import scipy.io.wavfile as wav
import os
def get_train1mfcc(path):
    if not os.path.isdir(path):
        print('路径不正确')
        return False
    for root,dirs,list in os.walk(path):
    #root遍历路径，dirs当前遍历路径下的目录，list当前遍历目录下的文件名 
        for i in list:
            dir=os.path.join(root,i)
            (rate,sig)=wav.read(dir)
            mfcc_feat=calcMFCC(sig,rate)
            with open(r'C:\Users\wumeng\Desktop\train_data1.txt','a') as f:
                #f.write(dir)
                #f.write('\n')
                for j in range(len(mfcc_feat)):
                    #f.write(str(j+1))
                    f.write(str(mfcc_feat[j]).strip('[]')+str(' >1 -1 '))
                    f.write('\n')
                f.close()
            print('mfcc训练参数成功生成！！')

def get_train2mfcc(path):
    if not os.path.isdir(path):
        print('路径不正确')
        return False
    for root,dirs,list in os.walk(path):
    #root遍历路径，dirs当前遍历路径下的目录，list当前遍历目录下的文件名 
        for i in list:
            dir=os.path.join(root,i)
            (rate,sig)=wav.read(dir)
            mfcc_feat=calcMFCC(sig,rate)
            with open(r'C:\Users\wumeng\Desktop\train_data1.txt','a') as f:
                #f.write(dir)
                #f.write('\n')
                for j in range(len(mfcc_feat)):
                    #f.write(str(j+1))
                    f.write(str(mfcc_feat[j]).strip('[]')+str(' >-1 1 '))
                    f.write('\n')
                f.close()
            print('mfcc训练参数成功生成！！')

def get_test1mfcc(path):
    if not os.path.isdir(path):
        print('路径不正确')
        return False
    for root,dirs,list in os.walk(path):
    #root遍历路径，dirs当前遍历路径下的目录，list当前遍历目录下的文件名 
        for i in list:
            dir=os.path.join(root,i)
            (rate,sig)=wav.read(dir)
            mfcc_feat=calcMFCC(sig,rate)
            with open(r'C:\Users\wumeng\Desktop\test_data1.txt','a') as f:
                #f.write(dir)
                #f.write('\n')
                for j in range(len(mfcc_feat)):
                    #f.write(str(j+1))
                    f.write(str(mfcc_feat[j]).strip('[]')+str(' >1 -1 '))
                    f.write('\n')
                f.close()
            print('mfcc测试参数成功生成！！')

def get_test2mfcc(path):
    if not os.path.isdir(path):
        print('路径不正确')
        return False
    for root,dirs,list in os.walk(path):
    #root遍历路径，dirs当前遍历路径下的目录，list当前遍历目录下的文件名 
        for i in list:
            dir=os.path.join(root,i)
            (rate,sig)=wav.read(dir)
            mfcc_feat=calcMFCC(sig,rate)
            with open(r'C:\Users\wumeng\Desktop\test_data1.txt','a') as f:
                #f.write(dir)
                #f.write('\n')
                for j in range(len(mfcc_feat)):
                    #f.write(str(j+1))
                    f.write(str(mfcc_feat[j]).strip('[]')+str(' >-1 1 '))
                    f.write('\n')
                f.close()
            print('mfcc测试参数成功生成！！')
            
#生成训练数据并自动标注
get_train1mfcc(r'C:\Users\wumeng\Desktop\train1_wave')
get_train2mfcc(r'C:\Users\wumeng\Desktop\train2_wave')
#生成测试数据并自动标注
get_test1mfcc(r'C:\Users\wumeng\Desktop\test1_wave')
get_test2mfcc(r'C:\Users\wumeng\Desktop\test2_wave')
                    
    
            
          