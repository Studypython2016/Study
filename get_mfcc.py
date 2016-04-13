#coding:gbk
'''
Created on 2016��4��10��

@author: wumeng
'''
from mfcc import  calcMFCC
from mfcc import  log_fbank
import scipy.io.wavfile as wav
import os
def get_train1mfcc(path):
    if not os.path.isdir(path):
        print('·������ȷ')
        return False
    for root,dirs,list in os.walk(path):
    #root����·����dirs��ǰ����·���µ�Ŀ¼��list��ǰ����Ŀ¼�µ��ļ��� 
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
            print('mfccѵ�������ɹ����ɣ���')

def get_train2mfcc(path):
    if not os.path.isdir(path):
        print('·������ȷ')
        return False
    for root,dirs,list in os.walk(path):
    #root����·����dirs��ǰ����·���µ�Ŀ¼��list��ǰ����Ŀ¼�µ��ļ��� 
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
            print('mfccѵ�������ɹ����ɣ���')

def get_test1mfcc(path):
    if not os.path.isdir(path):
        print('·������ȷ')
        return False
    for root,dirs,list in os.walk(path):
    #root����·����dirs��ǰ����·���µ�Ŀ¼��list��ǰ����Ŀ¼�µ��ļ��� 
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
            print('mfcc���Բ����ɹ����ɣ���')

def get_test2mfcc(path):
    if not os.path.isdir(path):
        print('·������ȷ')
        return False
    for root,dirs,list in os.walk(path):
    #root����·����dirs��ǰ����·���µ�Ŀ¼��list��ǰ����Ŀ¼�µ��ļ��� 
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
            print('mfcc���Բ����ɹ����ɣ���')
            
#����ѵ�����ݲ��Զ���ע
get_train1mfcc(r'C:\Users\wumeng\Desktop\train1_wave')
get_train2mfcc(r'C:\Users\wumeng\Desktop\train2_wave')
#���ɲ������ݲ��Զ���ע
get_test1mfcc(r'C:\Users\wumeng\Desktop\test1_wave')
get_test2mfcc(r'C:\Users\wumeng\Desktop\test2_wave')
                    
    
            
          