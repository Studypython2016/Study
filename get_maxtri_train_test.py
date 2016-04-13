#coding:gbk
'''
Created on 2016��4��12��

@author: wumeng
'''
from nrural_net import *

INPUTS=13
HIDDEN=20
OUTPUTS=2

#����ѵ�����ݾ���
trainpat=[]
file=open(r'C:\Users\wumeng\Desktop\train_data.txt')
while 1:
    line1=file.readline()
    line2=file.readline()
    line3=file.readline()
    if not line1:
        break
    line=line1+line2+line3
    try:
        inputs,targets=line.split('>')   
    except:
        line4=file.readline()
        line=line+line4
        inputs,targets=line.split('>')
    in_lst=inputs.split()
    trg_lst=targets.split()
    
    # Convert strings to numbers
    for i in range(len(in_lst)):        
        in_lst[i] = eval(in_lst[i])
    for i in range(len(trg_lst)):
        trg_lst[i] = eval(trg_lst[i])
    
    trainpat.append([in_lst,trg_lst])
    print(trainpat)
    print('ѵ�����ݾ���ɹ����ɣ���')

#create a network (input, hidden, output)    
net = NN(INPUTS, HIDDEN, OUTPUTS)

# train it with some patterns
net.train(trainpat) 

#���ɲ������ݾ���
testpat=[]
file=open(r'C:\Users\wumeng\Desktop\test_data.txt')
while 1:
    line1=file.readline()
    line2=file.readline()
    line3=file.readline()
    if not line1:
        break
    try:
        line=line1+line2+line3
        inputs,targets=line.split('>')   
    except:
        line4=file.readline()
        line=line+line4
        inputs,targets=line.split('>')
    in_lst=inputs.split()
    trg_lst=targets.split()
    
    # Convert strings to numbers
    for i in range(len(in_lst)):        
        in_lst[i] = eval(in_lst[i])
    for i in range(len(trg_lst)):
        trg_lst[i] = eval(trg_lst[i])
    
    testpat.append([in_lst,trg_lst])
    print(testpat)
    print('�������ݾ���ɹ����ɣ���')

#��������    
net.test(testpat)
