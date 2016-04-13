#coding:gbk
'''
Created on 2016年4月12日

@author: wumeng
'''
from nrural_net import *

INPUTS=13
HIDDEN=20
OUTPUTS=2

#生成训练数据矩阵
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
    print('训练数据矩阵成功生成！！')

#create a network (input, hidden, output)    
net = NN(INPUTS, HIDDEN, OUTPUTS)

# train it with some patterns
net.train(trainpat) 

#生成测试数据矩阵
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
    print('测试数据矩阵成功生成！！')

#测试数据    
net.test(testpat)
