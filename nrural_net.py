#coding:gbk
'''
Created on 2016��4��11��

@author: wumeng
'''
#���򴫲�������:����������
import math
import random
from test.test_threading_local import target

random.seed(0)

def rand(a,b):
    '''
    ����һ������a<=rand<b�������
    :param a:
    :param b:
    :return:
    '''
    return (b-a)*random.random()+a

def makeMatrix(I,J,fill=0.0):
    """
    ����һ�����󣨿��Կ�����NumPy�����٣�
    :param I: ����
    :param J: ����
    :param fill: ���Ԫ�ص�ֵ
    :return:
    """
    m = []
    for i in range(I):
        m.append([fill] * J)
    return m

def randomizeMatrix(matrix, a, b):
    """
          �����ʼ������
    :param matrix:
    :param a:
    :param b:
    """
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = random.uniform(a, b)
            
def sigmoid(x):
    """
    sigmoid ������1/(1+e^-x)
    :param x:
    :return:
    """
    return 1.0 / (1.0 + math.exp(-x))

def dsigmoid(y):
    """
    sigmoid �����ĵ���
    :param y:
    :return:
    """
    return y * (1 - y)

class NN:
    def __init__(self,ni,nh,no):
        # number of input, hidden, and output nodes
        """
                   ����������
        :param ni:���뵥Ԫ����
        :param nh:���ص�Ԫ����
        :param no:�����Ԫ����
        """
        self.ni = ni + 1  # +1 ��Ϊ��ƫ�ýڵ�
        self.nh = nh
        self.no = no

        # ����ֵ������ֵ��
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no
        
        #Ȩ�ؾ���
        self.wi = makeMatrix(self.ni, self.nh)  # ����㵽���ز�
        self.wo = makeMatrix(self.nh, self.no)  # ���ز㵽�����
        # ��Ȩ�ؾ��������
        randomizeMatrix(self.wi, -0.2, 0.2)
        randomizeMatrix(self.wo, -2.0, 2.0)
        # Ȩ�ؾ�����ϴ��ݶ�
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)
        
    def runNN(self,inputs):
        '''
         ǰ�򴫲����з���
        :param inputs:����
        :return:���
        '''
        if len(inputs) != self.ni - 1:
            print('incorrect number of inputs')
        
        for i in range(self.ni-1):
            self.ai[i]=inputs[i]
            
        for j in range(self.nh):
            sum=0.0
            for i in range(self.ni):
                sum += ( self.ai[i] * self.wi[i][j] )
            self.ah[j] = sigmoid(sum)
        
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum += ( self.ah[j] * self.wo[j][k] )
            self.ao[k] = sigmoid(sum)
        return self.ao

    def backPropagate(self, targets, N, M):
        """
          ���򴫲��㷨
    :param targets: ʵ�������
    :param N: ����ѧϰ��
    :param M: �ϴ�ѧϰ��
    :return: ���յ����ƽ���͵�һ��
        """
    # ��������� deltas
        output_deltas=[0.0]*self.no
        for k in range(self.no):
            error=targets[k]-self.ao[k]
            output_deltas[k]=error*dsigmoid(self.ao[k])
    
    #���������Ȩֵ
        for j in range(self.nh):
            for k in range(self.no):
                change=output_deltas[k]*self.ah[j]
                self.wo[j][k]=N*change+M*self.co[j][k]
                self.co[j][k]=change
    
    # �������ز� deltas
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = error * dsigmoid(self.ah[j])
    
    #���������Ȩֵ
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
            # print 'activation',self.ai[i],'synapse',i,j,'change',change
                self.wi[i][j] += N * change + M * self.ci[i][j]
                self.ci[i][j] = change
    
    #�������ƽ����
    # 1/2 ��Ϊ�˺ÿ���**2 ��ƽ��
        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def weights(self):
        """
        ��ӡȨֵ����
        """
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])   

    def test(self,patterns):
        """
          ����
    :param patterns:��������
        """
        print('--------------------��������-----------------------')
        for p in patterns:
            inputs = p[0]
            print('Inputs:', p[0], '--->', self.runNN(inputs), '\tTarget', p[1])
        hits=0
        for p in patterns:
            hits=hits+self.compare1(p[1], self.runNN(p[0]))
        hit_rate=(100*hits)/len(patterns)
        print('\nSuccess rate against test data:%.2f'%hit_rate)


    '''def compare(self,targets,activations):
        matches=0
        for n in range(len(targets)):
            error=abs(targets[n]-activations[n])
            print('%.3f'%error)
            if error<0.5:
                print('Ok')
                matches=matches+1
            else:
                print('Fail')
        if matches==self.no:
            print('--SUCCESS!!')
            return 1
        else:
            print('--FAILURE!!')
            return 0
    '''
    def compare1(self,targets,activations):
        if targets[0]>targets[1]:
            if activations[0]<activations[1]:
                return 1
                
        elif targets[0]<targets[1]:
            if activations[0]>activations[1]:
                return 1
        return 0
        
            
    def train(self, patterns, max_iterations=1000, N=0.5, M=0.1):
        """
        ѵ��
    :param patterns:ѵ����
    :param max_iterations:����������
    :param N:����ѧϰ��
    :param M:�ϴ�ѧϰ��
        """
        for i in range(max_iterations):
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.runNN(inputs)
                error = self.backPropagate(targets, N, M)
            if i % 50 == 0:
                print('Combined error', error)
            print('ѵ����'+str(i)+'��')
        print('����������ѵ���ɹ�����')
     

       