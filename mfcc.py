#coding:gbk
'''
Created on 2016��4��9��

@author: wumeng
'''
#����ÿһ֡��MFCCϵ��
__author__='wumeng'

import numpy
from frame import audio2frame
from frame import pre_emphasis
from frame import spectrum_power
from scipy.fftpack import dct

    
def calcMFCC_delta_delta(signal,samplerate=16000,win_length=0.025,win_step=0.01,cep_num=13,
                         filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97,cep_lifter=22,
                         appendEnergy=True):
    '''
          ������Ƶ�źŵ�13��MFCC+13��һ��΢��ϵ��+13������ϵ��,һ��39��ϵ��
    '''
    feat=calcMFCC(signal,samplerate,win_length,win_step,cep_num,filters_num,NFFT,low_freq,high_freq,
                   pre_emphasis_coeff,cep_lifter,appendEnergy)
    result1=derivate(feat)
    result2=derivate(result1)
    result3=numpy.concatenate((feat,result1),axis=1)
    result=numpy.concatenate((result3,result2),axis=1)
    return result

def calcMFCC_delta(signal,samplerate=16000,win_length=0.025,win_step=0.01,cep_num=13,
                         filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97,ceplifter=22,
                         appendEnergy=True):
    '''������Ƶ�źŵ�13��MFCC+13��һ��΢��ϵ��
    '''
    feat=calcMFCC(signal,samplerate,win_length,win_step,cep_num,filters_num,NFFT,low_freq,high_freq,
                   pre_emphasis_coeff,ceplifter,appendEnergy)
    result=derivate(feat) #����derivate����
    result=numpy.concatenate((feat,result),axis=1)
    return result 

def derivate(feat,big_theta=2,cep_num=13):
    '''������Ƶ�źŵ�һ��ϵ�����߼���ϵ����һ��任��ʽ
    feat:MFCC�������һ��ϵ������
    big_theta:��ʽ�еĴ�theta��Ĭ��ȡ2
    '''
    result=numpy.zeros(feat.shape) #���
    denominator=0  #��ĸ
    for theta in numpy.linspace(1,big_theta,big_theta):
        denominator=denominator+theta**2
    denominator=denominator*2 #����õ���ĸ��ֵ
    for row in numpy.linspace(0,feat.shape[0]-1,feat.shape[0]):
        tmp=numpy.zeros((cep_num,))
        numerator=numpy.zeros((cep_num,)) #����
        for t in numpy.linspace(1,cep_num,cep_num):
            a=0
            b=0
            s=0
            for theta in numpy.linspace(1,big_theta,big_theta):
                if (t+theta)>cep_num:
                    a=0
                else:
                    a=feat[row][t+theta-1]
                if (t-theta)<1:
                    b=0
                else:
                    b=feat[row][t-theta-1]
                s+=theta*(a-b)
            numerator[t-1]=s
        tmp=numerator*1.0/denominator
        result[row]=tmp
    return result

def calcMFCC(signal,samplerate=16000,win_length=0.025,win_step=0.01,cep_num=13,
                         filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97,cep_lifter=22,
                         appendEnergy=True): 
    '''������Ƶ�źŵ�13��MFCCϵ��
    signal:ԭʼ��Ƶ�źţ�һ��Ϊ.wav��ʽ�ļ�
    samplerate:����Ƶ�ʣ�����Ĭ��Ϊ16KHz
    win_length:�����ȣ�Ĭ�ϼ�һ֡Ϊ25ms
    win_step:�������Ĭ������¼�����֡��ʼʱ��֮�����10ms
    cep_num:����ϵ���ĸ�����Ĭ��Ϊ13
    filters_num:�˲����ĸ�����Ĭ��Ϊ26
    NFFT:����Ҷ�任��С��Ĭ��Ϊ512
    low_freq:���Ƶ�ʣ�Ĭ��Ϊ0
    high_freq:���Ƶ��
    pre_emphasis_coeff:Ԥ����ϵ����Ĭ��Ϊ0.97
    cep_lifter:���׵�������
    appendEnergy:�Ƿ����������Ĭ�ϼ�
    '''
    feat,energy=fbank(signal,samplerate=16000,win_length=0.025,win_step=0.01,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97)
    feat=numpy.log(feat)
    feat=dct(feat,type=2,axis=1,norm='ortho')[:,:cep_num]  #������ɢ���ұ任,ֻȡǰ13��ϵ��
    feat=lifter(feat,cep_lifter)
    if appendEnergy:
        feat[:,0]=numpy.log(energy)  #ֻȡ2-13��ϵ������һ���������Ķ���������
    return feat 

def fbank(signal,samplerate=16000,win_length=0.025,win_step=0.01,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97):
    '''������Ƶ�źŵ�MEL�˲��������������
    samplerate:����Ƶ��
    win_length:������
    win_step:�����
    filters_num:÷���˲�������
    NFFT:FFT��С
    low_freq:���Ƶ��
    high_freq:���Ƶ��
    pre_emphasis_coeff:Ԥ����ϵ��
    '''
    high_freq=high_freq or samplerate/2  #������Ƶ���������Ƶ��
    signal=pre_emphasis(signal,pre_emphasis_coeff)  #��ԭʼ�źŽ���Ԥ���ش���
    frames=audio2frame(signal,win_length*samplerate,win_step*samplerate) #�õ�֡����
    spec_power=spectrum_power(frames,NFFT)  #�õ�ÿһ֡FFT�Ժ��������
    energy=numpy.sum(spec_power,1)  #��ÿһ֡�������׽������
    energy=numpy.where(energy==0,numpy.finfo(float).eps,energy)  #������Ϊ0�ĵط�����Ϊeps
    fb=get_filter_banks(filters_num,NFFT,samplerate,low_freq,high_freq)  #���ÿһ���˲�����Ƶ�ʿ��
    feat=numpy.dot(spec_power,fb.T)  #���˲����������׽��е��
    feat=numpy.where(feat==0,numpy.finfo(float).eps,feat)  #ͬ�����ܳ���0
    return feat,energy

def log_fbank(signal,samplerate=16000,win_length=0.025,win_step=0.01,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97):
    '''������Ƶ�źŵ�MEL�˲��������������ȡ����
    '''
    feat,energy=fbank(signal,samplerate=16000,win_length=0.025,win_step=0.01,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97)
    return numpy.log(feat)

def ssc(signal,samplerate=16000,win_length=0.025,win_step=0.01,filters_num=26,NFFT=512,low_freq=0,high_freq=None,pre_emphasis_coeff=0.97):
    '''������Ƶ�źŵĹ��ײ��ֲ�����������
    '''
    high_freq=high_freq or samplerate/2
    signal=pre_emphasis(signal,pre_emphasis_coeff)
    frames=audio2frame(signal,win_length*samplerate,win_step*samplerate)
    spec_power=spectrum_power(frames,NFFT) 
    spec_power=numpy.where(spec_power==0,numpy.finfo(float).eps,spec_power) #������
    fb=get_filter_banks(filters_num,NFFT,samplerate,low_freq,high_freq) 
    feat=numpy.dot(spec_power,fb.T)  #��������
    R=numpy.tile(numpy.linspace(1,samplerate/2,numpy.size(spec_power,1)),(numpy.size(spec_power,0),1))
    return numpy.dot(spec_power*R,fb.T)/feat

def hz2mel(hz):
    '''��Ƶ��hzת��Ϊ÷��Ƶ��
    hz:Ƶ��
    '''
    return 2595*numpy.log10(1+hz/700.0)

def mel2hz(mel):
    '''��÷��Ƶ��ת��Ϊhz
    mel:÷��Ƶ��
    '''
    return 700*(10**(mel/2595.0)-1)

def get_filter_banks(filters_num=20,NFFT=512,samplerate=16000,low_freq=0,high_freq=None):
    '''����÷�����Ǽ���˲����飬���˲����ڵ�һ��Ƶ�ʺ͵�����Ƶ�ʴ�Ϊ0���ڵڶ���Ƶ�ʴ�Ϊ1
    filers_num:�˲�������
    NFFT:FFT��С
    samplerate:����Ƶ��
    low_freq:���Ƶ��
    high_freq:���Ƶ��
    '''
    #���ȣ���Ƶ��hzת��Ϊ÷��Ƶ�ʣ���Ϊ�˶��ֱ������Ĵ�С��Ƶ�ʲ����������ȣ����Ի�Ϊ÷��Ƶ�������Էָ�
    low_mel=hz2mel(low_freq)
    high_mel=hz2mel(high_freq)
    #��Ҫ��low_mel��high_mel֮��ȼ�����filters_num���㣬һ��filters_num+2����
    mel_points=numpy.linspace(low_mel,high_mel,filters_num+2)
    #�ٽ�÷��Ƶ��ת��ΪhzƵ�ʣ������ҵ���Ӧ��hzλ��
    hz_points=mel2hz(mel_points)
    #����������Ҫ֪����Щhz_points��Ӧ��fft�е�λ��
    bin=numpy.floor((NFFT+1)*hz_points/samplerate)
    #�����������˲����ı��ʽ�ˣ�ÿ���˲����ڵ�һ���㴦�͵������㴦��Ϊ0���м�Ϊ��������״
    fbank=numpy.zeros([filters_num,NFFT/2+1])
    for j in range(0,filters_num):
        for i in range(int(bin[j]),int(bin[j+1])):
            fbank[j,i]=(i-bin[j])/(bin[j+1]-bin[j])
        for i in range(int(bin[j+1]),int(bin[j+2])):
            fbank[j,i]=(bin[j+2]-i)/(bin[j+2]-bin[j+1])
    return fbank

def lifter(cepstra,L=22):
    '''���׺���
    cepstra:the matrix of mel-cepstra, will be numframes * numcep in size.
    param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter
    '''
    if L>0:
        nframes,ncoeff=numpy.shape(cepstra)
        n=numpy.arange(ncoeff)
        lift=1+(L/2)*numpy.sin(numpy.pi*n/L)
        return lift*cepstra
    else:
        return cepstra
