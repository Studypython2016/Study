#coding:gbk
'''
Created on 2016��4��9��

@author: wumeng
'''
#����Ƶ�źŴ������
__author__='wumeng'
#audio2frame:����Ƶ�ź�ת��Ϊ֡����
#deframesignal:��ÿһ֡��һ�����������ı任
#spectrum_magnitude:����ÿһ֡����Ҷ�任�Ժ�ķ���
#spectrum_power:����ÿһ֡����Ҷ�任�Ժ�Ĺ�����
#log_spectrum_power:����ÿһ֡����Ҷ�任�Ժ�Ķ���������
#pre_emphasis:��ԭʼ�źŽ���Ԥ����
import numpy
import math

def audio2frame(signal,frame_length,frame_step,winfunc=lambda x:numpy.ones((x,))):
    '''����Ƶ�ź�ת��Ϊ֡����
        signal:ԭʼ��Ƶ�ͺ�
        frame_length:ÿһ֡�ĳ���
        frame_step:����֡�ļ��
        winfunc:lambda��������������һ������
    '''
    signal_length=len(signal)
    frame_length=int(round(frame_length))
    frame_step=int(round(frame_step))
    if signal_length<=frame_length:
        frames_num=1
    else:#�������֡����
        frames_num=1+int(math.ceil((1.0*signal_length-frame_length)/frame_step))
    pad_length=int((frames_num-1)*frame_step+frame_length)
    zeros=numpy.zeros((pad_length-signal_length,))#�����ĳ���ʹ��0���������FFT�е������������
    pad_signal=numpy.concatenate((signal,zeros))
    indices=numpy.tile(numpy.arange(0,frame_length),(frames_num,1))+numpy.tile(numpy.arange(0,frames_num*frame_step,frame_step),(frame_length,1)).T
    indices=numpy.array(indices,dtype=numpy.int32)#��indicesת��Ϊ����
    frames=pad_signal[indices]#�õ�֡�ź�
    win=numpy.tile(winfunc(frame_length),(frames_num,1))#windows������,����Ĭ��ȡ1
    return frames*win

def deframesignal(frames,signal_length,frame_length,frame_step,winfunc=lambda x:numpy.ones((x,))):
    '''���庯����ԭ�źŵ�ÿһ֡���б任��Ϊ������������
    frames:audio2frame�������ص�֡����
    signal_length:�źų���
    frame_length:֡����
    frame_step:֡���
    winfunc:��ÿһ֡��window�������з�����Ĭ�ϴ˴�����window
    '''
    #�Բ�������ȡ������
    signal_length=round(signal_length)
    frame_length=round(frame_length)
    frames_num=numpy.shape(frames)[0] #֡������
    assert numpy.shape(frames)[1]==frame_length,'"frames"�����С����ȷ����������Ӧ�õ���һ֡����'
    indices=numpy.tile(numpy.arange(0,frame_length),(frames_num,1))+numpy.tile(numpy.array(0,frames_num*frame_step,frame_step),(frame_length,1).T)
    indices=numpy.array(indices,dtype=numpy.int32)
    pad_length=(frames_num-1)*frame_step+frame_length
    if signal_length<=0:
        signal_length=pad_length
    recalc_signal=numpy.zeros((pad_length,)) #��������ź�
    window_correction=numpy.zeros((pad_length,1))#������
    win=winfunc(frame_length)
    for i in range(0,frames_num):
        window_correction[indices[i,:]]=window_correction[indices[i,:]]+win+1e-15 #��ʾ�źŵ��ص��̶�
        recalc_signal[indices[i,:]]=recalc_signal[indices[i,:]]+frames[i,:] #ԭ�źż����ص��̶ȹ��ɵ����ź�
    recalc_signal=recalc_signal/window_correction #�µĵ�������źŵ��ڵ����źų���ÿ�����ص��̶� 
    return recalc_signal[0:signal_length] #���ظ��µĵ����ź�

def spectrum_magnitude(frames,NFFT):
    '''����ÿһ֡����FFY�任�Ժ��Ƶ�׵ķ��ȣ���frames�Ĵ�СΪN*L,�򷵻ؾ���Ĵ�СΪN*NFFT
    frames:��audio2frame�����еķ���ֵ����֡����
    NFFT:FFT�任�������С,���֡����С��NFFT����֡�����ಿ����0�������
    '''
    complex_spectrum=numpy.fft.rfft(frames,NFFT) #��frames����FFT�任
    return numpy.absolute(complex_spectrum) #����Ƶ�׵ķ���ֵ
    
def spectrum_power(frames,NFFT):
    '''����ÿһ֡����Ҷ�任�Ժ�Ĺ�����
    frames:audio2frame�������������֡����
    NFFT:FFT�Ĵ�С
    '''
    return (1.0/NFFT)*numpy.square(spectrum_magnitude(frames,NFFT))#�����׵���ÿһ��ķ���ƽ��/NFFT

def log_spectrum_power(frames,NFFT,norm=1):
    '''����ÿһ֡�Ĺ����׵Ķ�����ʽ
    frames:֡���󣬼�audio2frame���صľ���
    NFFT��FFT�任�Ĵ�С
    norm:����������һ��ϵ��
    '''
    spec_power=spectrum_power(frames,NFFT)
    spec_power[spec_power<1e-30]=1e-30 #Ϊ�˷�ֹ���ֹ����׵���0����Ϊ0�޷�ȡ����
    log_spec_power=10*numpy.log10(spec_power)
    if norm:
        return log_spec_power-numpy.max(log_spec_power)
    else:
        return log_spec_power
    
def pre_emphasis(signal,coefficient=0.95):
    '''���źŽ���Ԥ����
    signal:ԭʼ�ź�
    coefficient:����ϵ����Ĭ��Ϊ0.95
    '''
    return numpy.append(signal[0],signal[1:]-coefficient*signal[:-1])
