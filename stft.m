for j = 1:10
    batch_num = j;
    data_name = sprintf('input/RFRecord_%d.mat',batch_num);
    dataset = load(data_name);
    Y = dataset.mat;
    % each time: 0.0262s (I guess maybe 0.026199990005496976662835440507721s actual
    % total data length: 1048576 points
    % computing the sample rate
    fs = 40022000; % frequency of sample : in 1 s we take 400220 samples
    t = 0:1/fs:0.0262;
    for i = 1:10
        y = Y([i],:); % take the first group of samples
        f = figure(2);
        colormap default;
        spectrogram(y,256,128,256,fs,'yaxis');
        img_name = sprintf('outputs/%d_with_axis.jpg',i+(batch_num-1)*10);
%         set(gca,'position',[0 0 1 1]);
%         set(gca,'XTick',[],'YTick',[]);
        saveas(f,img_name);
        close(f);
    end
end
%[S,F,T,P]=spectrogram(x,window,noverlap,nfft,fs)
% [S,F,T,P]=spectrogram(x,window,noverlap,F,fs)
%x---输入信号的向量。默认情况下，即没有后续输入参数，x将被分成8段分别做变换处理，
% 如果x不能被平分成8段，则会做截断处理。默认情况下，其他参数的默认值为
% window---窗函数，默认为nfft长度的海明窗Hamming,如果window为一个整数，每段长度为window，每段使用Hamming窗函数加窗。
% noverlap---每一段的重叠样本数，默认值是在各段之间产生50%的重叠,它必须为一个小于window或length(window)的整数
% nfft---做FFT变换的长度，默认为256和大于每段长度的最小2次幂之间的最大值。
% 另外，此参数除了使用一个常量外，还可以指定一个频率向量F
% fs---采样频率，默认值归一化频率
% spectrogram(...,'freqloc')使用freqloc字符串可以控制频率轴显示的位置。
% 当freqloc=xaxis时，频率轴显示在x轴上，当freqloc=yaxis时，频率轴显示在y轴上，默认是显示在x轴上。如果在指定freqloc的同时，又有输出变量，则freqloc将被忽略。
% F---在输入变量中使用F频率向量，函数会使用Goertzel方法计算在F指定的频率处计算频谱图。
% 指定的频率被四舍五入到与信号分辨率相关的最近的DFT容器(bin)中。而在其他的使用nfft
% 语法中，短时傅里叶变换方法将被使用。对于返回值中的F向量，为四舍五入的频率，其长度
% 等于S的行数。
% T---频谱图计算的时刻点，其长度等于上面定义的k，值为所分各段的中点。
% P---能量谱密度PSD(Power Spectral Density)，对于实信号，P是各段PSD的单边周期估计；
% xlabel('时间(s)')
% ylabel('频率(Hz)')
% title('语谱图')