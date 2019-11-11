clc;
clear all;
close all;
info = h5info('./model2017-1_face12_nomouth.h5');
%% load the shape model parameters
shape_model = info.Groups(5).Groups(1);
shape_dataset = shape_model.Datasets;
shape_mean = h5read('./model2017-1_face12_nomouth.h5',replace(fullfile(shape_model.Name,shape_dataset(1).Name),'\','/'));
shape_noiseVariance = h5read('./model2017-1_face12_nomouth.h5',replace(fullfile(shape_model.Name,shape_dataset(2).Name),'\','/'));
shape_pcaBasis = h5read('./model2017-1_face12_nomouth.h5',replace(fullfile(shape_model.Name,shape_dataset(3).Name),'\','/'));
shape_pcaVariance = h5read('./model2017-1_face12_nomouth.h5',replace(fullfile(shape_model.Name,shape_dataset(4).Name),'\','/'));
%% load the expression model parameters
exp_model = info.Groups(3).Groups(1);
exp_dataset = exp_model.Datasets;
exp_mean = h5read('./model2017-1_face12_nomouth.h5',replace(fullfile(exp_model.Name,exp_dataset(1).Name),'\','/'));
exp_noiseVariance = h5read('./model2017-1_face12_nomouth.h5',replace(fullfile(exp_model.Name,exp_dataset(2).Name),'\','/'));
exp_pcaBasis = h5read('./model2017-1_face12_nomouth.h5',replace(fullfile(exp_model.Name,exp_dataset(3).Name),'\','/'));
exp_pcaVariance = h5read('./model2017-1_face12_nomouth.h5',replace(fullfile(exp_model.Name,exp_dataset(4).Name),'\','/'));
%% generate random face
%% classes: 10,000
%% scans for each class: 50
shapes=10000;
expressions=50;
save_folder = './TainData/';
if ~exist(save_folder,'dir') == 1
    mkdir(save_folder);
end
ClassNameFromat=400000000;
mean_alfa=zeros(199,1);
mean_beta=zeros(100,1);
Alfa = [mean_alfa  randn(199,shapes)];
Beta = [mean_beta  randn(100,expressions)];
for i=1:shapes+1
    alfa = Alfa(:,i);
    
    ShapeFolder = fullfile(save_folder,int2str(ClassNameFromat+i-1));
    if ~exist(ShapeFolder,'dir') == 1
        mkdir(ShapeFolder);
    end
    ScanNameFormat=000;
    for j=1:expressions+1
        fprintf('class [%d]/[%d]: expression[%d]/[%d]\n',i,shapes,j,expressions);
        beta = Beta(:,j);
        exp_beta = exp_pcaBasis' * (beta .* sqrt(exp_pcaVariance));
        shape_alfa = shape_pcaBasis' * (alfa .* sqrt(shape_pcaVariance));
        face = shape_mean + shape_alfa + exp_mean + exp_beta;       
        face=reshape(face,3,length(face)/3)';
        face = face - face(8157,:);
        ScanName=fullfile(ShapeFolder,num2str(ScanNameFormat+j-1,'%.3d'));
        bcName=[ScanName '.bc'];
        fid=fopen(bcName,'wb');
        fwrite(fid,face,'float');
        fclose(fid);
    end
end
