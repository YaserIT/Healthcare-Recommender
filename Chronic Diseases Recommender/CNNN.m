function [cnn, ConMat]=CNNN(X,C)
for i=1:size(X,1)
trainD(:,:,:,i)=X(i,:);
end
% trainD(:,:,:,2)=X(2,:);
% trainD(:,:,:,3)=X(3,:);
% trainD(:,:,:,4)=X(4,:);
% trainD(:,:,:,5)=X(5,:);
% trainD(:,:,:,6)=X(6,:);
% trainD(:,:,:,7)=X(7,:);
% trainD(:,:,:,8)=X(8,:);
% trainD(:,:,:,9)=X(9,:);
targetD=categorical(C);

%% Define Network Architecture
% Define the convolutional neural network architecture.
layers = [
    imageInputLayer([27 1 1]) % 22X1X1 refers to number of features per sample
    convolution2dLayer(3,50,'Padding','same')
    reluLayer
    fullyConnectedLayer(500) % 384 refers to number of neurons in next FC hidden layer
    fullyConnectedLayer(500) % 384 refers to number of neurons in next FC hidden layer
    fullyConnectedLayer(2) % 2 refers to number of neurons in next output layer (number of output classes)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm',...
    'MaxEpochs',20, ...
    'Verbose',false,...
    'Plots','training-progress');

net = trainNetwork(trainD,targetD',layers,options);
predictedLabels = classify(net,trainD);
%%
ConMat=confusionmat(targetD,predictedLabels);
%%
for i=1:length(predictedLabels)
if ismember(predictedLabels(i,1),'1')
    cnn(i,1)=1;
else
    cnn(i,1)=2;
end
end
%%
TP=ConMat(1,1);
FP=ConMat(1,2);
TN=ConMat(2,1);
FN=ConMat(2,2);

Accuracy= (TP+TN)/(TP+TN+FP+FN);
Recall=TP/(TP+FN);
Precision=TP/(TP+FP);
Fmeasure=2/((1/Precision)+(1/Recall));