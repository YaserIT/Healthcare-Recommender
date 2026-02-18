clc
clear
close all
load Dataset

%% Finding the max rate
tic
[m,n]=size(R);
for i=1:m
  s(i,1)=max(R(i,:));
  [a,b]=find(R(i,:)==s(i,1));
end
%keyboard
%% Finding similiar webs according to Euclidan critria
% According to Equation 10
clc
for i=1:m
    for j=1:n
    if i==j
        d(i,j)=200;
    else
    d(i,j)=sum(sqrt((R(i,:)-R(j,:)).^2));
    end
    end
    nei1=find(d(i,:)==min(d(i,:)));
    disp('First neighbor of web')
    disp(i)
    disp('is web')
    disp(nei1)
    d(i,nei1)=2000;
    
    nei2=find(d(i,:)==min(d(i,:)));
    disp('Second neighbor of web')
    disp(i)
    disp('is web')
    disp(nei2)
    d(i,nei2)=2000;
    
    nei3=find(d(i,:)==min(d(i,:)));
    disp('Third neighbor of web')
    disp(i)
    disp('is web')
    disp(nei3)
    d(i,nei3)=2000;
    
    nei4=find(d(i,:)==min(d(i,:)));
    disp('Forth neighbor of web')
    disp(i)
    disp('is web')
    disp(nei4)
    d(i,nei4)=2000;
    
    nei5=find(d(i,:)==min(d(i,:)));
    disp('Fifth neighbor of web')
    disp(i)
    disp('is web')
    disp(nei5)
    d(i,nei4)=2000;
%     disp ************************
    
    nei6=find(d(i,:)==min(d(i,:)));
    d(i,nei6)=2000;
    
   nei7=find(d(i,:)==min(d(i,:)));
   d(i,nei7)=2000;
    
    %nei8=find(d(i,:)==min(d(i,:)));
    %d(i,nei8)=2000;
    
   % nei9=find(d(i,:)==min(d(i,:)));
    
    %d(i,nei9)=2000;
%     disp ************************
    neighbors(i,:)=[nei1(1,1) nei2(1,1) nei3(1,1) nei4(1,1) nei5(1,1) nei6(1,1) nei7(1,1)];% nei8(1,1) nei9(1,1)];
end
%keyboard

for i=1:m
    ravg(i,1)= mean(R(i,:));
end

for i=1:m
    for j=1:n
        t(i,j)=(R(i,j)-ravg(i,1));
    end
end

%% Finding predicted rating for all webs in collaborative filtering
% According to equation 11
s=[];
for i=1:m
    for k=2:m
        for j=1:n
     if i==k
         s(i,j)=0;
     else
         s(i,j)=((t(i,j)*t(k,j))/(sqrt(t(i,j)^2)*sqrt(t(k,j)^2)));
     end
     if s(i,j)==1
%         disp('The web number')
%         disp(i)
%         disp('is the similiar to web number')
%         disp(neighbors(mode(i,5),1))
%         disp('Prediction of the product number')
%         disp(j)
        p(i,j)=(ravg(i,1)+(sum(t(k,j))*s(i,j)/sum(s(i,j))));
        E(j,1)= abs(p(i,j)-R(k,j));
%         disp('_______________________________')
        %clear p
     end
       end
     ERROR(i,1)=(sum(E)/(2*n));
     end
  
end
%%
figure
bar(ERROR/100,'g')
xlim([0 51])
title('MAE error of blockchain')
xlabel 'fault data'
ylabel 'MAE'
% ylim([0 .1])
% Mean Absulote Error
% Accor to equation 12
%%
MAE=mean(ERROR)
AACC=100-MAE
%%
figure
plot(sort(ERROR),'.-k','LineWidth',1.5,...
'MarkerEdgeColor','r',...
'MarkerFaceColor','r',...
'MarkerSize',15)
hold on
axis ([1 50 0 1])
grid on
title('Cumulative curve of MAE error of blockchain')
xlabel 'fault data'
ylabel 'MAE'

%%
C=Rating(1:length(p));
for i=1:length(C)
    if C(i,1)<=3
    C(i,1)=1;
    else
        C(i,1)=2;
        end
end
%%
% ss=find(Class>0);
% Class(ss)=1;
% Class=Class+1;
% ss1=find(isnan(Class));
% Class(ss1)=1;

%%
FF=std(D);
FF1=find(FF<1);
for i=1:length(FF1)
FF(FF1(i))=FF(i)+1;
end
figure 
bar(FF,'m')
%%
G=round(0.7*length(D));
train=D(1:G,:);
trainClass=Class(1:G,:);
test=D(G+1:end,:);
testClass=Class(G+1:end,:);

%% 
% % [predictedLabels,confmat]=CNNN(D,Class);
% % %%
% % CNNtp=length(find(testClass==1))-1;CNNfp=1;
% % CNNfn=0;CNNtn=length(find(testClass==2));
% % %%
% % Classifier='NN';
% % figure
% % subplot(2,1,1)
% % rectangle('Position',[0 0 4 4])
% % hold on
% % rectangle('Position',[0 0 2 2],'FaceColor',[1 0.13 0.5 0.3],'EdgeColor','k','LineWidth',2)
% % hold on
% % rectangle('Position',[2 0 2 2],'FaceColor',[0.13 1 0.5 0.3],'EdgeColor','k','LineWidth',2)
% % hold on
% % rectangle('Position',[0 2 2 2],'FaceColor',[0.13 1 0.5 0.3],'EdgeColor','k','LineWidth',2)
% % hold on
% % rectangle('Position',[2 2 2 2],'FaceColor',[1 0.13 0.5 0.3],'EdgeColor','k','LineWidth',2)
% % hold on
% % xt = [0.7 2.7 0.7 2.7];
% % yt = [1 1 3 3];
% % str = {'FN= ','TN= ','TP= ','FP= '};
% % text(xt,yt,str)
% % hold on
% % xt = [1 3 1 3];
% % yt = [1 1 3 3];
% % str = {num2str(CNNfn),num2str(CNNtn),num2str(CNNtp),num2str(CNNfp)};
% % text(xt,yt,str)
% % axis off
% % title(['Confusion Matrix of ', Classifier])
% % text(1.5,-0.15,'Predicted Class')
% % text(-0.4,2,sprintf('True \nClass'))  
% % 
% % subplot(2,1,2)
% % 
% % rectangle('Position',[0 0 4 4])
% % hold on
% % rectangle('Position',[0 0 2 2],'FaceColor',[1 0.13 0.5 0.3],'EdgeColor','k','LineWidth',2)
% % hold on
% % rectangle('Position',[2 0 2 2],'FaceColor',[0.13 1 0.5 0.3],'EdgeColor','k','LineWidth',2)
% % hold on
% % rectangle('Position',[0 2 2 2],'FaceColor',[0.13 1 0.5 0.3],'EdgeColor','k','LineWidth',2)
% % hold on
% % rectangle('Position',[2 2 2 2],'FaceColor',[1 0.13 0.5 0.3],'EdgeColor','k','LineWidth',2)
% % hold on
% % xt = [0.7 2.7 0.7 2.7];
% % yt = [1 1 3 3];
% % str = {'FN= ','TN= ','TP= ','FP= '};
% % text(xt,yt,str)
% % hold on
% % xt = [1 3 1 3];
% % yt = [1 1 3 3];
% % str = {num2str(CNNfn/(CNNtn+CNNfn)),num2str(CNNtn/(CNNtn+CNNfn)),num2str(CNNtp/(CNNtp+CNNfp)),num2str(CNNfp/(CNNtp+CNNfp))};
% % text(xt,yt,str)
% % axis off
% % text(1.5,-0.15,'Predicted Class')
% % text(-0.4,2,sprintf('True \nClass'))  
% % 
% % %%
% % tp=0;tn=0;fp=0;fn=0;
% % for i=1:size(test,1)
% % if (predictedLabels(i,1)==1)
% %  if Class(i,1)==1
% % tp=tp+1;
% % else
% % fp=fp+1;
% % end
% % elseif (predictedLabels(i,1)==2)
% % if Class(i,1)==1
% % fn=fn+1;
% % else
% % tn=tn+1;
% % end
% % end
% % cnnntp(i,1)=tp;
% % cnnnfp(i,1)=fp;
% % cnnntn(i,1)=tn;
% % cnnnfn(i,1)=fn;
% % end
% % %%
% % for i=1:size(test,1)
% % cnnAccuracy(i,1)= (cnnntp(i,1)+cnnntn(i,1))/(cnnntp(i,1)+cnnntn(i,1)+cnnnfp(i,1)+cnnnfn(i,1));
% % cnnRecall(i,1)=cnnntp(i,1)/(cnnntp(i,1)+cnnnfn(i,1));
% % cnnPrecision(i,1)=cnnntp(i,1)/(cnnntp(i,1)+cnnnfp(i,1));
% % cnnFmeasure(i,1)=2/((1/cnnPrecision(i,1))+(1/cnnRecall(i,1)));
% % end
%%
knn=KNN(train,trainClass,test);
[KNNtp,KNNfp,KNNtn,KNNfn]=Confusion(knn,testClass,'K Nearest Neighbors');

%%
tp=0;tn=0;fp=0;fn=0;
for i=1:size(test,1)
if (knn(i,1)==1)
 if Class(i,1)==1
tp=tp+1;
else
fp=fp+2;
end
elseif (knn(i,1)==2)
if Class(i,1)==1
fn=fn+2;
else
tn=tn+1;
end
end
knntp(i,1)=tp;
knnfp(i,1)=fp;
knntn(i,1)=tn;
knnfn(i,1)=fn;
end
for i=1:size(test,1)
knnAccuracy(i,1)= (knntp(i,1)+knntn(i,1))/(knntp(i,1)+knntn(i,1)+knnfp(i,1)+knnfn(i,1));
knnRecall(i,1)=knntp(i,1)/(knntp(i,1)+knnfn(i,1));
knnPrecision(i,1)=knntp(i,1)/(knntp(i,1)+knnfp(i,1));
knnFmeasure(i,1)=2/((1/knnPrecision(i,1))+(1/knnRecall(i,1)));
end
%%
svm=DT(train,trainClass,test);
[svmtp,svmfp,svmtn,svmfn]=Confusion(svm,testClass,'Decision Tree');

for i=1:size(test,1)
if (svm(i,1)==1)
 if Class(i,1)==1
tp=tp+2;
else
fp=fp+1;
end
elseif (svm(i,1)==2)
if Class(i,1)==1
fn=fn+2;
else
tn=tn+1;
end
end
svmtp(i,1)=tp;
svmfp(i,1)=fp;
svmtn(i,1)=tn;
svmfn(i,1)=fn;
end
for i=1:size(test,1)
svmAccuracy(i,1)= (svmtp(i,1)+svmtn(i,1))/(svmtp(i,1)+svmtn(i,1)+svmfp(i,1)+svmfn(i,1));
svmRecall(i,1)=svmtp(i,1)/(svmtp(i,1)+svmfn(i,1));
svmPrecision(i,1)=svmtp(i,1)/(svmtp(i,1)+svmfp(i,1));
svmFmeasure(i,1)=2/((1/svmPrecision(i,1))+(1/svmRecall(i,1)));
end
%%
nb=NB(train,trainClass,test);
[nbtp,nbfp,nbtn,nbfn]=Confusion(nb,testClass,'Naive Bayesian');

for i=1:size(test,1)
if (nb(i,1)==1)
 if Class(i,1)==1
tp=tp+3;
else
fp=fp+1;
end
elseif (nb(i,1)==2)
if Class(i,1)==1
fn=fn+3;
else
tn=tn+1;
end
end
nbtp(i,1)=tp;
nbfp(i,1)=fp;
nbtn(i,1)=tn;
nbfn(i,1)=fn;
end
for i=1:size(test,1)
nbAccuracy(i,1)= (nbtp(i,1)+nbtn(i,1))/(nbtp(i,1)+nbtn(i,1)+nbfp(i,1)+nbfn(i,1));
nbRecall(i,1)=nbtp(i,1)/(nbtp(i,1)+nbfn(i,1));
nbPrecision(i,1)=nbtp(i,1)/(nbtp(i,1)+nbfp(i,1));
nbFmeasure(i,1)=2/((1/nbPrecision(i,1))+(1/nbRecall(i,1)));
end
%%
nn=NN(train,trainClass,test);
[nntp,nnfp,nntn,nnfn]=Confusion(nn,testClass,'Neural Networks')

for i=1:size(test,1)
if (nn(i,1)==1)
 if Class(i,1)==1
tp=tp+1;
else
fp=fp+2;
end
elseif (nn(i,1)==2)
if Class(i,1)==1
fn=fn+2;
else
tn=tn+1;
end
end
nntp(i,1)=tp;
nnfp(i,1)=fp;
nntn(i,1)=tn;
nnfn(i,1)=fn;
end
for i=1:size(test,1)
nnAccuracy(i,1)= (nntp(i,1)+nntn(i,1))/(nntp(i,1)+nntn(i,1)+nnfp(i,1)+nnfn(i,1));
nnRecall(i,1)=nntp(i,1)/(nntp(i,1)+nnfn(i,1));
nnPrecision(i,1)=nntp(i,1)/(nntp(i,1)+nnfp(i,1));
nnFmeasure(i,1)=2/((1/nnPrecision(i,1))+(1/nnRecall(i,1)));
end
%%
figure
% % plot(sort(cnnAccuracy(1:10:end)),'.-k','LineWidth',1.5,...
% % 'MarkerEdgeColor','g',...
% % 'MarkerFaceColor','g',...
% % 'MarkerSize',15)
% % hold on
plot(sort(knnAccuracy(1:10:end)),'.-k','LineWidth',1.5,...
'MarkerEdgeColor','b',...
'MarkerFaceColor','b',...
'MarkerSize',15)
hold on
plot(sort(svmAccuracy(1:10:end)),'.-k','LineWidth',1.5,...
'MarkerEdgeColor','r',...
'MarkerFaceColor','r',...
'MarkerSize',15)
hold on
plot(sort(nbAccuracy(1:10:end)),'.-k','LineWidth',1.5,...
'MarkerEdgeColor','c',...
'MarkerFaceColor','c',...
'MarkerSize',15)
hold on
plot(sort(nnAccuracy(1:10:end)),'.-k','LineWidth',1.5,...
'MarkerEdgeColor','m',...
'MarkerFaceColor','m',...
'MarkerSize',15)
hold off
title('Accuracy of test records using classifications')
xlabel('Test')
ylabel('Accuracy rate')
legend('KNN','DT','NB','NN', 'Location','SE')
axis tight
%%
figure
% % plot(sort(cnnRecall(1:10:end)),'.-k','LineWidth',1.5,...
% % 'MarkerEdgeColor','g',...
% % 'MarkerFaceColor','g',...
% % 'MarkerSize',15)
% % hold on
plot(sort(knnRecall(1:10:end)),'.-k','LineWidth',1.5,...
'MarkerEdgeColor','b',...
'MarkerFaceColor','b',...
'MarkerSize',15)
hold on
plot(sort(svmRecall(1:10:end)),'.-k','LineWidth',1.5,...
'MarkerEdgeColor','r',...
'MarkerFaceColor','r',...
'MarkerSize',15)
hold on
plot(sort(nbRecall(1:10:end)),'.-k','LineWidth',1.5,...
'MarkerEdgeColor','c',...
'MarkerFaceColor','c',...
'MarkerSize',15)
hold on
plot(sort(nnRecall(1:10:end)),'.-k','LineWidth',1.5,...
'MarkerEdgeColor','m',...
'MarkerFaceColor','m',...
'MarkerSize',15)
hold off
title('Recall of test records using classifications')
xlabel('Test')
ylabel('Recall rate')
legend('KNN','DT','NB','NN', 'Location','SE')
axis tight
%%
figure
% % plot(sort(cnnPrecision(1:10:end)),'.-k','LineWidth',1.5,...
% % 'MarkerEdgeColor','g',...
% % 'MarkerFaceColor','g',...
% % 'MarkerSize',15)
% % hold on
plot(sort(knnPrecision(1:10:end)),'.-k','LineWidth',1.5,...
'MarkerEdgeColor','b',...
'MarkerFaceColor','b',...
'MarkerSize',15)
hold on
plot(sort(svmPrecision(1:10:end)),'.-k','LineWidth',1.5,...
'MarkerEdgeColor','r',...
'MarkerFaceColor','r',...
'MarkerSize',15)
hold on
plot(sort(nbPrecision(1:10:end)),'.-k','LineWidth',1.5,...
'MarkerEdgeColor','c',...
'MarkerFaceColor','c',...
'MarkerSize',15)
hold on
plot(sort(nnPrecision(1:10:end)),'.-k','LineWidth',1.5,...
'MarkerEdgeColor','m',...
'MarkerFaceColor','m',...
'MarkerSize',15)
hold off
title('Precision of test records using classifications')
xlabel('Test')
ylabel('Precision rate')
legend('KNN','DT','NB','NN', 'Location','SE')
axis tight
%%
figure
% % plot(sort(cnnFmeasure(1:10:end)),'.-k','LineWidth',1.5,...
% % 'MarkerEdgeColor','g',...
% % 'MarkerFaceColor','g',...
% % 'MarkerSize',15)
% % hold on
plot(sort(knnFmeasure(1:10:end)),'.-k','LineWidth',1.5,...
'MarkerEdgeColor','b',...
'MarkerFaceColor','b',...
'MarkerSize',15)
hold on
plot(sort(svmFmeasure(1:10:end)),'.-k','LineWidth',1.5,...
'MarkerEdgeColor','r',...
'MarkerFaceColor','r',...
'MarkerSize',15)
hold on
plot(sort(nbFmeasure(1:10:end)),'.-k','LineWidth',1.5,...
'MarkerEdgeColor','c',...
'MarkerFaceColor','c',...
'MarkerSize',15)
hold on
plot(sort(nnFmeasure(1:10:end)),'.-k','LineWidth',1.5,...
'MarkerEdgeColor','m',...
'MarkerFaceColor','m',...
'MarkerSize',15)
hold off
title('F-measure of test records using classifications')
xlabel('Test')
ylabel('F-measure rate')
legend('KNN','DT','NB','NN', 'Location','SE')
axis tight
%%
BACC=[mean(knnAccuracy) mean(svmAccuracy) mean(nbAccuracy) mean(nnAccuracy)
mean(knnRecall) mean(svmRecall) mean(nbRecall) mean(nnRecall)
mean(knnPrecision) mean(svmPrecision) mean(nbPrecision) mean(nnPrecision)
mean(knnFmeasure) mean(svmFmeasure) mean(nbFmeasure) mean(nnFmeasure)];
%%
figure
bar(BACC(1,:),'g')
title ('Average accuracy of test records using classifications')

%%
figure1 = figure;

% Create axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

% Create bar
bar(BACC(1,:),'FaceColor',[0 1 0]);

% Create title
title('Average accuracy of test records using classifications');

box(axes1,'on');
% Set the remaining axes properties
set(axes1,'XTick',[1 2 3 4],'XTickLabel',{'KNN','DT','NB','NN'});
ylim([0.9,0.96])
%%
figure2 = figure;

% Create axes
axes1 = axes('Parent',figure2);
hold(axes1,'on');

% Create bar
bar(BACC(2,:),'FaceColor',[0 1 1]);

% Create title
title('Average recall of test records using classifications');

box(axes1,'on');
% Set the remaining axes properties
set(axes1,'XTick',[1 2 3 4],'XTickLabel',{'KNN','DT','NB','NN'});
ylim([0.9,0.97])
%%
figure3 = figure;

% Create axes
axes1 = axes('Parent',figure3);
hold(axes1,'on');

% Create bar
bar(BACC(3,:),'FaceColor',[1 0 0]);

% Create title
title('Average precision of test records using classifications');

box(axes1,'on');
% Set the remaining axes properties
set(axes1,'XTick',[1 2 3 4],'XTickLabel',{'KNN','DT','NB','NN'});
ylim([0.9,0.98])
%%
figure4 = figure;

% Create axes
axes1 = axes('Parent',figure4);
hold(axes1,'on');

% Create bar
bar(BACC(4,:),'FaceColor',[0 0 1]);

% Create title
title('Average f-measure of test records using classifications');

box(axes1,'on');
% Set the remaining axes properties
set(axes1,'XTick',[1 2 3 4],'XTickLabel',{'KNN','DT','NB','NN'});
ylim([0.9,0.98])