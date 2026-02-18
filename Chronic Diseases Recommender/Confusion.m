function [tp,fp,tn,fn]=Confusion(x,y,Classifier)
for i=1:length(x)
if(x(i)==min(x))
    x(i)=1;
else x(i)=2;end
if(y(i)==min(y))
    y(i)=1;
else y(i)=2;end
end  

tp=0;fp=0;tn=0;fn=0;
for i=1:length(x)
    if x(i)==1
        if y(i)==1
           tp=tp+1;
        elseif y(i)==2
            fp=fp+1;
        end
    elseif x(i)==2
        if y(i)==1
           fn=fn+1;
        elseif y(i)==2
            tn=tn+1;
        end
    end
end
%%
figure
subplot(2,1,1)
rectangle('Position',[0 0 4 4])
hold on
rectangle('Position',[0 0 2 2],'FaceColor',[1 0.13 0.5 0.3],'EdgeColor','k','LineWidth',2)
hold on
rectangle('Position',[2 0 2 2],'FaceColor',[0.13 1 0.5 0.3],'EdgeColor','k','LineWidth',2)
hold on
rectangle('Position',[0 2 2 2],'FaceColor',[0.13 1 0.5 0.3],'EdgeColor','k','LineWidth',2)
hold on
rectangle('Position',[2 2 2 2],'FaceColor',[1 0.13 0.5 0.3],'EdgeColor','k','LineWidth',2)
hold on
xt = [0.7 2.7 0.7 2.7];
yt = [1 1 3 3];
str = {'FN= ','TN= ','TP= ','FP= '};
text(xt,yt,str)
hold on
xt = [1 3 1 3];
yt = [1 1 3 3];
str = {num2str(fn),num2str(tn),num2str(tp),num2str(fp)};
text(xt,yt,str)
axis off
title(['Confusion Matrix of ', Classifier])
text(1.5,-0.15,'Predicted Class')
text(-0.4,2,sprintf('True \nClass'))  

subplot(2,1,2)

rectangle('Position',[0 0 4 4])
hold on
rectangle('Position',[0 0 2 2],'FaceColor',[1 0.13 0.5 0.3],'EdgeColor','k','LineWidth',2)
hold on
rectangle('Position',[2 0 2 2],'FaceColor',[0.13 1 0.5 0.3],'EdgeColor','k','LineWidth',2)
hold on
rectangle('Position',[0 2 2 2],'FaceColor',[0.13 1 0.5 0.3],'EdgeColor','k','LineWidth',2)
hold on
rectangle('Position',[2 2 2 2],'FaceColor',[1 0.13 0.5 0.3],'EdgeColor','k','LineWidth',2)
hold on
xt = [0.7 2.7 0.7 2.7];
yt = [1 1 3 3];
str = {'FN= ','TN= ','TP= ','FP= '};
text(xt,yt,str)
hold on
xt = [1 3 1 3];
yt = [1 1 3 3];
str = {num2str(fn/(tn+fn)),num2str(tn/(tn+fn)),num2str(tp/(tp+fp)),num2str(fp/(tp+fp))};
text(xt,yt,str)
axis off
text(1.5,-0.15,'Predicted Class')
text(-0.4,2,sprintf('True \nClass'))  


