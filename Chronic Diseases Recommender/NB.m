function [out]=NB(data,label,test);

x=unique(label);



for i=1:numel(x)
    
    m(i,:)=mean(data(label==x(i),:));
    s(i,:)=std(data(label==x(i),:),1);
    
end

for i=1:numel(x)
   
    mn=repmat(m(i,:),size(test,1),1);
    sn=repmat(s(i,:),size(test,1),1);
    
    gau=(1./(2*pi*sn.^2)).*exp(-((test-mn).^2)./(2.*sn.^2));
    
 
g(:,i)=sum(log(gau'))';
p(:,i)=prod(gau')';    
end

[val,class]=max(g');

out=(class-1)';

prob=p./repmat(sum(p,2),1,numel(x));

prob=prob;
