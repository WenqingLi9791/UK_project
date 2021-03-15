for i=1:length(gain)-25        %gain is 501x1 vector
    input(i,:)=gain(i:i+23);   %input size is 476x24
    output(i,:)=gain(i+24);    %output size is 476x1 vector
end


maxx=max(gain);
minn=min(gain);

for i=1:11
    a(i)=minn+(i-1)*(maxx-minn)/10;
end

%t = discretize(prediction',[-Inf a Inf]);
%prediction_position=[t-1;t]';
 t = discretize(output,a);
 onehot=zeros(size(output,1),10);    %10 classes, 501 samples
 for i=1:(size(output,1))
     onehot(i,t(i))=1;   % onehot size is 476x10
 end