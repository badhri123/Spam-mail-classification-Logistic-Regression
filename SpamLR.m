%% Spam Mail Classification
% Note: The threshold was increased to reduce number of False % positives
% Loading and preprocessing Data

x=csvread('spam.csv');
xp=x(1:1813,:);      % Spam mails in particular( y=1 )  
load('po.mat');
po=po(:);
xp=xp(po,:); 
x=[x;xp];           % Straightification of data ( By duplicating Spam examples)
x=[x;x];            % Repeating the dataset
m=length(x);
load('p.mat');      % Splitting the Dataset into training and testing
p=p(:);

% Test Data: xunseen , yunseen


xunseen=x(p,:);
yunseen=xunseen(:,58);
xunseen(:,58)=[];
x(p,:)=[];
y=x(:,58);

% Initialization 

sh=[];
lambda=5*10^(-3);       % Regularization paramter
iter=2000;              % Number of Gradient Descent Iterations
k=10;                   % k-fold Cross validation
costs=[];
weight=[];
bfinal=[];
x(:,58)=[];

xperm=x;
[m,n]=size(xperm);
% Normalization
nn=1;
while nn<=n
    xperm(:,nn)=(xperm(:,nn)-mean(xperm(:,nn)))/std(xperm(:,nn));
    nn=nn+1;
end

yperm=y;
m=length(x);

load('shuf.mat');    % Shuffling the dataset
shuf=shuf(:);
xperm=xperm(shuf,:);
yperm=yperm(shuf);

j=1;
v=floor(m/k);        
while j<m
    
    x=xperm;
    y=yperm; 
 % Dividing the data into k-blocks and using each one of them as test set
 
    if j>=m-v             
        ptr=[j:m]';
        ptr=ptr(:);
    
    else
        ptr=[j:j+v-1]';
        ptr=ptr(:);
    end
        xtest=x(ptr,:);
        x(ptr,:)=[];
        ytest=y(ptr);
        y(ptr)=[];
        
        
        b=0;          % Bias term in Logistic Regression hypothesis
        % The dimensions are made as follow
        % X:nxm , Y:1xm  [ n=Number of feature, m=Number of examples]
        x=x';
        y=y';
        xtest=xtest';
        ytest=ytest';
        [n,m]=size(x);
        w=zeros(n,1);      % Weight vector 
        
        % Gradient Descent
        
        k=1;
        alpha=0.2;        % Learning rate
        L=[];             
        while k<=iter
            
            h=(w'*x) + b;   % Logistic regression hypothesis
            a=sigmoid(h);   % Activation function:Sigmoid
            
            L=[L;-(1/m)*(y*(log(a)') + (1-y)*(log(1-a))')   +   (lambda/(2*m))*sum(w.^2)]; % Cost
           
            % Computing Gradients
           
            dz=a-y;         
            dw=(1/m)*(x*dz');
            db=(1/m)*sum(dz);
            
            % Weight update
            w=w-(alpha*(dw+ (lambda/m)*sum(w)));
            b=b-(alpha*db);
            k=k+1;
        end
        
        % Testing
        
        htest=w'*xtest + b;
        atest=sigmoid(htest);
        atest=atest>=0.63;
        
        ero=atest-ytest;    % Misclassification Error in prediction
        fp=ero>0;           % False positives
        fn=ero<0;           % False negatives
        f=(sum(fp)/(length(xtest)))*100;
        fn=(sum(fn)/(length(xtest)))*100;
        er=sum(abs(ero));
        accuracy=(1-(er/length(xtest)))*100;
        
        % Testing on training data to check overfitting
        
        htrain=w'*x + b;
        atrain=sigmoid(htrain);
        atrain=atrain>=0.63;
        ert=abs(atrain-y);
        ert=sum(ert);
        train_accuracy=(1-(ert/length(x)))*100;
    
j=j+v;
% The weights and biases are stored for each block and the best one is
% selected based on the performance

weight=[weight,w];
bfinal=[bfinal,b];
costs=[costs,L];

hf=[f;fn;accuracy;train_accuracy];

sh=[sh,hf];

end

sh(:,end)=[];
[l1,l2]=min(sh(1,:));      % The block that gives minimum False positives is chosen
% Corresponding Weights,biases and Cost 
theta=weight(:,l2); 
bfin=bfinal(1,l2);
loss=costs(:,l2);
figure;
plot(loss);
title('Cost vs Number of iterations')

% Final testing ( With unseen data)

[an,bn]=size(xunseen);

% Normalization
u=1;
while u<=bn
    xunseen(:,u)=((xunseen(:,u)-mean(xunseen(:,u)))/std(xunseen(:,u)));
    u=u+1;
end

hfinal=theta'*(xunseen') + bfin;
afinal=sigmoid(hfinal);
afinal=afinal>=0.63;
erfinal=(afinal-(yunseen'));
fpp=(sum(erfinal>0)/length(xunseen))*100;
fnn=(sum(erfinal<0)/length(xunseen))*100;
accuracy_unseen=(1-(sum(abs(erfinal))/length(xunseen)))*100;

fprintf('False positives : %f ',fpp) 
display('%')
fprintf('False negatives : %f ',fnn)
display('%')
fprintf('Accuracy        : %f ',accuracy_unseen)
display('%')


