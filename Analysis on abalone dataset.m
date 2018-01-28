% Loading the dataset %
data = xlsread('abalone data.xlsx');

% Constructing Binary Classification %
for i = 1:length(data)
    if data(i,9)<9
        data(i,9)=-1;
    else
       data(i,9) = 1;
    end
end
% 
% Splitting the data %
Ltrain = round(0.8*size(data,1));
Ytrain = data (1:Ltrain,1:8);
Ytest = data(Ltrain +1:end,1:8);
agetrain = data(1:Ltrain,9);
agetest = data(Ltrain+1:end,9);

% adding augmented vector of 1 in feature matrix %
Ytrain = [ones(size(Ytrain,1),1), Ytrain];
Ytest = [ones(size(Ytest,1),1), Ytest];

% Data Normalization for the training dataset
idx = find(agetrain(:,1) == -1);
Ytrain(idx, :) = -Ytrain(idx,:);

% Initialize the algorithm %
b = ones(size(Ytrain,1) ,1);
a = ones(size(Ytrain,2), 1)';
theta = 0.0001;
eta = 0.2;

% Gradient Searching Function %
[w_widrowHuff, K] = grad_descentWH(Ytrain, a, b, eta, theta);

%Do prediction for each testing sample %
n = size(Ytest, 1);
Lpred = [];
for i = 1:n 
    if Ytest(i,:)*w_widrowHuff' > 0
        Lpred(i) = 1;
    else 
        Lpred(i) = -1 
    end
end

cm = confusionmat(agetest, Lpred);
accuracy = (cm(1,1)+cm(2,2))/(cm(1,1)+cm(2,1)+cm(2,1)+cm(2,2));
