clear all
 
 
Is_fix_alpha = true;
 
 
%close all
warning('off','all')
datanames = {'banana', %1
    'breast_cancer', %2
    'titanic', %3
    'waveform', %4
    'german', %5
    'image', %6
    'pima_diabetes', %7
    'ijcnn1', %8
    'a9a', %9
    'diabetis', %10
    'circle', %11
    'xor', %12
    'dbmoon', %13
    'USPS3v5', %14
    'mnist2vother', %15
    'mnist2vother256', %16
    'mnist2vNo2',
    'mnist3v5',
    'USPS4vother',
    'pc_mac',
    'sat',
    'swissroll', %22
    'moon_rise', %23
    'small_data' %24
    'spam' %25
    'nodel'%26  
    };
%Errors=zeros(12,10);
%ROC=zeros(12,10);
%PR=zeros(12,10);
%addpath /Users/zhoum/Downloads/GBN-master/plot_images_tree
 
 
 
 
 
%inference = 'VB';
 
%for i=[1:6,8,9]
%for i=1:6
for Gibbs_VB=1:2
     if Gibbs_VB==1
        inference = 'Gibbs';
     else
         inference = 'VB';
     end
    
    
for i=26 %[1:9]
%for i=24
    if i<=6
        MaxTry=10
    else
        MaxTry=5
    end
    for randtry = 1 %1:MaxTry
        
       % clearvars -EXCEPT i datanames randtry Errors ACC MaxTry inference ML sample mu_Beta Sig_Beta chol_Sig_Beta mu_alpha c_v d_v
        
        
        [i,randtry]
        
        
        trial=randtry
        
        
        PolyaGammaTruncation = 5;
        
        
        
        Burnin  = 2500;
        Collection = 2500;
        
        
        
        CollectionStep = 10;
        PruneIdx_K = [];
        
        dataname = datanames{i}
        
        rng(randtry,'twister');
        addpath('data')
        addpath('liblinear-2.1/matlab')
        
        if i==24
            
            load small_data.mat
            features = [X_train',X_test' ]'; %n*p
            species = [y_train;y_test]+1;  %-1,1
            idx.train = 1:length(y_train);
            idx.test = length(y_train) + (1:length(y_test));
        elseif i==25
            X_train = load('/Users/mingzhangyin/Desktop/HIVE_experiments/BLR/rdata_xtr.txt');
            X_test= load('/Users/mingzhangyin/Desktop/HIVE_experiments/BLR/rdata_xte.txt');
            y_train= load('/Users/mingzhangyin/Desktop/HIVE_experiments/BLR/rdata_ytr.txt');
            y_test=load('/Users/mingzhangyin/Desktop/HIVE_experiments/BLR/rdata_yte.txt');
            X_train = X_train(:,2:3); %remove intercept here, which will be added later
            X_test = X_test(:,2:3);    
            features = [X_train',X_test' ]';
            species = ([y_train;y_test]*2-1).*(-1);  %-1,1
            idx.train = 1:length(y_train);
            idx.test = length(y_train) + (1:length(y_test));
        elseif i==26
            XX = load('/Users/mingzhangyin/Desktop/nodal/rdata_xtr.txt');
            XX = XX(:,2:6);  %remove intercept here, which will be added later
            X_train = XX(1:25,:);
            X_test = XX(26:53,:);
            Y = load('/Users/mingzhangyin/Desktop/nodal/rdata_ytr.txt');
            y_train = Y(1:25);
            y_test = Y(26:53);
            features = [X_train',X_test' ]';
            species = ([y_train;y_test]*2-1);  %-1,1
            idx.train = 1:length(y_train);
            idx.test = length(y_train) + (1:length(y_test));
        else
            [features,species,idx] = loaddata_SR(dataname,randtry);
            if i==8||i==9
                features=full(features);
                idx.train = randtry:10:size(features,1);
                idx.test = 1:size(features,1);
                idx.test(idx.train)=[];
            end
        end
        
        
        
        [Nall,V]=size(features);
        N = length(idx.train);
        
        X.train = features(idx.train,1:V)';
        X.test = features(idx.test,1:V)';
        [species_unique,~,species_label]=unique(species);
        Label.train = species_label(idx.train);
        Label.test = species_label(idx.test);
        S = length(species_unique); %the number of classes
        
        Yall = double(sparse(species_label,1:Nall,1,S,Nall)); %Class labels as one-hot vectors
        Y =  Yall(:,idx.train); %training class labels
        y=Y(1,:)';
        
        Xall = features';
        Xcovariate=[ones(1,size(X.train,2));X.train];
        Vcovariate=size(Xcovariate,1);
        
        switch inference
            case 'Gibbs'
                [ML,sample,LogLike_iter]= logistic_reg_Gibbs(Xcovariate,y,PolyaGammaTruncation,Burnin,Collection,CollectionStep,Is_fix_alpha);
                
                 PSI = [ones(1,size(Xall,2));Xall]'*ML.Beta;
                class_prob = exp(PSI-logOnePlusExp(PSI));
                PredictLabel = double(class_prob>0.5);
                train_error =1-nnz(PredictLabel(idx.train)'==Yall(1,idx.train))/length(idx.train);
                test_error = 1-nnz(PredictLabel(idx.test)'==Yall(1,idx.test))/length(idx.test);
                train_LogLike = mean(Yall(1,idx.train)'.*PSI(idx.train)-logOnePlusExp(PSI(idx.train)));
                test_LogLike = mean(Yall(1,idx.test)'.*PSI(idx.test)-logOnePlusExp(PSI(idx.test)));
                Error_Gibbs_ML=[train_error,test_error,train_LogLike,test_LogLike]
                
                num_sample = length(sample.Beta);
                class_prob_Gibbs=zeros(Nall,num_sample);
                for ii = 1:num_sample
                    
                    PSI = [ones(1,size(Xall,2));Xall]'*sample.Beta{ii};
                    class_prob_Gibbs(:,ii)= exp(PSI-logOnePlusExp(PSI));
                end
                class_prob_ave = min(max(mean(class_prob_Gibbs,2),eps),1-eps);
                PredictLabel = double(class_prob_ave>0.5);
                PSI = log(class_prob_ave)-log(1-class_prob_ave);
                train_error =1-nnz(PredictLabel(idx.train)'==Yall(1,idx.train))/length(idx.train);
                test_error = 1-nnz(PredictLabel(idx.test)'==Yall(1,idx.test))/length(idx.test);
                %train_LogLike = mean(Yall(1,idx.train)'.*log(class_prob_ave(idx.train))+(1-Yall(1,idx.train)').*log(1-class_prob_ave(idx.train)));
                %test_LogLike = mean(Yall(1,idx.test)'.*log(class_prob_ave(idx.test))+(1-Yall(1,idx.test)').*log(1-class_prob_ave(idx.test)));
                
                train_LogLike = mean(Yall(1,idx.train)'.*PSI(idx.train)-logOnePlusExp(PSI(idx.train)));
                test_LogLike = mean(Yall(1,idx.test)'.*PSI(idx.test)-logOnePlusExp(PSI(idx.test)));
                Error_Gibbs_ave=[train_error,test_error,train_LogLike,test_LogLike]
                
            case 'VB'
                maxIter = 5000;
                [mu_Beta,Sig_Beta,chol_Sig_Beta,mu_alpha,c_v,d_v] =  logistic_reg_VB(Xcovariate,y,maxIter,Is_fix_alpha);
                
                %maxIter = 5000;
               %[mu_Beta,Sig_Beta,chol_Sig_Beta,mu_alpha,c_v,d_v] =  logistic_reg_VB_diag(Xcovariate,y,maxIter,Is_fix_alpha);
 
                
                PSI = [ones(1,size(Xall,2));Xall]'*mu_Beta;
                class_prob = exp(PSI-logOnePlusExp(PSI));
                PredictLabel = double(class_prob>0.5);
                train_error =1-nnz(PredictLabel(idx.train)'==Yall(1,idx.train))/length(idx.train);
                test_error = 1-nnz(PredictLabel(idx.test)'==Yall(1,idx.test))/length(idx.test);
                train_LogLike = mean(Yall(1,idx.train)'.*PSI(idx.train)-logOnePlusExp(PSI(idx.train)));
                test_LogLike = mean(Yall(1,idx.test)'.*PSI(idx.test)-logOnePlusExp(PSI(idx.test)));
                Error_VB=[train_error,test_error,train_LogLike,test_LogLike]
                
                num_sample = Collection/CollectionStep;
                class_prob_VB=zeros(Nall,num_sample);
                Beta_VB_sample=zeros(Vcovariate,num_sample);
                for ii = 1:num_sample
                    Beta_VB_sample(:,ii) = chol_Sig_Beta*randn(Vcovariate,1) +mu_Beta;
                    PSI = [ones(1,size(Xall,2));Xall]'*Beta_VB_sample(:,ii);
                    class_prob_VB(:,ii)= exp(PSI-logOnePlusExp(PSI));
                end
                class_prob_ave = min(max(mean(class_prob_VB,2),eps),1-eps);
                PredictLabel = double(class_prob_ave>0.5);
                PSI = log(class_prob_ave)-log(1-class_prob_ave);
                train_error =1-nnz(PredictLabel(idx.train)'==Yall(1,idx.train))/length(idx.train);
                test_error = 1-nnz(PredictLabel(idx.test)'==Yall(1,idx.test))/length(idx.test);
                %train_LogLike = mean(Yall(1,idx.train)'.*log(class_prob_ave(idx.train))+(1-Yall(1,idx.train)').*log(1-class_prob_ave(idx.train)));
                %test_LogLike = mean(Yall(1,idx.test)'.*log(class_prob_ave(idx.test))+(1-Yall(1,idx.test)').*log(1-class_prob_ave(idx.test)));
                
                train_LogLike = mean(Yall(1,idx.train)'.*PSI(idx.train)-logOnePlusExp(PSI(idx.train)));
                test_LogLike = mean(Yall(1,idx.test)'.*PSI(idx.test)-logOnePlusExp(PSI(idx.test)));
                Error_VB_ave=[train_error,test_error,train_LogLike,test_LogLike];
        end
        
        
        
    end
    
end
end
 
disp('train error, test error, train LogLike, test LogLike')
 
Error_Gibbs_ML
Error_Gibbs_ave
 
Error_VB
Error_VB_ave
 
d1 = 2;
d2 = min(20,Vcovariate);
BetaMCMC=cell2mat(sample.Beta);
 
figure;
%subplot(1,2,1)
plot(mean(class_prob_Gibbs(idx.test,:),2),std(class_prob_Gibbs(idx.test,:),0,2),'bp')
hold on;
plot(mean(class_prob_VB(idx.test,:),2),std(class_prob_VB(idx.test,:),0,2),'ro')
 
xlabel('Sample mean of predicted probabilities')
ylabel('Sample standard deviation of predicted probabilities')
 
figure;
subplot(2,2,1);
scatter(Beta_VB_sample(d1,:),Beta_VB_sample(d2,:),'w.')
hold on;
 
scatter(BetaMCMC(d1,:),BetaMCMC(d2,:),'bp');
 
subplot(2,2,2);
scatter(BetaMCMC(d1,:),BetaMCMC(d2,:),'w.');
 
hold on;
scatter(Beta_VB_sample(d1,:),Beta_VB_sample(d2,:),'ro')
 
 
subplot(2,2,3);
[R,P] = corrcoef(BetaMCMC');
imagesc(R)
title('Estimated correlation matrix of \beta using MCMC samples')
 
subplot(2,2,4);
[R,P] = corrcoef(Beta_VB_sample');
imagesc(R)
%imagesc(diag(Sig_Beta)./sqrt((diag(Sig_Beta)*diag(Sig_Beta)')))
title('Estimated correlation matrix of \beta using VB')
 
%%
 
X_train = [ones(1,size(X.train,2));X.train]';
X_test = [ones(1,size(X.test,2));X.test]';
y_train = full(Yall(1,idx.train));
y_test = full(Yall(1,idx.test));
save('/Users/mingzhangyin/Desktop/HIVE_experiments/BLR/nodal.mat',...
    'X_train','X_test','y_train','y_test','Beta_VB_sample',...
    'BetaMCMC','mu_Beta','chol_Sig_Beta','Sig_Beta')
 
 
 
 
 
 

