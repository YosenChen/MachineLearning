%%
%read in images
for i = 1:9
 
    imgs{i}= im2double(imread(strcat('kodim0', num2str(i),'.png')));
    
end

for i = 10:24
 
    imgs{i}= im2double(imread(strcat('kodim', num2str(i),'.png')));
    
end
%%
best_lambda=0;
best_MSE=0;
for k = 1:3
    k
    S=0;
    %randomly generated a lambda
    lambda = 0.002 + abs(normrnd(0,0.002));
    for i = 1:5
    
        MSE(i) = DMD_Kodak(imgs{i},lambda,'tv','');
    
    end
    
    S=sum(MSE);
    if(k==1)
       best_MSE=S;
       best_lambda=lambda;
    end
    if (S< best_MSE)
        best_MSE=S;
        best_lambda=lambda;
    end
    
end

