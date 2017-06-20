%%
%read in images
for k = 1:9
 
    imgs{k}= im2double(imread(strcat('kodim0', num2str(k),'.png')));
    
end

for k = 10:24
 
    imgs{k}= im2double(imread(strcat('kodim', num2str(k),'.png')));
    
end
%%
%find opt lambda and min MSE
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
%%

%run the opt lambda for all 24 images
for i = 1:length(imgs)
    I = imgs{i};
    save_file = strcat('kodim_opt_',num2str(i),'.png');
    MSE(i) = DMD_Kodak(imgs{i},best_lambda,'tv',save_file);
    PSNR(i) = 10*log10(((max(I(:)))^2)/MSE(i));
end

save ADMM_TV_ALL MSE PSNR
   

