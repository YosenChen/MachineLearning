%%optimize 3 images from kodak set
I1=im2double(imread('kodim04.png'));
I2=im2double(imread('kodim19.png'));
I3=im2double(imread('kodim23.png'));

%%

lambda = [0.0001 0.0003 0.005 0.0007 0.0009];
for k = 1:length(lambda)
   
  
   
   
        %save_file = strcat('kodim_',num2str(lambda(k)),'_', num2str(i),'.png');
        MSE(k) = DMD_Kodak(I1,lambda(k),'tv','');
   
    
end
I1_opt =  DMD_Kodak(I1,lambda(1),'tv','Kod04_opt.png');

save Kod4 lambda MSE


lambda = [0.0001 0.0003 0.005 0.0007 0.0009];
for k = 1:length(lambda)
   
   
        %save_file = strcat('kodim_',num2str(lambda(k)),'_', num2str(i),'.png');
        MSE(k) = DMD_Kodak(I2,lambda(k),'tv','');
   
    
end
I2_opt =  DMD_Kodak(I2,lambda(1),'tv','Kod19_opt.png');
save Kod19 lambda MSE


lambda = [0.0001 0.0003 0.005 0.0007 0.0009];
for k = 1:length(lambda)
  
   
   
        %save_file = strcat('kodim_',num2str(lambda(k)),'_', num2str(i),'.png');
        MSE(k) = DMD_Kodak(I3,lambda(k),'tv','');
   
    
end
I3_opt =  DMD_Kodak(I3,lambda(1),'tv','Kod23_opt.png');

save Kod23 lambda MSE

