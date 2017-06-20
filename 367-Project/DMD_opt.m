I1=im2double(imread('kodim04.png'));
I2=im2double(imread('kodim19.png'));
I3=im2double(imread('kodim23.png'));
%%
lambda= [0.0001 0.001 0.002 0.004  0.006];
for k =1:length(lambda)
    
    MSE4(k) = DMD_Kodak(I1,lambda(k),'tv','');
    PSNR4(k) = 10*log10(((max(I1(:)))^2)/MSE4(k));
    MSE19(k) = DMD_Kodak(I2,lambda(k),'tv','');
    PSNR19(k) = 10*log10(((max(I2(:)))^2)/MSE19(k));
    MSE23(k) = DMD_Kodak(I3,lambda(k),'tv','');
    PSNR23(k) = 10*log10(((max(I3(:)))^2)/MSE23(k));
  
    
end

  
    save Kod4 lambda MSE4 PSNR4
    save Kod19 lambda MSE19 PSNR19
    save Kod23 lambda MSE23 PSNR23