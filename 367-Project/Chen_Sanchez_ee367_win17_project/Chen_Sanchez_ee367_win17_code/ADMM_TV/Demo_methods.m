 ] 
 %Benchmark with Bilinear and Malvar
%read in images
for i = 1:9
 
    imgs{i}= im2double(imread(strcat('kodim0', num2str(i),'.png')));
    
end

for i = 10:24
 
    imgs{i}= im2double(imread(strcat('kodim', num2str(i),'.png')));
    
end
%%
%Interpolation
for k = 1:length(imgs)
    I=imgs{k};
    [m,n,z] = size(I);
    %extract the 3 channels from LH using the Bayer patter
    %Red pattern
    R=toeplitz(mod(1:m,2),mod(1:n,2));
    for i=1:m
        if (rem(i,2)==0)
            R(i,:)=0; 
        end
    end

    %Blue pattern
    B=toeplitz(mod(1:m,2),mod(1:n,2));
    for i=1:m
        if (rem(i,2)~=0)
            B(i,:)=0; 
        end
    end

    %green pattern
    G=fliplr(toeplitz(mod(1:m,2),mod(1:n,2)));

    sigma = 0.01;
    % simulated measurements
    %add Noise for EACH color channel 
    R_CH = I(:,:,1).*R + sigma*(randn([m n])).*R; 
    G_CH = I(:,:,2).*G +  sigma.*(randn([m n])).*G;
    B_CH = I(:,:,3).*B + sigma.*(randn([m n])).*B;
    bayer = R_CH + G_CH + B_CH;
  
    save_file = strcat('kodim', num2str(k),'.png');
    Demo = InterpDemo(bayer,save_file);
    for c =1:3
        x = Demo(:,:,c);
        CH = I(:,:,c);
        MSE(1,c) = sum((CH(:)-x(:)).^2) /(m*n); 
    end
    MSE_Interp(k)=sum(MSE,2)./3;
    PSNR_Interp(k)=10*log10(((max(I(:)))^2)/MSE_Interp(k));

end
save InterpData MSE_Interp PSNR_Interp
%%
for k =1:length(imgs)
    I=imgs{k};
    [m, n, z] = size(I);
    %extract the 3 channels from LH using the Bayer patter
    %Red pattern
    R=toeplitz(mod(1:m,2),mod(1:n,2));
    for i=1:m
        if (rem(i,2)==0)
            R(i,:)=0; 
        end
    end

    %Blue pattern
    B=toeplitz(mod(1:m,2),mod(1:n,2));
    for i=1:m
        if (rem(i,2)~=0)
            B(i,:)=0; 
        end
    end

    %green pattern
    G=fliplr(toeplitz(mod(1:m,2),mod(1:n,2)));
 	sigma = 0.01;
    % simulated measurements
    %add Noise for EACH color channel 
    %get the R G B channels from the LH img
    R_CH = I(:,:,1).*R + sigma*(randn([m n])).*R; 
    G_CH = I(:,:,2).*G +  sigma.*(randn([m n])).*G;
    B_CH = I(:,:,3).*B + sigma.*(randn([m n])).*B;
    
    bayer = R_CH + G_CH + B_CH;
    save_file = strcat('kodim', num2str(k),'.png');
    malvar = Malvar(bayer,save_file);

    for c =1:3
        x = malvar(:,:,c);
        CH = I(:,:,c);
        MSE(1,c) = sum((CH(:)-x(:)).^2) /(m*n); 
      
    end

    MSE_Malv(k)=sum(MSE,2)./3;
    PSNR_Malv(k)=10*log10(((max(I(:)))^2)/MSE_Malv(k));

end
save MalvData MSE_Malv PSNR_Malv
