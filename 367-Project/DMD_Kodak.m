function MSE = DMD_Kodak(I, lambda,method,file_name )
%Bayer pattern is R,G,G,B
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

%get the R G B channels from the LH img
R_CH = I(:,:,1).*R;
G_CH = I(:,:,2).*G;
B_CH = I(:,:,3).*B;

pat = cat(3,R,G,B);

colorBayer = cat(3,R_CH,G_CH,B_CH);
bayer = R_CH + G_CH + B_CH;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set up problem and function handles
Afun = @(c,idx,x) idx(:,:,c).*x;

%use bicubic interpolation for Atfun
%AAtfun = @(x) reshape(Afun(Atfun(x)), [N 1]);
DtD = @(x) opDtx(opDx(x));
% noise parameter - standard deviation
sigma = 0.01;
% simulated measurements
%add Noise for EACH color channel 
bR = Afun(1,colorBayer,R) + sigma*(randn([m n])).*R;
bG = Afun(2,colorBayer,G) + sigma.*(randn([m n])).*G;
bB = Afun(3,colorBayer,B) + sigma.*(randn([m n])).*B;

b = cat(3,bR,bG,bB);

if(strcmp(method,'tv'))
    %SOLVE " minimize{I_DM} ||AI_DM-b||_2^2 + TV 
    IDMD = zeros(m,n,3);
    p=10;
    k = lambda/p;
    niter=20;
    Q = 1:niter;
    MSE_DMD = zeros(niter,3);
    PSNR_DMD=zeros(niter,3);
    for c = 1:3
    
        u = zeros(m,n,2);
        z = zeros(m,n,2);
        x = zeros(m,n,1);

        for i=1:niter
            %update x
            v = z-u;
            Atilde= @(x) reshape(AtfunDM(c,Afun(c,pat,reshape(x,[m n])))+p*DtD(reshape(x,[m n])),[m*n 1]); 
            btilde = reshape(AtfunDM(c,b(:,:,c)) + p*opDtx(v),[m*n 1]);
            x = pcg(Atilde, btilde, 10e-12, 50);
            Dx=opDx(reshape(x, [m n])); 
            v=Dx+u; % v update 
            v(v>k)=v(v>k)-k;
            v(v<-k)=v(v<-k)+k;
            v(abs(v)<=k)=0;
            z=v; % z update
            u=Dx+u-z; % u update
            x = reshape(x, [m n]);
            CH = I(:,:,c);
            MSE_DMD(i,c) = sum((CH(:)-x(:)).^2) /(m*n); 
            PSNR_DMD(i,c) = 10*log10(((max(CH(:)))^2)/MSE_DMD(i,c));


        end
        IDMD(:,:,c) = x;
    end





else    %SOLVE " minimize{I_DM} ||AI_DM-b||_2^2 + NLM 

    IDMD = zeros(m,n,3);
    lambda=0.001; %no bicubic
    p=10;
    k = lambda/p;   
    niter=20;
    Q = 1:niter;
    MSE = zeros(niter,3);
    PSNR=zeros(niter,3);
    for c=1:3

    u = zeros(m,n,2);
    z = zeros(m,n,2);
    x = zeros(n,m,2);

    for i=1:niter

        %update x
        v = z-u;
        Atilde=@(x) reshape(AtfunDM(c,Afun(c,pat,reshape(x,[m n])))+p.*DtD(reshape(x,[m n])),[m*n 1]); 
        btilde = reshape(AtfunDM(c,b(:,:,c)) + p.*opDtx(v),[m*n 1]);
        x =reshape(pcg(Atilde, btilde, 10e-12, 50),[m n]); 
        Dx=opDx(reshape(x, [m n])); 
        v=Dx+u; % v update for z 
        Options.filterstrength = sqrt(lambda/p);
        z=NLMF(v,Options); % update z with NLM
        u=Dx+u-z; % u update
        x = reshape(x, [m n]);
        CH = I(:,:,c);
        MSE(i,c) = sum((CH(:)-x(:)).^2) /(m*n); 
        PSNR(i,c) = 10*log10(((max(CH(:)))^2)/MSE(i,c));    
    end

    IDMD(:,:,c) = x;
    end
    
end

MSE = sum(MSE_DMD,2);
MSE = MSE(end);
%imshow(IDMD)
end

