%%
% resolution of image
% load image
I = im2double(imread('FluorescentCells.jpg'));
%I = im2double(imread('lighthouse.png'));


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

imwrite(R_CH, 'R_CH.png');
imwrite(G_CH, 'G_CH.png');
imwrite(B_CH, 'B_CH.png');

colorBayer = cat(3,R_CH,G_CH,B_CH);
imwrite(colorBayer, 'colorBayer.png');
bayer = R_CH + G_CH + B_CH;
imwrite(bayer, 'bayer.png');
%Bayer = bayer(194:257,100:163,:);
%figure,imshow(Bayer)


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% set up problem and function handles

Afun    = @(idx,x) idx.*x;
%use bicubic interpolation for Atfun
%AAtfun  = @(x) reshape(Afun(Atfun(x)), [N 1]);
DtD = @(x) opDtx(opDx(x));
% noise parameter - standard deviation
sigma = 0;

% simulated measurements
%add Noise for EACH color channel 
bR   = Afun(R_CH,R) + sigma.*(randn([m n])).*R;
bG   = Afun(G_CH,G) + sigma.*(randn([m n])).*G;
bB   = Afun(B_CH,B) + sigma.*(randn([m n])).*B;


%%
%SOLVE " minimize{I_SR} ||AI_SR-b||_2^2 + TV 
%b.) ADMM + TV
% deconvolution using ADMM here

%xR = ones(n,m,1);
%[M, N] = size(AtfunR(Afun(R,xR)));

niter = 20;
res = zeros(niter,1);
u = zeros(m,n,2);
z = zeros(m,n,2);
xR = zeros(m,n,1);
lambda = 0.002; % Atfun w/o bicubic interp2
% lambda = 0.025; % Atfun w/ bicubic interp2
p=10;
k = lambda/p;

for i=1:niter

  %update x
  v = z-u;
  %temp = p*DtD(xR);
  %temp = temp(1:m-1,1:n-1);%crop the nans
  %temp = temp(2:end,2:end);
  Atilde=@(x) reshape(AtfunR(Afun(R,reshape(x,[m n])), R)+p*DtD(reshape(x,[m n])),[m*n 1]); 
  %temp = p*opDtx(v);
  %temp = temp(1:m-1,1:n-1);%crop the nans
  %temp = temp(2:end,2:end);
  btilde = reshape(AtfunR(bR, R) + p*opDtx(v),[m*n 1]);
  xR = pcg(Atilde, btilde, 10e-12, 50); 
  Dx=opDx(reshape(xR, [m, n]));
  v=Dx+u; % v update 
  v(v>k)=v(v>k)-k;
  v(v<-k)=v(v<-k)+k;
  v(abs(v)<=k)=0;
  z=v; % z update
  u=Dx+u-z; % u update
  if (mod(i, 5)==0)
    imwrite(reshape(xR,[m n]), sprintf('xR_iter%d.png', i));
    imwrite(reshape(btilde, [m n]), sprintf('btilde_iter%d.png', i));
  end
end
imwrite(reshape(xR,[m n]), 'xR_final.png');

%%
%SOLVE " minimize{I_SR} ||AI_SR-b||_2^2 + NLM 
%c.) ADMM + NLM
[m, n] = size(I);

niter = 20;
res = zeros(niter,1);
u = zeros(m,n,1);
z = zeros(m,n,1);
x = zeros(n,m,1);
lambda = 0.1;
p=10;
k = lambda/p;
K=eye(m,n);
PSNR=zeros(1,niter);
Q = 1:niter;
for i=1:niter

  %update x
  v = z-u;
  Atilde=@(x) reshape(Atfun(Afun(reshape(x,[m n])))+p.*reshape(x,[m n]),[4096 1]); 
  btilde = reshape(Atfun(b) + p.*v,[4096 1]);
  x =reshape(pcg(Atilde, btilde, 10e-12, 20),[m n]); 
 
  v=x+u; % v update for z 
  Options.filterstrength = sqrt(lambda/p);
  z=NLMF(v,Options); % update z with NLM
  u=x+u-z; % u update
  mse = sum((I(:)-x(:)).^2) /(size(I,1)*size(I,2)); 
  PSNR(i) = 10*log10((max(I(:)))^2/mse); 
 

end
imshow(x)
figure,plot(Q,PSNR)
ylabel('PSNR')
xlabel('Iteration')
axis([0 20 0 45])
title('PSNR')
mse = sum((I(:)-x(:)).^2) /(size(I,1)*size(I,2)); 
