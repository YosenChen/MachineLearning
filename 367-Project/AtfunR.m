function R_channel = AtfunR(R_CH, R)
%bicubic interp on x, for x a red bayer pattern 
[m, n] = size(R_CH);

if isequal(R_CH,zeros(m,n))
  
    R_channel = zeros(m,n);
  
else

R_CH_backup = R_CH;

R_CH( ~any(R_CH,2), : ) = []; %remove 0s
R_CH( :, ~any(R_CH,1) ) = [];  
R_x = 1:2:n;
R_y = 1:2:m;
[X,Y] = meshgrid(R_x,R_y);
out_x = 1:1:n;
out_y = 1:1:m;
[R_X, R_Y]=meshgrid(out_x,out_y); %create mesh grids
R_channel = interp2(X,Y,R_CH,R_X,R_Y,'cubic');
%R_channel = R_channel(1:m-1,1:n-1);%crop the nans
%R_channel = R_channel(2:end,2:end);
R_channel(isnan(R_channel(:)))=0;

R_channel(logical(1-R)) = 0; % zero-out all other pixels

% whos
if (R_channel(logical(R)) ~= R_CH_backup(logical(R)))
    sprintf('warning: bicubic interp changes the src data');
end

end

end

