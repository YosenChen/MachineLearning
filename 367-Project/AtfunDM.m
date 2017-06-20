function Channel = AtfunDM( c, CH )
[m, n] = size(CH);

if  isequal(CH,zeros(m,n))
    
    Channel = zeros(m,n);    
else

    if c==1 %red channel
        
      CH( ~any(CH,2), : ) = []; %remove 0s
      CH( :, ~any(CH,1) ) = [];  
      R_x = 1:2:n;
      R_y = 1:2:m;
      [X,Y] = meshgrid(R_x,R_y);
      out_x = 1:1:n;
      out_y = 1:1:m;
      [R_X, R_Y]=meshgrid(out_x,out_y); %create mesh grids
      Channel = interp2(X,Y,CH,R_X,R_Y,'cubic');
      Channel(isnan(Channel(:)))=0;
    
      R=toeplitz(mod(1:m,2),mod(1:n,2));
      for i=1:m
       if (rem(i,2)==0)
          R(i,:)=0; 
       end
      end
      %no bicubic
      Channel(logical(1-R)) = 0; % zero-out all other pixels
    
    elseif c==3 %blue channel
    
      CH( ~any(CH,2), : ) = []; %remove 0s
      CH( :, ~any(CH,1) ) = [];  
      B_x = 2:2:n;
      B_y = 2:2:m;
      [X,Y] = meshgrid(B_x,B_y);
      out_x = 1:1:n;
      out_y = 1:1:m;
      [B_X, B_Y]=meshgrid(out_x,out_y); %create mesh grids
      Channel = interp2(X,Y,CH,B_X,B_Y,'cubic');
      Channel(isnan(Channel(:)))=0;


      %Blue pattern
      B=toeplitz(mod(1:m,2),mod(1:n,2));
      for i=1:m
        if (rem(i,2)~=0)
          B(i,:)=0; 
        end
      end

      %no bicubic
      Channel(logical(1-B)) = 0; % zero-out all other pixels

    
    
    else %green channel
    
      %shift up
      Up=circshift(CH,[-1 0]);
       
      %shift down
      Down=circshift(CH,[1 0]);

      %shift left
      Left=circshift(CH,[0 -1]);
    
      %shift right
      Right=circshift(CH,[0 1]);

      Avg = (Up+ Down + Left + Right)/4;
      Avg(1,:)=0;
      Avg(end,:)=0;
      Avg(:,1)=0;
      Avg(:,end)=0;
      Channel = CH + Avg;
      Channel(isnan(Channel(:)))=0;

      %green pattern
      G=fliplr(toeplitz(mod(1:m,2),mod(1:n,2)));
      %no bicubic
      Channel(logical(1-G)) = 0; % zero-out all other pixels
    
    end
end
        
end

