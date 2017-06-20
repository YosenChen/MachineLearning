function color_out = demosaic_linear(bayer_in)

    [nrow, ncol] = size(bayer_in);
    R = bayer_in;
    G = bayer_in;
    B = bayer_in;
    % set 0's for R channel, can be verified by showing R(1:10, 1:10)
    R(2:2:end, :) = 0;
    R(1:2:end, 2:2:end) = 0;
    % set 0's for B channel, can be verified by showing B(1:10, 1:10)
    B(1:2:end, :) = 0;
    B(2:2:end, 1:2:end) = 0;
    % set 0's for G channel
    G(1:2:end, 1:2:end) = 0;
    G(2:2:end, 2:2:end) = 0;
    
    % fill in R
    for r = 2:(nrow-1)
        for c = 2:(ncol-1)
            if (mod(r, 2)==1 & mod(c, 2)==0)
                R(r, c) = R(r, c-1)*0.5 + R(r, c+1)*0.5;
            elseif (mod(r, 2)==0 & mod(c, 2)==1)
                R(r, c) = R(r-1, c)*0.5 + R(r+1, c)*0.5;
            elseif (mod(r, 2)==0 & mod(c, 2)==0)
                R(r, c) = R(r-1, c-1)*0.25 + R(r-1, c+1)*0.25 + R(r+1, c-1)*0.25 + R(r+1, c+1)*0.25;
            end
        end
    end

    % fill in B
    for r = 2:(nrow-1)
        for c = 2:(ncol-1)
            if (mod(r, 2)==0 & mod(c, 2)==1)
                B(r, c) = B(r, c-1)*0.5 + B(r, c+1)*0.5;
            elseif (mod(r, 2)==1 & mod(c, 2)==0)
                B(r, c) = B(r-1, c)*0.5 + B(r+1, c)*0.5;
            elseif (mod(r, 2)==1 & mod(c, 2)==1)
                B(r, c) = B(r-1, c-1)*0.25 + B(r-1, c+1)*0.25 + B(r+1, c-1)*0.25 + B(r+1, c+1)*0.25;
            end
        end
    end

    % fill in G
    for r = 2:(nrow-1)
        for c = 2:(ncol-1)
            if (mod(r, 2)==0 & mod(c, 2)==0)
                G(r, c) = G(r-1, c)*0.25 + G(r+1, c)*0.25 + G(r, c-1)*0.25 + G(r, c+1)*0.25;
            elseif (mod(r, 2)==1 & mod(c, 2)==1)
                G(r, c) = G(r-1, c)*0.25 + G(r+1, c)*0.25 + G(r, c-1)*0.25 + G(r, c+1)*0.25;
            end
        end
    end

    color_out = cat(3, R, G, B);
    color_out(1, :, :) = color_out(2, :, :);
    color_out(nrow, :, :) = color_out(nrow-1, :, :);
    color_out(:, 1, :) = color_out(:, 2, :);
    color_out(:, ncol, :) = color_out(:, ncol-1, :);

end
