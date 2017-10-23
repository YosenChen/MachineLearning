file_name = dir('*.gif');
[dim1, dim2] = size(file_name);
data = zeros(dim1, 28*28);
for i= 1:dim1
    vec = imread(file_name(1).name);
    [dim1_vec, dim2_vec] = size(vec);
    if dim1_vec == 370
        data(i,:) = reshape(imresize(vec,0.075),[1,28*28]);
    end   
end

csvwrite('xingshu.csv',data)