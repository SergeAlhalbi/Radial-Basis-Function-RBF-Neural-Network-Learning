%/*******************************************************
% * Copyright (C) 2019-2020 Ruixu Liu <liur05@udayton.edu>
% * 
% * This file is part of Artificial Neural Network.
% * 
% * MIT License
% *******************************************************/
%% loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
function images = loadMNISTImages(filename)
%the raw MNIST images

fileID = fopen(filename, 'r','ieee-be');
Image_matrix = fread(fileID,4,'uint32');
number_of_total_images = Image_matrix(2);
xdim = Image_matrix(3);
ydim = Image_matrix(4);
% fprintf('Number of images: %d\n',numberofimages);

images = fread(fileID,xdim*ydim*number_of_total_images, 'uint8=>uint8');
images = reshape(images, [xdim, ydim, number_of_total_images]);
images = permute(images,[2 1 3]);

fclose(fileID);

% Reshape to #pixels x #examples
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% Convert to double and rescale to [0,1]
images = double(images) / 255;

end
