%/*******************************************************
% * Copyright (C) 2019-2020 Ruixu Liu <liur05@udayton.edu>
% * 
% * This file is part of Artificial Neural Network.
% * 
% * MIT License
% *******************************************************/
%% loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
function labels = loadMNISTLabels(filename)
%the labels for the MNIST images

fileID = fopen(filename,'r','ieee-be');

number_of_total_labels=fread(fileID,2,'uint32');

label_value=fread(fileID);

fclose(fileID);

labels=reshape(label_value,number_of_total_labels(2),1);

end
