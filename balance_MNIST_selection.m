%/*******************************************************
% * Copyright (C) 2019-2020 Ruixu Liu <liur05@udayton.edu>
% * 
% * This file is part of Artificial Neural Network.
% * 
% * MIT License
% *******************************************************/
%% Balance dataset function
function [balanced_data, balanced_labels] = balance_MNIST_selection(data, labels, total_numbers)
    %Input:  data            784 x 60000
    %        label           60000  x 1
    %        total_number    number of training data
    %Output: balanced_data   784 x training data
    %        balanced_labels training data x 1
    
    number_of_each_digit = total_numbers/10;

    balanced_data = zeros(784,total_numbers);
    balanced_labels = zeros(total_numbers,1);
    count_number_of_digits = zeros(10,1);
    index = 1;
    for i = 1: length(labels)
        if count_number_of_digits(labels(i)+1)<number_of_each_digit
            balanced_data(:,index) = data(:,i);
            balanced_labels(index) = labels(i);
            index = index + 1;
            count_number_of_digits(labels(i)+1) = count_number_of_digits(labels(i)+1) + 1;
        end
        if isempty(find(count_number_of_digits<number_of_each_digit, 1))
            break
        end
    end
end
