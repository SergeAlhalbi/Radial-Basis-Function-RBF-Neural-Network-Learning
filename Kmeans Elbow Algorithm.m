%% K-Means Elbow Algorithm

% Load the train and test images
training_images = loadMNISTImages('train-images.idx3-ubyte');
% train_images(:,i) is a double matrix of size 784xi(where i = 1 to 60000)
% Intensity rescale to [0,1]

training_labels = loadMNISTLabels('train-labels.idx1-ubyte');
% train_labels(i) - 60000x1 vector

testing_images = loadMNISTImages('t10k-images.idx3-ubyte');
% testing_images(:,i) is a double matrix of size 784xi(where i = 1 to 10000)
% Intensity rescale to [0,1]

testing_labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
% test_labels(i) - 10000x1 vector

% Prepare experinment data
number_of_training_images = 1000;
number_of_testing_images = 300;
[balanced_train_image, balanced_train_labels] = balance_MNIST_selection(...
    training_images,training_labels,number_of_training_images);
[balanced_test_image, balanced_test_labels] = balance_MNIST_selection(...
    testing_images,testing_labels,number_of_training_images);

T1_train = zeros(784, number_of_training_images);
T1_train_label = zeros(number_of_training_images,1);
T1_test = zeros(784, number_of_testing_images);
T1_test_label = zeros(number_of_testing_images,1);

for i = 1: number_of_training_images
    T1_train(:,i) = balanced_train_image(:,i);
    T1_train_label(i) = balanced_train_labels(i);
end

for i = 1: number_of_testing_images
    T1_test(:,i) = balanced_test_image(:,i);
    T1_test_label(i) = balanced_test_labels(i);
end

%% Task 1

%-------------------------------------
% Clustering
%-------------------------------------

All_K = 10:10:200; % Number of K clusters for the elbow algorithm
Mean_of_mean_m = zeros(1,length(All_K));

for counter = 1:length(All_K)
    rng(8); % For repeated results
    K = All_K(counter);
    [mu_k, cluster_index] = datasample(T1_train,K,2);
    sigma_k = zeros(1,K);
    mu_k_previous = mu_k;
    center_changed = 1;
    max_iterations = 100;
    iteration = 0;
    while center_changed > 0.001 && iteration < max_iterations

        number_of_digits_in_each_cluster = ones(1,K); %(+50 centers => total = 1050)

        for m = 1 : size(T1_train,2)
            % Ex: from 1 to 1000
            x_i = repmat(T1_train(:,m),[1,K]);
            % Compute the distance for each input
            distance = sum((x_i - mu_k).^2);
            % Find the minimum distance index to the temporal center
            [~ ,close_center_idx] = min(distance);

            % Join the current input data index to the cluster index
            number_of_digits_in_each_cluster(close_center_idx) = ...
            number_of_digits_in_each_cluster(close_center_idx)+1;
            cluster_index(number_of_digits_in_each_cluster(close_center_idx),close_center_idx) = m;
        end 

        % Adjsut the cluster index to remove duplicate patterns
        cluster_index_adj = zeros(size(cluster_index));
        for i = 1 : K
            if cluster_index(2,i)~=0
                cluster_index_adj(:,i) = [cluster_index(2:end,i); 0];
            else
                cluster_index_adj(:,i) = cluster_index(:,i);
            end
        end

        % Computing the new mean of each cluster 
        for i = 1 : K
            non_zeros = find(cluster_index_adj(:,i));
            C = T1_train(:,cluster_index_adj(non_zeros,i));
            mu_k(:,i) = mean(C,2); % Mean
        end
        % Stop if the centers do not change
        center_changed = sum((mu_k_previous - mu_k).^2,'all');
        mu_k_previous = mu_k;
        iteration = iteration + 1;
        cluster_index=cluster_index_adj(1,:);
    end

    % Evaluating clustering score
    Mdis_k = zeros(1,K);
        for i = 1:K
            non_zeros = find(cluster_index_adj(:,i));
            C = T1_train(:,cluster_index_adj(non_zeros,i));
            Mdis_k(i) = mean(sqrt(mean((C-repmat(mu_k(:,i),1,size(C,2))).^2)),2); % Average intra distance to cluster i
        end
        Mean_of_mean_m(counter) = mean(Mdis_k)./(K); % Average inter distance between clusters
end

figure(1)
plot(All_K,Mean_of_mean_m)
grid;
str = sprintf('Elbow Algorithm, Increment = 10');
title(str);
xlabel('Cluster Numbers K');
ylabel('Mean of Mean Inter Class Distance');