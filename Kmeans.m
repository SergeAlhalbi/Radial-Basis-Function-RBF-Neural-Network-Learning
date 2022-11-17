%% Kmeans Algorithm

function[mu_k,deno] = Kmeans(K,T1_train,max_iterations)

rng(8); % For repeated results
[mu_k, cluster_index] = datasample(T1_train,K,2);
sigma_k = zeros(1,K);
mu_k_previous = mu_k;
center_changed = 1;
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

    % Computing the new mean and standard deviation of each cluster 
    for i = 1 : K
        non_zeros = find(cluster_index_adj(:,i));
        C = T1_train(:,cluster_index_adj(non_zeros,i));
        mu_k(:,i) = mean(C,2); % Mean
        sig_k = sqrt(sum((C - repmat(mu_k(:,i),[1,size(C,2)])).^2)); % Standard deviation element
        sigma_k(i) = mean(sig_k); % Standard deviation for the i'th hidden neuron
        % Avoid having one pattern only in one cluster to avoid a standard deviation = 0 and a divison by 0 later on
    end
    % Stop if the centers do not change
    center_changed = sum((mu_k_previous - mu_k).^2,'all');
    mu_k_previous = mu_k;
    iteration = iteration + 1;
    cluster_index=cluster_index_adj(1,:);
end

% Kernel function denominator
deno = 2* sigma_k.^2;