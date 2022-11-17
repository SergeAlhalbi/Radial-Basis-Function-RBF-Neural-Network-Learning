%==========================================================================
% Project 3
%==========================================================================

%% Loading

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

%% Task 1 (Kmeans Elbow Algorithm Function)

%% Tasks 2, 3, 4, and 5

%-------------------------------------
% Training with the first learning rate
%-------------------------------------

% Clustering
K=50;
max_iterations=100;
[mu_k,deno] = Kmeans(K,T1_train,max_iterations);

% SLP
Learning_rate = 0.1; % Learning_rate
Hidden_layer_neurons = K; % Hidden layer
output_neurons = 10; % Output layer
w_ji = -0.5+rand(Hidden_layer_neurons,output_neurons); % initial weights
iterations = 1;
Error_end = 1;
Error_threshold = 0.001;
max_iterations = 1000;
Error_update = zeros(max_iterations,1);

while Error_end > Error_threshold && iterations < max_iterations
    E = 0;
    for m = 1 : size(T1_train,2)
        % from 1 to 3000

        d = zeros(1,output_neurons);
        x_i = repmat(T1_train(:,m),[1,Hidden_layer_neurons]);
        
        % Using clusters to extract the inpute data feature
        kmeans_input = sum((x_i - mu_k).^2);
        u_k = exp (- kmeans_input./deno);
        u_k_x = repmat(u_k',[1,output_neurons]);

        % train a SLP
        Net = sum(w_ji .* u_k_x);
        % Sigmoid activation function
        y = 1./(1+ exp(-Net));
        % K * 10
        y_j = repmat(y,[K,1]);   

        d(T1_train_label(m)+1) = 1;

        sigma = d-y;
        sigma_ji = repmat(sigma,[Hidden_layer_neurons,1]);

        part_w1 = sigma_ji.*u_k_x;         % step 6
        part_w2 = y_j.*(1 - y_j);
        delta_w = Learning_rate*part_w1.*part_w2;  

        w_ji = w_ji + delta_w;

        En = sum((d-y).^2);
        E = E + En;
    end

    Error_end = E/(10*size(T1_train,2));
    Error_update(iterations) = Error_end;
    iterations = iterations+1;
end
Error_for_plot = Error_update(1:iterations-1);

figure(2);
plot(Error_for_plot);
grid;
str = sprintf('learning rate %g, %d hidden nodes'...
    ,Learning_rate,Hidden_layer_neurons);
title(str);
xlabel('iteration');
ylabel('Mean square error');

%-------------------------------------
% Testing with the first learning rate
%-------------------------------------

W = w_ji;
Threshold = 0.5;
% True Positive
TP = zeros(1,10);
% False Poitive
FP = zeros(1,10);
% False Negative
FN = zeros(1,10);
% True Negative
TN = zeros(1,10);
for m_test = 1 : size(T1_test,2)
 % from 1 to 300

    y_real = zeros(1,10);
    y_test = zeros(1,10);
    test_x_i = repmat(T1_test(:,m_test),[1,K]);

    % Hidden layer
    kmeans_test_input = sum((test_x_i - mu_k).^2);
    u_test_ki = exp (- kmeans_test_input./deno);
    u_test_k = repmat(u_test_ki',[1,10]);

    % Output layer
    Net_test = sum( W .* u_test_k);
    % activation
    y_t = 1./(1+ exp(-Net_test)); 

    y_test(y_t > Threshold) = 1;
    y_real(1,T1_test_label(m_test)+1) = 1;
    
    if y_test(T1_test_label(m_test)+1)==1
        % Calculate TP
        TP = TP + y_real;
    else
        % Calculate FN
        FN = FN + y_real;
        y_test(T1_test_label(m_test)+1)=1;
    end
    % Calculate FP
    FP = FP + y_test - y_real;
    % Calculate TN
    TN = TN + abs(y_test - 1);

end

% True Positive Rate = TP/(TP+FN)
TPR = TP./(TP+FN);
% False Positive Rate = FP/(FP+TN)
FPR = FP./(FP+TN);

figure(3);
subplot(1,2,1);
x_axis = 0:1:9;
bar(x_axis, TPR);
str_test = sprintf('%d training, %d testing, %d nodes,\n learning rate %g'...
    ,size(T1_train,2),size(T1_test,2),Hidden_layer_neurons,Learning_rate);
title(str_test);
xlabel('digits');
ylabel('TPR');

subplot(1,2,2);
bar(x_axis, FPR);
str_test = sprintf('%d training, %d testing, %d nodes,\n learning rate %g'...
    ,size(T1_train,2),size(T1_test,2),Hidden_layer_neurons,Learning_rate);
title(str_test);
xlabel('digits');
ylabel('FPR');

%-------------------------------------
% Training with the second learning rate
%-------------------------------------

% Clustering
K=50;
max_iterations=100;
[mu_k,deno] = Kmeans(K,T1_train,max_iterations);

% SLP
Learning_rate = 0.15; % Learning_rate
Hidden_layer_neurons = K; % Hidden layer
output_neurons = 10; % Output layer
w_ji = -0.5+rand(Hidden_layer_neurons,output_neurons); % initial weights
iterations = 1;
Error_end = 1;
Error_threshold = 0.001;
max_iterations = 1000;
Error_update = zeros(max_iterations,1);

while Error_end > Error_threshold && iterations < max_iterations
    E = 0;
    for m = 1 : size(T1_train,2)
        % from 1 to 3000

        d = zeros(1,output_neurons);
        x_i = repmat(T1_train(:,m),[1,Hidden_layer_neurons]);
        
        % Using clusters to extract the input data feature
        kmeans_input = sum((x_i - mu_k).^2);
        u_k = exp (- kmeans_input./deno);
        u_k_x = repmat(u_k',[1,output_neurons]);

        % train a SLP
        Net = sum(w_ji .* u_k_x);
        % Sigmoid activation function
        y = 1./(1+ exp(-Net));
        % K * 10
        y_j = repmat(y,[K,1]);   

        d(T1_train_label(m)+1) = 1;

        sigma = d-y;
        sigma_ji = repmat(sigma,[Hidden_layer_neurons,1]);

        part_w1 = sigma_ji.*u_k_x;         % step 6
        part_w2 = y_j.*(1 - y_j);
        delta_w = Learning_rate*part_w1.*part_w2;  

        w_ji = w_ji + delta_w;

        En = sum((d-y).^2);
        E = E + En;
    end

    Error_end = E/(10*size(T1_train,2));
    Error_update(iterations) = Error_end;
    iterations = iterations+1;
end
Error_for_plot = Error_update(1:iterations-1);

figure(4);
plot(Error_for_plot);
grid;
str = sprintf('learning rate %g, %d hidden nodes'...
    ,Learning_rate,Hidden_layer_neurons);
title(str);
xlabel('iteration');
ylabel('Mean square error');

%-------------------------------------
% Testing with the second learning rate
%-------------------------------------

W = w_ji;
Threshold = 0.5;
% True Positive
TP = zeros(1,10);
% False Poitive
FP = zeros(1,10);
% False Negative
FN = zeros(1,10);
% True Negative
TN = zeros(1,10);
for m_test = 1 : size(T1_test,2)
 % from 1 to 300

    y_real = zeros(1,10);
    y_test = zeros(1,10);
    test_x_i = repmat(T1_test(:,m_test),[1,K]);

    % Hidden layer
    kmeans_test_input = sum((test_x_i - mu_k).^2);
    u_test_ki = exp (- kmeans_test_input./deno);
    u_test_k = repmat(u_test_ki',[1,10]);

    % Output layer
    Net_test = sum( W .* u_test_k);
    % activation
    y_t = 1./(1+ exp(-Net_test)); 

    y_test(y_t > Threshold) = 1;
    y_real(1,T1_test_label(m_test)+1) = 1;
    
    if y_test(T1_test_label(m_test)+1)==1
        % Calculate TP
        TP = TP + y_real;
    else
        % Calculate FN
        FN = FN + y_real;
        y_test(T1_test_label(m_test)+1)=1;
    end
    % Calculate FP
    FP = FP + y_test - y_real;
    % Calculate TN
    TN = TN + abs(y_test - 1);

end

% True Positive Rate = TP/(TP+FN)
TPR = TP./(TP+FN);
% False Positive Rate = FP/(FP+TN)
FPR = FP./(FP+TN);

figure(5);
subplot(1,2,1);
x_axis = 0:1:9;
bar(x_axis, TPR);
str_test = sprintf('%d training, %d testing, %d nodes,\n learning rate %g'...
    ,size(T1_train,2),size(T1_test,2),Hidden_layer_neurons,Learning_rate);
title(str_test);
xlabel('digits');
ylabel('TPR');

subplot(1,2,2);
bar(x_axis, FPR);
str_test = sprintf('%d training, %d testing, %d nodes,\n learning rate %g'...
    ,size(T1_train,2),size(T1_test,2),Hidden_layer_neurons,Learning_rate);
title(str_test);
xlabel('digits');
ylabel('FPR');

%-------------------------------------
% Training with the third learning rate
%-------------------------------------

% Clustering
K=50;
max_iterations=100;
[mu_k,deno] = Kmeans(K,T1_train,max_iterations);

% SLP
Learning_rate = 0.05; % Learning_rate
Hidden_layer_neurons = K; % Hidden layer
output_neurons = 10; % Output layer
w_ji = -0.5+rand(Hidden_layer_neurons,output_neurons); % initial weights
iterations = 1;
Error_end = 1;
Error_threshold = 0.001;
max_iterations = 1000;
Error_update = zeros(max_iterations,1);

while Error_end > Error_threshold && iterations < max_iterations
    E = 0;
    for m = 1 : size(T1_train,2)
        % from 1 to 3000

        d = zeros(1,output_neurons);
        x_i = repmat(T1_train(:,m),[1,Hidden_layer_neurons]);
        
        % Using clusters to extract the inpute data feature
        kmeans_input = sum((x_i - mu_k).^2);
        u_k = exp (- kmeans_input./deno);
        u_k_x = repmat(u_k',[1,output_neurons]);

        % train a SLP
        Net = sum(w_ji .* u_k_x);
        % Sigmoid activation function
        y = 1./(1+ exp(-Net));
        % K * 10
        y_j = repmat(y,[K,1]);   

        d(T1_train_label(m)+1) = 1;

        sigma = d-y;
        sigma_ji = repmat(sigma,[Hidden_layer_neurons,1]);

        part_w1 = sigma_ji.*u_k_x;         % step 6
        part_w2 = y_j.*(1 - y_j);
        delta_w = Learning_rate*part_w1.*part_w2;  

        w_ji = w_ji + delta_w;

        En = sum((d-y).^2);
        E = E + En;
    end

    Error_end = E/(10*size(T1_train,2));
    Error_update(iterations) = Error_end;
    iterations = iterations+1;
end
Error_for_plot = Error_update(1:iterations-1);

figure(6);
plot(Error_for_plot);
grid;
str = sprintf('learning rate %g, %d hidden nodes'...
    ,Learning_rate,Hidden_layer_neurons);
title(str);
xlabel('iteration');
ylabel('Mean square error');

%-------------------------------------
% Testing with the third learning rate
%-------------------------------------

W = w_ji;
Threshold = 0.5;
% True Positive
TP = zeros(1,10);
% False Poitive
FP = zeros(1,10);
% False Negative
FN = zeros(1,10);
% True Negative
TN = zeros(1,10);
for m_test = 1 : size(T1_test,2)
 % from 1 to 300

    y_real = zeros(1,10);
    y_test = zeros(1,10);
    test_x_i = repmat(T1_test(:,m_test),[1,K]);

    % Hidden layer
    kmeans_test_input = sum((test_x_i - mu_k).^2);
    u_test_ki = exp (- kmeans_test_input./deno);
    u_test_k = repmat(u_test_ki',[1,10]);

    % Output layer
    Net_test = sum( W .* u_test_k);
    % activation
    y_t = 1./(1+ exp(-Net_test)); 

    y_test(y_t > Threshold) = 1;
    y_real(1,T1_test_label(m_test)+1) = 1;
    
    if y_test(T1_test_label(m_test)+1)==1
        % Calculate TP
        TP = TP + y_real;
    else
        % Calculate FN
        FN = FN + y_real;
        y_test(T1_test_label(m_test)+1)=1;
    end
    % Calculate FP
    FP = FP + y_test - y_real;
    % Calculate TN
    TN = TN + abs(y_test - 1);

end

% True Positive Rate = TP/(TP+FN)
TPR = TP./(TP+FN);
% False Positive Rate = FP/(FP+TN)
FPR = FP./(FP+TN);

figure(7);
subplot(1,2,1);
x_axis = 0:1:9;
bar(x_axis, TPR);
str_test = sprintf('%d training, %d testing, %d nodes,\n learning rate %g'...
    ,size(T1_train,2),size(T1_test,2),Hidden_layer_neurons,Learning_rate);
title(str_test);
xlabel('digits');
ylabel('TPR');

subplot(1,2,2);
bar(x_axis, FPR);
str_test = sprintf('%d training, %d testing, %d nodes,\n learning rate %g'...
    ,size(T1_train,2),size(T1_test,2),Hidden_layer_neurons,Learning_rate);
title(str_test);
xlabel('digits');
ylabel('FPR');

%-------------------------------------
% Training with the second k value lower than optimum K
%-------------------------------------


% Clustering
K=25;
max_iterations=100;
[mu_k,deno] = Kmeans(K,T1_train,max_iterations);

% SLP
Learning_rate = 0.1; % Learning_rate
Hidden_layer_neurons = K; % Hidden layer
output_neurons = 10; % Output layer
w_ji = -0.5+rand(Hidden_layer_neurons,output_neurons); % initial weights
iterations = 1;
Error_end = 1;
Error_threshold = 0.001;
max_iterations = 1000;
Error_update = zeros(max_iterations,1);

while Error_end > Error_threshold && iterations < max_iterations
    E = 0;
    for m = 1 : size(T1_train,2)
        % from 1 to 3000

        d = zeros(1,output_neurons);
        x_i = repmat(T1_train(:,m),[1,Hidden_layer_neurons]);
        
        % Using clusters to extract the inpute data feature
        kmeans_input = sum((x_i - mu_k).^2);
        u_k = exp (- kmeans_input./deno);
        u_k_x = repmat(u_k',[1,output_neurons]);

        % train a SLP
        Net = sum(w_ji .* u_k_x);
        % Sigmoid activation function
        y = 1./(1+ exp(-Net));
        % K * 10
        y_j = repmat(y,[K,1]);   

        d(T1_train_label(m)+1) = 1;

        sigma = d-y;
        sigma_ji = repmat(sigma,[Hidden_layer_neurons,1]);

        part_w1 = sigma_ji.*u_k_x;         % step 6
        part_w2 = y_j.*(1 - y_j);
        delta_w = Learning_rate*part_w1.*part_w2;  

        w_ji = w_ji + delta_w;

        En = sum((d-y).^2);
        E = E + En;
    end

    Error_end = E/(10*size(T1_train,2));
    Error_update(iterations) = Error_end;
    iterations = iterations+1;
end
Error_for_plot = Error_update(1:iterations-1);

figure(8);
plot(Error_for_plot);
grid;
str = sprintf('learning rate %g, %d hidden nodes'...
    ,Learning_rate,Hidden_layer_neurons);
title(str);
xlabel('iteration');
ylabel('Mean square error');

%-------------------------------------
% Testing with the second k value lower than optimum K
%-------------------------------------

W = w_ji;
Threshold = 0.5;
% True Positive
TP = zeros(1,10);
% False Poitive
FP = zeros(1,10);
% False Negative
FN = zeros(1,10);
% True Negative
TN = zeros(1,10);
for m_test = 1 : size(T1_test,2)
 % from 1 to 300

    y_real = zeros(1,10);
    y_test = zeros(1,10);
    test_x_i = repmat(T1_test(:,m_test),[1,K]);

    % Hidden layer
    kmeans_test_input = sum((test_x_i - mu_k).^2);
    u_test_ki = exp (- kmeans_test_input./deno);
    u_test_k = repmat(u_test_ki',[1,10]);

    % Output layer
    Net_test = sum( W .* u_test_k);
    % activation
    y_t = 1./(1+ exp(-Net_test)); 

    y_test(y_t > Threshold) = 1;
    y_real(1,T1_test_label(m_test)+1) = 1;
    
    if y_test(T1_test_label(m_test)+1)==1
        % Calculate TP
        TP = TP + y_real;
    else
        % Calculate FN
        FN = FN + y_real;
        y_test(T1_test_label(m_test)+1)=1;
    end
    % Calculate FP
    FP = FP + y_test - y_real;
    % Calculate TN
    TN = TN + abs(y_test - 1);

end

% True Positive Rate = TP/(TP+FN)
TPR = TP./(TP+FN);
% False Positive Rate = FP/(FP+TN)
FPR = FP./(FP+TN);

figure(9);
subplot(1,2,1);
x_axis = 0:1:9;
bar(x_axis, TPR);
str_test = sprintf('%d training, %d testing, %d nodes,\n learning rate %g'...
    ,size(T1_train,2),size(T1_test,2),Hidden_layer_neurons,Learning_rate);
title(str_test);
xlabel('digits');
ylabel('TPR');

subplot(1,2,2);
bar(x_axis, FPR);
str_test = sprintf('%d training, %d testing, %d nodes,\n learning rate %g'...
    ,size(T1_train,2),size(T1_test,2),Hidden_layer_neurons,Learning_rate);
title(str_test);
xlabel('digits');
ylabel('FPR');

%-------------------------------------
% Training with the third k value higher than optimum K
%-------------------------------------

% Clustering
K=65;
max_iterations=100;
[mu_k,deno] = Kmeans(K,T1_train,max_iterations);

% SLP
Learning_rate = 0.1; % Learning_rate
Hidden_layer_neurons = K; % Hidden layer
output_neurons = 10; % Output layer
w_ji = -0.5+rand(Hidden_layer_neurons,output_neurons); % initial weights
iterations = 1;
Error_end = 1;
Error_threshold = 0.001;
max_iterations = 1000;
Error_update = zeros(max_iterations,1);

while Error_end > Error_threshold && iterations < max_iterations
    E = 0;
    for m = 1 : size(T1_train,2)
        % from 1 to 3000

        d = zeros(1,output_neurons);
        x_i = repmat(T1_train(:,m),[1,Hidden_layer_neurons]);
        
        % Using clusters to extract the inpute data feature
        kmeans_input = sum((x_i - mu_k).^2);
        u_k = exp (- kmeans_input./deno);
        u_k_x = repmat(u_k',[1,output_neurons]);

        % train a SLP
        Net = sum(w_ji .* u_k_x);
        % Sigmoid activation function
        y = 1./(1+ exp(-Net));
        % K * 10
        y_j = repmat(y,[K,1]);   

        d(T1_train_label(m)+1) = 1;

        sigma = d-y;
        sigma_ji = repmat(sigma,[Hidden_layer_neurons,1]);

        part_w1 = sigma_ji.*u_k_x;         % step 6
        part_w2 = y_j.*(1 - y_j);
        delta_w = Learning_rate*part_w1.*part_w2;  

        w_ji = w_ji + delta_w;

        En = sum((d-y).^2);
        E = E + En;
    end

    Error_end = E/(10*size(T1_train,2));
    Error_update(iterations) = Error_end;
    iterations = iterations+1;
end
Error_for_plot = Error_update(1:iterations-1);

figure(10);
plot(Error_for_plot);
grid;
str = sprintf('learning rate %g, %d hidden nodes'...
    ,Learning_rate,Hidden_layer_neurons);
title(str);
xlabel('iteration');
ylabel('Mean square error');

%-------------------------------------
% Testing with the third k value higher than optimum K
%-------------------------------------

W = w_ji;
Threshold = 0.5;
% True Positive
TP = zeros(1,10);
% False Poitive
FP = zeros(1,10);
% False Negative
FN = zeros(1,10);
% True Negative
TN = zeros(1,10);
for m_test = 1 : size(T1_test,2)
 % from 1 to 300

    y_real = zeros(1,10);
    y_test = zeros(1,10);
    test_x_i = repmat(T1_test(:,m_test),[1,K]);

    % Hidden layer
    kmeans_test_input = sum((test_x_i - mu_k).^2);
    u_test_ki = exp (- kmeans_test_input./deno);
    u_test_k = repmat(u_test_ki',[1,10]);

    % Output layer
    Net_test = sum( W .* u_test_k);
    % activation
    y_t = 1./(1+ exp(-Net_test)); 

    y_test(y_t > Threshold) = 1;
    y_real(1,T1_test_label(m_test)+1) = 1;
    
    if y_test(T1_test_label(m_test)+1)==1
        % Calculate TP
        TP = TP + y_real;
    else
        % Calculate FN
        FN = FN + y_real;
        y_test(T1_test_label(m_test)+1)=1;
    end
    % Calculate FP
    FP = FP + y_test - y_real;
    % Calculate TN
    TN = TN + abs(y_test - 1);

end

% True Positive Rate = TP/(TP+FN)
TPR = TP./(TP+FN);
% False Positive Rate = FP/(FP+TN)
FPR = FP./(FP+TN);

figure(11);
subplot(1,2,1);
x_axis = 0:1:9;
bar(x_axis, TPR);
str_test = sprintf('%d training, %d testing, %d nodes,\n learning rate %g'...
    ,size(T1_train,2),size(T1_test,2),Hidden_layer_neurons,Learning_rate);
title(str_test);
xlabel('digits');
ylabel('TPR');

subplot(1,2,2);
bar(x_axis, FPR);
str_test = sprintf('%d training, %d testing, %d nodes,\n learning rate %g'...
    ,size(T1_train,2),size(T1_test,2),Hidden_layer_neurons,Learning_rate);
title(str_test);
xlabel('digits');
ylabel('FPR');
%==========================================================================