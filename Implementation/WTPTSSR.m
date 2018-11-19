%% In the name of ALLAH
% Mohammad malmir & Milad khademi noori

clc 
clear all
close all

%% Defining parameters
num_tests = 10;           % number of test
la_p = 0.01;               % local adaptor parameter                          
reg_p = 1.3;               % regularization parameter
sigma = 0.01;              % small positive Constant
num_training_samples = 5;  % number of training sample
M = 60;                    % M nearest neighbor

%% Main code
Images=importdata('ORL_images.txt');
Labels=importdata('ORL_labels.txt');
% Images=importdata('Yale_images.txt');
% Labels=importdata('Yale_labels.txt');
num_classes = max(Labels);
num_features = size(Images, 2);
accs = zeros(1, num_tests);
for j = 1:num_tests

    [tr_idx, te_idx] = rnd_selection(Labels, num_training_samples);
    
    test_samples = Images(te_idx, :)'; test_labels = Labels(te_idx);
    train_samples = Images(tr_idx, :)'; train_labels = Labels(tr_idx);
    
    num_train = size(train_samples, 2);
    num_test = size(test_samples, 2);
    predicted_labels = zeros(num_test, 1);

    W = zeros(num_train);


    for i = 1:num_test

        y = test_samples(:, i);
        for k = 1:num_train
            W(k,k) = sum(abs(y - train_samples(:,k))) ^ la_p;
            
        end

        X = (train_samples' * train_samples + reg_p * (W' * W)) \ (train_samples' * y);

        cons = zeros(num_train, 1);
        for k = 1:num_train
            cons(k) = sqrt( sum((y - X(k) * train_samples(:,k)) .^2));            
        end
        
        cons = sortrows([cons, (1:num_train)'], 1);
        Ahat = train_samples(:, cons(1:M,2));
        Xhat = (Ahat' * Ahat + sigma * eye(M)) \ (Ahat' * y);
        hat_class = train_labels(cons(1:M,2));
        cons = zeros(num_classes, 1);
        cons_data = zeros(num_features, num_classes);
        for k = 1:M
            cons_data(:,hat_class(k)) = cons_data(:,hat_class(k)) + Xhat(k) * Ahat(:,k);
        end

        for k = 1:num_classes
            cons(k) = sqrt( sum((y - cons_data(:,k)) .^2));
        end

        [~, predicted_labels(i)] = min(cons);
    end

    accuracy=size((find(test_labels==predicted_labels)),1)/size(test_labels,1);
    disp(accuracy);
    
    accs(j) = accuracy;
end

disp(['accuracy: %' num2str(100 * mean(accs)) ' +- %' num2str(100 * std(accs))]);

