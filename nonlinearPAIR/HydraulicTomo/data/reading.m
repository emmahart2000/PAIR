%% Reshaping Target Data
target_all = zeros(10000,20,19);
for i = 1:100
    name = append('targetHT', string(i),'.mat');
    target_entry = matfile(name);
    for j = 1:100
        target_all((i-1)*100+j,:,:) = target_entry.target(:,:,j);
    end
end

target_train = target_all(1:9000,:,:);
target_test = target_all(9001:10000,:,:);

target_min = min(min(min(target_all)));
target_all = target_all - target_min;
target_max = max(max(max(target_all)));
target_all_norm = target_all / target_max;
target_train_norm = target_all_norm(1:9000,:,:);
target_test_norm = target_all_norm(9001:10000,:,:);

save('targetHT.mat','target_train_norm', 'target_test_norm')

%% Reshaping Input Data
input_all = zeros(10000,100,100);
for i = 1:100
    name = append('inputHT', string(i),'.mat');
    input_entry = matfile(name);
    for j = 1:100
        input_all((i-1)*100+j,:,:) = input_entry.input(:,:,j);
    end
end
input_train = input_all(1:9000,:,:);
input_test = input_all(9001:10000,:,:);

input_min = min(min(min(input_all)));
input_all = input_all - input_min;
input_max = max(max(max(input_all)));
input_all_norm = input_all / input_max;
input_train_norm = input_all_norm(1:9000,:,:);
input_test_norm = input_all_norm(9001:10000,:,:);

save('inputHT.mat','input_train_norm', 'input_test_norm')

