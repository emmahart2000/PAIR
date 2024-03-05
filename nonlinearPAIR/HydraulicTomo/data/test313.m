%% Load input data and show information
input1 = matfile('inputHT2.mat');
whos(input1)

%% Show one image
clf
figure(1)
input_img = input1.input(:,:,1);
max_scale = max(max(input_img));
max(max(input_img/max_scale))
imshow(input_img/max_scale)

for i = 1:10
    figure(i)
    input_img = input1.input(:,:,i);
    max_scale = max(max(input_img));
    max(max(input_img/max_scale))
    imshow(input_img/max_scale)
end

%% Load target data and show information
target1 = matfile('')