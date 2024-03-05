%% Load Existing Hydrolic Tomography Data
for i=1:10
        load(append('data/inputHT',string(i),'.mat'))
        load(append('data/targetHT',string(i),'.mat'))
    for j=1:100
        imwrite(input(:,:,j),append('images/train/input',string((i-1)*100+j),'.png'),'PNG')
        imwrite(target(:,:,j),append('images/train/target',string((i-1)*100+j),'.png'),'PNG')
    end
end