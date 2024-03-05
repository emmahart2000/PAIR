close all

n_xwells = 4; n_ywells = 5;
x_loc = linspace(-1,1,n_xwells+2); x_loc = x_loc(2:end-1);
y_loc = linspace(-1,1,n_ywells+2); y_loc = y_loc(2:end-1);
count = 1;
for i = 1:n_xwells
    for  j = 1:n_ywells
        xwell(count) = x_loc(i); ywell(count) = y_loc(j);
        count = count + 1;
    end
end

%%
for k = 1:1
    idx = randi(100);
    inputName = ['data/inputHT',num2str(k),'.mat'];
    targetName = ['data/targetHT',num2str(k),'.mat'];
    load(inputName,'input')
    load(targetName,'target')
    figure(1)
    imagesc([-1, 1],[-1, 1], log10(flipud(input(:,:,idx)))), hold on
    colormap cool, title(['input: file ',num2str(k),'# ',num2str(idx)])
    colorbar
    figure(2)
    for j = 1:length(xwell)
        subplot(n_xwells,n_ywells,j)
        scatter(xwell([1:j-1,j+1:end]),ywell([1:j-1,j+1:end]),[], target(j,:,k),'filled')
    end
    sgtitle(['target: file ',num2str(k),'# ',num2str(idx)])
end
    