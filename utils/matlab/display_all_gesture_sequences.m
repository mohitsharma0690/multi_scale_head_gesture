function display_all_gesture_sequences(gest_h5)
    h5_info = h5info(gest_h5, '/');
    num_datasets = numel(h5_info.Datasets);
    
    for i=1:num_datasets
        dataset_name = sprintf('/%s', h5_info.Datasets(i).Name);
        h5_data = h5read(gest_h5, dataset_name);
        % Since the data as read by matlab is in column major order.
        h5_data = h5_data';
        USE_CLIM = 1;
        if USE_CLIM
            clim_th = max(h5_data(:));
            h5_data(h5_data == 0) = clim_th;
            clim_th = [min(h5_data(:)), clim_th];
            figure
            im = imagesc(h5_data, clim_th);
            colorbar
            
%             set(gca,'XTick',[]) % Remove the ticks in the x axis!
%             set(gca,'YTick',[]) % Remove the ticks in the y axis
%             set(gca,'Position',[0 0 1 1]) % Make the axes occupy the hole figure
            
            saveas(gcf, sprintf('%s.png', h5_info.Datasets(i).Name),'png');
            pause(0.1)
            
        else
            imagesc(h5_data);
        end
        % imshow(im);
        colorbar
    end
end