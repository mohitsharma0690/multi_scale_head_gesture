function display_gesture_sequences(gest_h5, target_file)
    h5_data = h5read(gest_h5, target_file);
    % Since the data as read by matlab is in column major order.
    h5_data = h5_data';
    USE_CLIM = 1;
    if USE_CLIM 
        clim_th = max(h5_data(:));
        h5_data(h5_data == 0) = clim_th;
        clim_th = [min(h5_data(:)), clim_th];
        imagesc(h5_data, clim_th);        
    else
        imagesc(h5_data);
    end
    % imshow(im);
    colorbar
end