function read_gest_seq(gest_seq_h5)
    h5_info = h5info(gest_seq_h5, '/');
    num_datasets = numel(h5_info.Datasets);
end