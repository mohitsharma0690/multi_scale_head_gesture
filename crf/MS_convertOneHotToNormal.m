function N = MS_convertOneHotToNormal(one_hot_N)
%
% one_hot_N is a cell of size 1xN. Each element of the cell is a matrix
% of the probability values for each sequence.
% Return: N a cell array where each item of the cell array is a matrix of
% size 1xN.
%
num_seq = size(one_hot_N, 2);
N = {};
for i=1:num_seq
    preds = one_hot_N{i};
    [max_prob, max_idx] = max(preds, [], 1);
    N{i} = max_idx - 1;
end
end