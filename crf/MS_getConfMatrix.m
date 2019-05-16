function conf = MS_getConfMatrix(targets, preds)
% targets: 1xN cell array with each cell item being a matrix of size 1xM
% preds: 1xN cell array with each cell item being a matrix of size 1xM
N = size(targets, 2);
assert(N == size(preds, 2));
conf = zeros(5);
for i=1:N
    t = targets{i};
    p = preds{i};
    assert(size(t, 1) == size(p, 1))
    assert(size(t, 2) == size(p, 2))
    for j=1:size(t, 2)
        conf(t(j)+1, p(j)+1) = conf(t(j)+1, p(j)+1) + 1;
    end
end
disp(conf);
end