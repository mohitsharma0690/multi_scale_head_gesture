function [detect falsePos threshValue rawResults] = CreateROC(labels, ll, rangeThresh)

if iscell(ll) && numel(ll)>0
    if iscell(labels) && numel(labels)
        if numel(labels{1}) ~= size(ll{1},2)
            ll = transposeCellArray(ll)';
        end
    end
    ll = cell2mat(ll);
end
if iscell(labels) && numel(labels)
    labels = cell2mat(labels);
end

detect = zeros(1,numel(rangeThresh));
falsePos = zeros(1,numel(rangeThresh));
threshValue = zeros(1,numel(rangeThresh));
if numel(rangeThresh) == 1
    minValue = min(ll);
    maxValue = max(ll);
    inc = (maxValue - minValue)/abs(rangeThresh);
    rangeThresh = minValue:inc:maxValue;
end

rawResults = zeros(numel(rangeThresh),4);
for i=1:size(rangeThresh,2)
    thresh = rangeThresh(i);
    d = (ll > thresh);
    n = sum(d == 1 & labels == 1);
    f = sum(d == 1 & labels ~= 1);
    t = sum(labels == 1);
    totalfalsepos = sum(labels ~= 1);
    if t == 0
        detect(i) = 0;
    else
        detect(i) = n/t;
    end
    falsePos(i) = f/totalfalsepos;
    threshValue(i) = thresh;
    rawResults(i,:) = [n t f totalfalsepos];
end
