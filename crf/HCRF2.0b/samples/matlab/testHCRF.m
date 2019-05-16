% [ll newlabels] = testHCRF(model, seqs, labels)
%     Test sequences using a Hidden-CRF model: Compute the log likelihood
%     for each sequence. Also, based on the labels, compute error
%     measurements. 
% 
% INPUT:
%    - model            : Model parameters as returned by the Train
%                         function.
%    - seqs             : Cell array of matrices which contains the encoded
%                         features for each sequence
%    - labels           : Cell array of vectors which contains the ground
%                         truth label for each sample.
%    OUTPUT:
%    - ll               : Log likelyhood of a label for each sample. For
%                         some models, equal to the maginal probability.
%    - newLabels        : Ground truth label for each sample. In most
%                         cases, same as labels. For HMM and HCRF, the
%                         number of samples may be slightly different.
function [ll newLabels] = testHCRF(modelHCRF, seqs, labels)

%matHCRF('createToolbox','hcrf',modelHCRF.optimizer, modelHCRF.nbHiddenStates, modelHCRF.windowSize);  {-KGB}
assert (strcmp(modelHCRF.modelType, 'hcrf') || strcmp(modelHCRF.modelType, 'ghcrf')); %{+KGB}
matHCRF('createToolbox',modelHCRF.modelType,modelHCRF.optimizer, modelHCRF.nbHiddenStates, modelHCRF.windowSize); %{+KGB}
if isfield(modelHCRF,'debugLevel')
    matHCRF('set','debugLevel',modelHCRF.debugLevel);
end
matHCRF('setModel',modelHCRF.model, modelHCRF.features);

ll = cell(1,size(seqs, 2));
newLabels = cell(1,size(seqs, 2));

% Normalise data {%+KGB}
if isfield(modelHCRF,'normalise') && modelHCRF.normalise == 1
    seqs = normaliseCellArray(seqs);
    seqs = removeOutliers(seqs, modelHCRF.outlierTresh(2), modelHCRF.outlierTresh(1));     
end

for i = 1:size(seqs, 2)
    if size(seqs{i},2) <= modelHCRF.windowRecSize
        subSeq = seqs(i);
        newLabels{i}(1) = mode(labels{i});
        matHCRF('setData',subSeq,[],int32(newLabels{i}(1)));        
        matHCRF('test');
        llSeq =matHCRF('getResults');
        ll{i} = llSeq{1};
    else
        for w = 1:size(seqs{i}, 2)-modelHCRF.windowRecSize
            subSeq = {seqs{i}(:,w:w+modelHCRF.windowRecSize-1)};
            newLabels{i}(w) = labels{i}(w+modelHCRF.windowRecSize/2);
            matHCRF('setData',subSeq,[],int32(newLabels{i}(w)));
            matHCRF('test');
            llSeq =matHCRF('getResults');
            ll{i}(:,w) = llSeq{1};
        end
    end
end
      
