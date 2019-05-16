% [ll newlabels] = testCRF(model, seqs, labels)
%     Test sequences using a CRF model: Compute the log likelihood for each
%     sequence. Also, based on the labels, compute error measurements.
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
function [ll newLabels] = testCRF(modelCRF, seqs, labels)

intLabels = cellInt32(labels);
matHCRF('createToolbox','crf',modelCRF.optimizer, 0, modelCRF.windowSize);
matHCRF('setData',seqs,intLabels);
matHCRF('setModel',modelCRF.model, modelCRF.features);
if isfield(modelCRF,'debugLevel')
    matHCRF('set','debugLevel',modelCRF.debugLevel);
end
matHCRF('test');

ll=matHCRF('getResults');
newLabels = labels;