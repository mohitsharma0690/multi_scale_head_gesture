% [ll newlabels] = test(model, seqs, labels)
%     Test sequences : Compute the log likelihood for each  sequence. Also,
%     based on the labels, compute error measurements. 
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

function [ll newlabels] = test(model, seqs, labels)

if strcmp(model.modelType,'crf')
    [ll newlabels] = testCRF(model, seqs, labels);
elseif strcmp(model.modelType,'ldcrf') || strcmp(model.modelType,'fhcrf')
    [ll newlabels] = testLDCRF(model, seqs, labels);
elseif strcmp(model.modelType,'sldcrf')
    [ll newlabels] = testSLDCRF(model, seqs, labels);
elseif strcmp(model.modelType,'hcrf') || strcmp(model.modelType,'ghcrf')
    [ll newlabels] = testHCRF(model, seqs, labels);
elseif strcmp(model.modelType,'svm')
    [ll newlabels] = testSVM(model, seqs, labels);
elseif strcmp(model.modelType,'hmm')
    [ll newlabels] = testHMM(model, seqs, labels);
elseif strcmp(model.modelType,'hhmm')
    [ll newlabels] = testHHMM(model, seqs, labels);
end
