% [model stats] = train(seqs, labels, params)
%     Train model based on feature sequences and corresponding labels.
% 
% INPUT:
%    - seqs             : Cell array of matrices which contains the encoded
%                         features for each sequence
%    - labels           : Cell array of vectors which contains the ground
%                         truth label for each sample.
%    - params            : Parameters for the training procedure.
%
%    OUTPUT:
%    - model            : internal paramters from the trained model
%    - stats            : Statistic from the training procedure (e.g.,
%                         gradient norm, likelyhood error, training time)

function [model stats] = train(seqs, labels, params)
if isfield(params,'seqWeights') && isfield(params, 'normalizeWeights')
    params.seqWeights = params.seqWeights/sum(params.seqWeights);
end
startTime = datestr(now,0);
switch params.modelType
    case 'crf'
        [model stats] = trainCRF(seqs, labels, params);
    case 'crfRealtime'
        [model stats] = trainCRFRealtime(seqs, labels, params); 
    case {'ldcrf', 'fhcrf'}
        [model stats] = trainLDCRF(seqs, labels, params);
    case {'sldcrf'}
        [model stats] = trainSLDCRF(seqs, labels, params);
    case {'hcrf', 'ghcrf'}
        [model stats] = trainHCRF(seqs, labels, params);
    case 'svm'
        [model stats] = trainSVM(seqs, labels, params);
    case 'hmm'
        [model stats] = trainHMM(seqs, labels, params);
    case 'hhmm'
        [model stats] = trainHHMM(seqs, labels, params);
    case 'lvperceptron'
        [model stats] = trainLVPerceptron(seqs, labels, params);
end
stats.startTime = startTime;
stats.endTime = datestr(now,0);
    
    