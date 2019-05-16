% [model stats] = trainCRF(seqs, labels, params)
%     Train CRF model based on feature sequences and corresponding labels.
% 
% INPUT:
%    - seqs             : Cell array of matrices which contains the encoded
%                         features for each sequence
%    - labels           : Cell array of vectors which contains the ground
%                         truth label for each sample.
%    - params           : Parameters for the training procedure.
%
%    OUTPUT:
%    - model            : internal paramters from the trained model
%    - stats            : Statistic from the training procedure (e.g.,
%                         gradient norm, likelyhood error, training time)
function [modelCRF stats] = trainCRF(seqs, labels, params)

intLabels = cellInt32(labels);

matHCRF('createToolbox','crf',params.optimizer, 0, params.windowSize);
if isfield(params,'seqWeights') && isfield(params,'factorSeqWeights') && params.factorSeqWeights ~= 0
    validW = params.seqWeights ~=0;
    params.seqWeights(validW) = params.seqWeights(validW) * params.factorSeqWeights + 1 - params.factorSeqWeights;
    matHCRF('setData',seqs,intLabels,[],params.seqWeights);
else
    matHCRF('setData',seqs,intLabels);
end
if isfield(params,'rangeWeights')
    matHCRF('set','minRangeWeights',params.rangeWeights(1));
    matHCRF('set','maxRangeWeights',params.rangeWeights(2));
end
if isfield(params,'weightsInitType')
    matHCRF('set','weightsInitType',params.weightsInitType);
end  
if isfield(params,'debugLevel')
    matHCRF('set','debugLevel',params.debugLevel);
end
if isfield(params,'regFactor') % For backward compatibility
    matHCRF('set','regularizationL2',params.regFactor);
end
if isfield(params,'regFactorL2')
    matHCRF('set','regularizationL2',params.regFactorL2);
end
if isfield(params,'regFactorL1')
    matHCRF('set','regularizationL1',params.regFactorL1);
end
if isfield(params,'maxIterations')
    matHCRF('set','maxIterations',params.maxIterations);
end
if isfield(params,'regL1FeatureTypes')
    matHCRF('set','regL1FeatureTypes',params.regL1FeatureTypes);
end

if isfield(params,'initWeights')
    matHCRF('set','initWeights',params.initWeights);
end


matHCRF('train');
[modelCRF.model modelCRF.features] = matHCRF('getModel');
modelCRF.optimizer = params.optimizer;
modelCRF.windowSize = params.windowSize;
modelCRF.debugLevel = params.debugLevel;
modelCRF.modelType = params.modelType;
modelCRF.caption = params.caption;

stats.NbIterations = matHCRF('get','statsNbIterations');
stats.FunctionError = matHCRF('get','statsFunctionError');
stats.NormGradient = matHCRF('get','statsNormGradient');

