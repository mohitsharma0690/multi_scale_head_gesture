% [modelLDCRF stats] = trainLDCRF(seqs, labels, params)
%     Train LDCRF model based on feature sequences and corresponding labels.
% 
% INPUT:
%    - seqs             : Cell array of matrices which contains the encoded
%                         features for each sequence
%    - labels           : Cell array of vectors which contains the ground
%                         truth label for each sample.
%    - params           : Parameters for the training procedure.
%
%    OUTPUT:
%    - modelLDCRF       : internal paramters from the trained model
%    - stats            : Statistic from the training procedure (e.g.,
%                         gradient norm, likelyhood error, training time)
function [modelLDCRF stats] = trainLDCRF(seqs, labels, params) 

intLabels = cellInt32(labels);

matHCRF('createToolbox','ldcrf',params.optimizer, params.nbHiddenStates, params.windowSize);
matHCRF('setData',seqs,intLabels);
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
if isfield(params,'regL1FeatureTypes')
    matHCRF('set','regL1FeatureTypes',params.regL1FeatureTypes);
end
if isfield(params,'maxIterations')
    matHCRF('set','maxIterations',params.maxIterations);
end
if isfield(params,'initWeights')
    matHCRF('set','initWeights',params.initWeights);
end



matHCRF('train');
[modelLDCRF.model modelLDCRF.features] = matHCRF('getModel');
modelLDCRF.optimizer = params.optimizer;
modelLDCRF.nbHiddenStates = params.nbHiddenStates;
modelLDCRF.windowSize = params.windowSize;
modelLDCRF.debugLevel = params.debugLevel;
modelLDCRF.modelType = params.modelType;
modelLDCRF.caption = params.caption;

stats.NbIterations = matHCRF('get','statsNbIterations');
stats.FunctionError = matHCRF('get','statsFunctionError');
stats.NormGradient = matHCRF('get','statsNormGradient');
