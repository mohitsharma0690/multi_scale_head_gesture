function [modelHCRF stats] = trainHCRF(seqs, labels, params)
% Train (Gaussian) Hidden-CRF models based on feature sequences and corresponding labels.
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

%if sequences are not "complete" sequence (i.e., with only one label per
%   sequence then segment the sequences based on the labels

% K. Bousmalis Summer 2010 Added option to normalise data, by using HMM's model.
% This however couples the BNT toolbox in here, which might not be
% desirable. 
%
% K. Bousmalis 09/27/2010 Added support for GHCRF

GHCRF = 0; % If this switch is on, Gaussian hCRF will be called instead
if (isfield(params, 'GaussianHCRF') && params.GaussianHCRF)
    GHCRF = 1;
end

nbSeq = numel(seqs);
if max(cellfun(@numel,cellfun(@unique,labels,'UniformOutput',false))) > 1
    ii = 1;
    for i = 1:nbSeq
        %Find transition points
        transIndices = find(labels{i}(2:end) - labels{i}(1:end-1) ~= 0);
        startIndice = 1;
        for k = 1:(numel(transIndices)+1)
            if k > numel(transIndices)
                endIndice = size(seqs{i},2);
            else
                endIndice = transIndices(k);
            end
            segmentSeqs{ii} = seqs{i}(:,startIndice:endIndice);
            segmentLabels{ii} = labels{i}(:,startIndice:endIndice);
            ii = ii + 1;
            startIndice = endIndice+1;
        end
    end
    seqs = segmentSeqs;
    labels = segmentLabels;
end

intLabels = zeros(1,numel(labels));
for i =1:numel(labels)
    intLabels(i) = labels{i}(1);
end
intLabels = int32(intLabels);

% Normalise data {%+KGB}
if isfield(params,'normalise') && params.normalise == 1
    seqs = normaliseCellArray(seqs);
    seqs = removeOutliers(seqs,params.outlierTresh(2),params.outlierTresh(1)); 
end


   
if GHCRF
    matHCRF('createToolbox', 'ghcrf',params.optimizer, params.nbHiddenStates, params.windowSize);
else
    matHCRF('createToolbox', 'hcrf',params.optimizer, params.nbHiddenStates, params.windowSize);
end

% RESET RND GENERATOR TO DESIRED SEED 
if isfield(params, 'useSameSeedPerIterationNb') && params.useSameSeedPerIterationNb %{+KGB}
    assert(isfield(params, 'seeds'), 'modelParams.seeds were not set, but useSameSeedPerIterationNb switch was on'); %{+KGB}
    assert(isfield(params, 'seedIndex'), 'seedIndex not set'); %{+KGB}        
    matHCRF('set', 'randomSeed', params.seeds(params.seedIndex)); %{+KGB}    
end

matHCRF('setData',seqs,[],intLabels);
if isfield(params,'rangeWeights')
    matHCRF('set','minRangeWeights',params.rangeWeights(1));
    matHCRF('set','maxRangeWeights',params.rangeWeights(2));
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
[modelHCRF.model modelHCRF.features] = matHCRF('getModel');
modelHCRF.optimizer = params.optimizer;
modelHCRF.nbHiddenStates = params.nbHiddenStates;
modelHCRF.windowSize = params.windowSize;
modelHCRF.windowRecSize = params.windowRecSize;
modelHCRF.debugLevel = params.debugLevel;
modelHCRF.modelType = params.modelType;
modelHCRF.caption = params.caption;
if isfield(params,'normalise')
    modelHCRF.normalise = params.normalise;
end
if isfield(params,'outlierTresh')
    modelHCRF.outlierTresh = params.outlierTresh; %{+KGB}
end

stats.NbIterations = matHCRF('get','statsNbIterations');
stats.FunctionError = matHCRF('get','statsFunctionError');
stats.NormGradient = matHCRF('get','statsNormGradient');
