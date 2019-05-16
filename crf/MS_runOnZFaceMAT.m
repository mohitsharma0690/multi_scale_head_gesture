function MS_runOnZFaceMAT(train_data_mat, train_labels_mat, test_data_mat, ...
    test_labels_mat)

train_data = load(train_data_mat);
train_data = train_data.seq;
train_labels = load(train_labels_mat);
train_labels = cellInt32(train_labels.labels);
%for i=1:size(train_labels)
%    train_labels{i} = MS_convertOneHotToNormal(train_labels{i});
%end

model = train(train_data, train_labels);

test_data = load(test_data_mat);
test_data = test_data.seq;
test_labels = load(test_labels_mat);
test_labels = cellInt32(test_labels.labels);
% for i=1:size(test_labels)
%     test_labels{i} = MS_convertToFiveClass(test_labels{i});
% end

results = test(test_data, test_labels, model);
% Results are in one-hot encoding format apparently
save('results.mat', 'results');

results = MS_convertOneHotToNormal(results);
conf = MS_getConfMatrix(test_labels, results);
save('conf.mat', 'conf');

end

function [model, stats] = train(train_data, train_labels)

model_type = 'ldcrf';
num_hidden_states = 5;
optimizer_type = 'bfgs';
window_size = 0;

matHCRF('createToolbox','ldcrf', optimizer_type, num_hidden_states, window_size);
matHCRF('setData', train_data, train_labels);

matHCRF('train');
fprintf('Did train the model\n');
[modelLDCRF.model modelLDCRF.features] = matHCRF('getModel');
modelLDCRF.optimizer = optimizer_type;
modelLDCRF.nbHiddenStates = num_hidden_states;
modelLDCRF.windowSize = window_size;
% modelLDCRF.debugLevel = params.debugLevel;
modelLDCRF.modelType = model_type;
% modelLDCRF.caption = params.caption;

model = modelLDCRF;

end

function [ll] = test(test_data, test_labels, modelLDCRF)
matHCRF('createToolbox','ldcrf',modelLDCRF.optimizer, modelLDCRF.nbHiddenStates, modelLDCRF.windowSize);
matHCRF('setData', test_data, test_labels);
matHCRF('setModel',modelLDCRF.model, modelLDCRF.features);
if isfield(modelLDCRF,'debugLevel')
    matHCRF('set','debugLevel',modelLDCRF.debugLevel);
end
fprintf('Will test the model\n');
matHCRF('test');
fprintf('Did test the model\n');

ll=matHCRF('getResults');
end

function int32cell = cellInt32(originalCell)
% Convert all the element in the cell to 32-bit integer.
int32cell = cellfun(@int32, originalCell, 'uniformOutput', false);
end

