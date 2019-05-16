function MS_runOnZFaceCSV(train_data_csv, train_labels_csv, test_data_csv, ...
    test_labels_csv)

model = train(train_data_csv, train_labels_csv);
results = test(test_data_csv, test_labels_csv, model);
print(results);

end

function [model, stats] = train(train_data_csv, train_labels_csv)

model_type = 'ldcrf';
num_hidden_states = 5;
optimizer_type = 'bfgs';
window_size = 0;

matHCRF('createToolbox','ldcrf', optimizer_type, num_hidden_states, window_size);
matHCRF('loadData', train_data_csv, train_labels_csv);

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

function [ll] = test(test_data_csv, test_labels_csv, modelLDCRF)
matHCRF('createToolbox','ldcrf',modelLDCRF.optimizer, modelLDCRF.nbHiddenStates, modelLDCRF.windowSize);
matHCRF('loadData', test_data_csv, test_labels_csv);
matHCRF('setModel',modelLDCRF.model, modelLDCRF.features);
if isfield(modelLDCRF,'debugLevel')
    matHCRF('set','debugLevel',modelLDCRF.debugLevel);
end
fprintf('Will test the model\n');
matHCRF('test');
fprintf('Did test the model\n');

ll=matHCRF('getResults');
end

