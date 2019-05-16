load sampleData;

paramsData.weightsPerSequence = ones(1,128) ;
paramsData.factorSeqWeights = 1;

paramsNodCRF.normalizeWeights = 1;
R{1}.params = paramsNodCRF;
[R{1}.model R{1}.stats] = train(trainSeqs, trainLabels, R{1}.params);
[R{1}.ll R{1}.labels] = test(R{1}.model, testSeqs, testLabels);

paramsNodHCRF.normalizeWeights = 1;
R{2}.params = paramsNodHCRF;
[R{2}.model R{2}.stats] = train(trainCompleteSeqs, trainCompleteLabels, R{2}.params);
[R{2}.ll R{2}.labels] = test(R{2}.model, testSeqs, testLabels);

paramsNodLDCRF.normalizeWeights = 1;
R{3}.params = paramsNodLDCRF;
[R{3}.model R{3}.stats] = train(trainSeqs, trainLabels, R{3}.params);
[R{3}.ll R{3}.labels] = test(R{3}.model, testSeqs, testLabels);

%plotResults(R);
