function plotResults(Results,mask)
    if ~exist('mask')
        mask = ones(1,size(Results,2));
    end
    colors = {'b' 'r' 'k' 'g' 'c' 'y' 'm' 'b--' 'r--' 'k--' 'g--' 'c--' 'y--' 'm--' 'b-.' 'r-.' 'k-.' 'g-.' 'c-.' 'y-.' 'm-.' 'b:' 'r:' 'k:' 'g:' 'c:' 'y:' 'm:' };
    figure;
    hold on;
    k = 1;
    for i=1:size(Results,2)
        if ~isempty(Results{i}) & mask(i) == 1
            plot(Results{i}.f,Results{i}.d,colors{k});
            if isfield(Results{i},'params') & isfield(Results{i}.params,'caption')
                strLegend{k} = Results{i}.params.caption;
            else
                strLegend{k} = 'ROC curve';
            end                
            k = k + 1;
        end
    end 
    legend(char(strLegend));
    xlabel('False positive rate');
    ylabel('True positive rate');

