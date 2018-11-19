function [TrIdx, TeIdx] = rnd_selection(labels, num)
    
    TrIdx = false(size(labels));
    numClasses = max(labels);
    samplePerClass = numel(labels) / numClasses;
    
    for i = 1:samplePerClass:numel(labels)
        rndInd = randperm(samplePerClass) + i - 1;
        TrIdx(rndInd(1:num)) = true;
    end
    
    TeIdx = ~TrIdx;
end