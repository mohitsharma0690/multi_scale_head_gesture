function new_labels = MS_convertToFiveClass(labels)
new_labels = zeros(size(labels));
for i=1:size(labels, 1)
    if labels(i) == 0
        new_labels(i) = 0;
    elseif labels(i) >= 1 && labels(i) <= 5
        new_labels(i) = 1;
    elseif labels(i) == 6
        new_labels(i) = 2;
    elseif labels(i) == 7 || labels(i) == 8
        new_labels(i) = 3;
    elseif labels(i) == 9 || labels(i) == 10
        new_labels(i) = 4;
    else
        assert(2 > 3);
    end
end

end