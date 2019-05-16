function int32cell = cellInt32(originalCell)
% Convert all the element in the cell to 32-bit integer.
int32cell = cellfun(@int32, originalCell, 'uniformOutput', false);