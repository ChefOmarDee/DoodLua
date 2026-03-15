--!strict
local DoodleFlatten = {}

-- Because we use the channels-last (HWC) format in a 1D buffer,
-- the data is already perfectly formatted for Keras's Dense layer.
-- This function simply calculates the 1D size.
function DoodleFlatten.Flatten(iH: number, iW: number, iC: number): number
    -- Returns the total number of 1D nodes for the Dense layer
    return iH * iW * iC
end

return DoodleFlatten