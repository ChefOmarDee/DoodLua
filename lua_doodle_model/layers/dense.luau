--!strict
local DoodleDense = {}

local MAX = math.max
local WORD_SIZE = 4

-- iNodes: Number of input nodes (from flatten step)
-- oNodes: Number of output neurons
function DoodleDense.Dense(
    inputB: buffer, outputB: buffer, weightsB: buffer, biasB: buffer,
    wOffset: number, bOffset: number,
    iNodes: number, oNodes: number, applyRelu: boolean
)
    -- loop through every output neuron
    for o = 0, oNodes - 1 do
        -- start with the bias for this specific neuron
        local sum = buffer.readf32(biasB, bOffset + (o * WORD_SIZE))
        
        -- dot product: multiply every input by its connecting weight
        for i = 0, iNodes - 1 do
            local inVal = buffer.readf32(inputB, i * WORD_SIZE)
            
            -- karas dense weights are stored linearly as [InputNodes, OutputNodes]
            local weightIdx = ((i * oNodes) + o) * WORD_SIZE
            local wt = buffer.readf32(weightsB, wOffset + weightIdx)
            
            sum += inVal * wt
        end
        
        -- fused activation
        if applyRelu then
            sum = MAX(sum, 0)
        end
        
        -- write to the output buffer
        buffer.writef32(outputB, o * WORD_SIZE, sum)
    end
end

return DoodleDense