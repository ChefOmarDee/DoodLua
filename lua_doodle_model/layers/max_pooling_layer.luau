--!strict
local DoodleMaxPooling = {}

local MAX = math.max
local HUGE = math.huge
local FLOOR = math.floor

-- localize byte size for speed
local WORD_SIZE = 4

function DoodleMaxPooling.MaxPool2D(
    inputB: buffer, outputB: buffer,
    iH: number, iW: number, iC: number,
    poolSize: number, stride: number
): (number, number)
    -- calculate the new shrunken dimensions
    local oH = FLOOR(iH / stride)
    local oW = FLOOR(iW / stride)

    -- loop through the new smaller grid
    for oh = 0, oH - 1 do
        for ow = 0, oW - 1 do
            -- process each channel independently
            for c = 0, iC - 1 do
                local maxVal = -HUGE
                
                -- slide the pooling window (usually 2x2)
                for ph = 0, poolSize - 1 do
                    for pw = 0, poolSize - 1 do
                        -- map to the original input grid coordinates
                        local ih = (oh * stride) + ph
                        local iw = (ow * stride) + pw
                        
                        -- 0-indexed HWC format math
                        local inputIdx = ((ih * iW * iC) + (iw * iC) + c) * WORD_SIZE
                        local val = buffer.readf32(inputB, inputIdx)
                        
                        if val > maxVal then maxVal = val end
                    end
                end
                
                -- write the single highest value to the new shrunken buffer
                local outputIdx = ((oh * oW * iC) + (ow * iC) + c) * WORD_SIZE
                buffer.writef32(outputB, outputIdx, maxVal)
            end
        end
    end
    
    -- return the new height and width for the next layer to use
    return oH, oW 
end

return DoodleMaxPooling