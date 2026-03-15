--!strict
local DoodleConvolution = {}

-- localization for "Reduced Scope Lookup" speed
local MAX = math.max
local FLOOR = math.floor

-- since each f32 is 4 bytes, we localize this to avoid extra math in the loops
local WORD_SIZE = 4

--[[
    inputB: Buffer containing the input feature map
    outputB: Destination buffer for results
    weightsB: The raw weights buffer (from your DataLoader)
    biasB: The raw weights buffer (from your DataLoader)
    
    wOffset: Starting byte for this layer's kernel weights
    bOffset: Starting byte for this layer's biases
    
    iH, iW: Input Height/Width
    iC: Input Channels (e.g., 1 for drawing because theres no depth but its 16 if theres 16 filters)
    oC: Output Channels (Number of filters)
    kS: Kernel Size (e.g., 3 for a 3x3)
]]
function DoodleConvolution.Conv2D(
    inputB: buffer, outputB: buffer, weightsB: buffer, biasB: buffer,
    wOffset: number, bOffset: number,
    iH: number, iW: number, iC: number, oC: number, kS: number
)
    -- assuming 'same' padding to maintain image dimensions
    local pad = FLOOR(kS / 2) 
    
    -- loop through every output filter
    for oc = 0, oC - 1 do
        -- get the bias for this specific filter (1 bias per filter)
        local filterBias = buffer.readf32(biasB, bOffset + (oc * WORD_SIZE))
        
        -- loop through every pixel in the image height
        for ih = 0, iH - 1 do
            -- loop through every pixel in the image width
            for iw = 0, iW - 1 do
                local sum = 0
                
                -- convolution: slide the kernel over the neighborhood
                for kh = 0, kS - 1 do
                    for kw = 0, kS - 1 do
                        -- calculate neighborhood coordinates with padding offset
                        local inY = ih + kh - pad
                        local inX = iw + kw - pad
                        
                        -- boundary check (0-padding logic)
                        if inY >= 0 and inY < iH and inX >= 0 and inX < iW then
                            -- loop through input channels (depth)
                            for ic = 0, iC - 1 do
                                -- BUFFER ADDRESS MATH (0-INDEXED):
                                -- input: ((y * W * C) + (x * C) + c) * 4
                                local inputIdx = ((inY * iW * iC) + (inX * iC) + ic) * WORD_SIZE
                                
                                -- weight: ((filter * kH * kW * iC) + (kh * kW * iC) + (kw * iC) + ic) * 4
                                -- Keras kernel layout: (kH, kW, iC, oC)
                                local weightIdx = ((kh * kS * iC * oC) + (kw * iC * oC) + (ic * oC) + oc) * WORD_SIZE
                                local px = buffer.readf32(inputB, inputIdx)
                                local wt = buffer.readf32(weightsB, wOffset + weightIdx)
                                
                                sum += px * wt
                            end
                        end
                    end
                end
                
                local activatedResult = MAX(sum + filterBias, 0)
                
                -- write output in 0-indexed linear order (Height, Width, Channel)
                local outputIdx = ((ih * iW * oC) + (iw * oC) + oc) * WORD_SIZE
                buffer.writef32(outputB, outputIdx, activatedResult)
            end
        end
    end
end

return DoodleConvolution