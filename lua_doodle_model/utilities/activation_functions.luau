--!strict
local DoodleActivationFunction={}

-- this E is basically luau's implementation of e, we localize it so we dont waste speed pulling from global which causes Reduced Scope Lookup
local E = math.exp
-- local MAX = math.max
local HUGE = math.huge

--MAKE SURE THE TABLE PASSED IN IS 0 INDEXED
function DoodleActivationFunction.Softmax(inputNumbers:{number}): {number}
    --preallocate the table because its much much faster, REMEMBER #TABLE DOESNT COUNT FOR 0 INDEX
    local probabilities:{number}=table.create(#inputNumbers+1)
    local sumOfExpScores=0
    local maxVal = -HUGE

    for i = 0, #inputNumbers do
        local val = inputNumbers[i]
        if val > maxVal then maxVal = val end
    end

    -- get the softmax denominator and will be used for normalization
    for i = 0, #inputNumbers do
        sumOfExpScores += E(inputNumbers[i] - maxVal)
    end

    --get the probability softmax value for all the input variables
    for i = 0, #inputNumbers do
        probabilities[i] = E(inputNumbers[i] - maxVal) / sumOfExpScores
    end

    return probabilities
end

--NOTE: BUFFERS ARE NATIVELY 0 INDEXED
function DoodleActivationFunction.SoftmaxWithBuffer(b: buffer, offset: number, count: number)
    local sumOfExpScores = 0
    local maxVal = -HUGE

    -- find the max value for numerical stability to prevent exp overflow
    for i = 0, count - 1 do
        -- we multiply by 4 to step through the f32 memory addresses
        local val = buffer.readf32(b, offset + (i * 4))
        if val > maxVal then maxVal = val end
    end

    -- get the softmax denominator and will be used for normalization
    for i = 0, count - 1 do
        local val = buffer.readf32(b, offset + (i * 4))
        sumOfExpScores += E(val - maxVal)
    end

    -- get the probability softmax value for all the input variables
    for i = 0, count - 1 do
        local val = buffer.readf32(b, offset + (i * 4))
        local probability = E(val - maxVal) / sumOfExpScores
        
        -- overwrite the buffer directly to save on table creation costs
        buffer.writef32(b, offset + (i * 4), probability)
    end
end


--its better just to hard code the function instead of calling it so many times
-- function DoodleActivationFunction.Relu(inputNumber: number): number
--     return MAX(inputNumber, 0)
-- end

--note: luau already has a tanh so we didnt have to make one
-- function DoodleActivationFunction.Tanh(inputNumber: number): number
    
--     return (E(inputNumber)-E(-inputNumber))/(E(inputNumber)+E(-inputNumber))
-- end

return DoodleActivationFunction