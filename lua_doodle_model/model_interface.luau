--!strict
local DataLoader = require("./utilities/load_weights")
local Conv = require("./layers/convolution_2d_layer")
local Pool = require("./layers/max_pooling_layer")
local Flat = require("./layers/flatten")
local Dense = require("./layers/dense")
local Act = require("./utilities/activation_functions")

-- Allocation: 28*28*64 filters * 4 bytes
local MAX_WORKSPACE_BYTES = 200704 
local WorkspaceA = buffer.create(MAX_WORKSPACE_BYTES)
local WorkspaceB = buffer.create(MAX_WORKSPACE_BYTES)

local function PredictDoodle(rawDrawingBuffer: buffer): number
    buffer.copy(WorkspaceA, 0, rawDrawingBuffer)

    local function bufMax(b: buffer, count: number): number
        local m = -math.huge
        for i = 0, count - 1 do
            local v = buffer.readf32(b, i*4)
            if v > m then m = v end
        end
        return m
    end

    local function readRow(b: buffer, startIdx: number, count: number): string
        local parts = {}
        for i = 0, count - 1 do
            parts[i+1] = string.format("%.6f", buffer.readf32(b, (startIdx + i) * 4))
        end
        return table.concat(parts, "  ")
    end

    -- Input check
    print(string.format("Input:      max=%.6f, [0,0,0]=%.6f", bufMax(WorkspaceA, 28*28), buffer.readf32(WorkspaceA, 0)))

    -- BLOCK 1
    local b, w, bias = DataLoader.GetLayerOffsets(0)
    Conv.Conv2D(WorkspaceA, WorkspaceB, b, b, w, bias, 28, 28, 1, 16, 3)
    print(string.format("After conv1: max=%.6f, [0,0,0]=%.6f, [0,0,1]=%.6f",
        bufMax(WorkspaceB, 28*28*16), buffer.readf32(WorkspaceB, 0), buffer.readf32(WorkspaceB, 4)))

    b, w, bias = DataLoader.GetLayerOffsets(1)
    Conv.Conv2D(WorkspaceB, WorkspaceA, b, b, w, bias, 28, 28, 16, 16, 3)
    print(string.format("After conv2: max=%.6f, [0,0,0]=%.6f, [0,0,1]=%.6f",
        bufMax(WorkspaceA, 28*28*16), buffer.readf32(WorkspaceA, 0), buffer.readf32(WorkspaceA, 4)))

    local h, w_dim = Pool.MaxPool2D(WorkspaceA, WorkspaceB, 28, 28, 16, 2, 2)
    print(string.format("After pool1: max=%.6f, [0,0,0]=%.6f, [0,0,1]=%.6f",
        bufMax(WorkspaceB, h*w_dim*16), buffer.readf32(WorkspaceB, 0), buffer.readf32(WorkspaceB, 4)))

    -- BLOCK 2
    b, w, bias = DataLoader.GetLayerOffsets(2)
    Conv.Conv2D(WorkspaceB, WorkspaceA, b, b, w, bias, h, w_dim, 16, 32, 3)
    print(string.format("After conv3: max=%.6f, [0,0,0]=%.6f, [0,0,1]=%.6f",
        bufMax(WorkspaceA, h*w_dim*32), buffer.readf32(WorkspaceA, 0), buffer.readf32(WorkspaceA, 4)))

    b, w, bias = DataLoader.GetLayerOffsets(3)
    Conv.Conv2D(WorkspaceA, WorkspaceB, b, b, w, bias, h, w_dim, 32, 32, 3)
    print(string.format("After conv4: max=%.6f, [0,0,0]=%.6f, [0,0,1]=%.6f",
        bufMax(WorkspaceB, h*w_dim*32), buffer.readf32(WorkspaceB, 0), buffer.readf32(WorkspaceB, 4)))
    local conv4max = -math.huge
    local conv4maxIdx = 0
    for i = 0, h*w_dim*32 - 1 do
        local v = buffer.readf32(WorkspaceB, i*4)
        if v > conv4max then conv4max = v; conv4maxIdx = i end
    end
    print("After conv4 max value:", conv4max, "flat index:", conv4maxIdx)
    print("conv4 max location: h=", conv4maxIdx // (w_dim*32), "w=", (conv4maxIdx // 32) % w_dim, "c=", conv4maxIdx % 32)

    local row0 = {}
    for col = 0, w_dim - 1 do
        row0[col+1] = string.format("%.6f", buffer.readf32(WorkspaceB, (col*32 + 0)*4))
    end
    print("conv4 [0,:,0]: " .. table.concat(row0, "  "))

    h, w_dim = Pool.MaxPool2D(WorkspaceB, WorkspaceA, h, w_dim, 32, 2, 2)
    print(string.format("After pool2: max=%.6f, [0,0,0]=%.6f, [0,0,1]=%.6f",
        bufMax(WorkspaceA, h*w_dim*32), buffer.readf32(WorkspaceA, 0), buffer.readf32(WorkspaceA, 4)))

    -- BLOCK 3
    b, w, bias = DataLoader.GetLayerOffsets(4)
    Conv.Conv2D(WorkspaceA, WorkspaceB, b, b, w, bias, h, w_dim, 32, 64, 3)
    print(string.format("After conv5: max=%.6f, [0,0,0]=%.6f, [0,0,1]=%.6f",
        bufMax(WorkspaceB, h*w_dim*64), buffer.readf32(WorkspaceB, 0), buffer.readf32(WorkspaceB, 4)))

    b, w, bias = DataLoader.GetLayerOffsets(5)
    Conv.Conv2D(WorkspaceB, WorkspaceA, b, b, w, bias, h, w_dim, 64, 64, 3)
    print(string.format("After conv6: max=%.6f, [0,0,0]=%.6f, [0,0,1]=%.6f",
        bufMax(WorkspaceA, h*w_dim*64), buffer.readf32(WorkspaceA, 0), buffer.readf32(WorkspaceA, 4)))

    h, w_dim = Pool.MaxPool2D(WorkspaceA, WorkspaceB, h, w_dim, 64, 2, 2)
    print(string.format("After pool3: max=%.6f, [0,0,0]=%.6f, [0,0,1]=%.6f",
        bufMax(WorkspaceB, h*w_dim*64), buffer.readf32(WorkspaceB, 0), buffer.readf32(WorkspaceB, 4)))

    -- Pool3 spatial breakdown
    print("Pool3 [0,0,:8]: " .. readRow(WorkspaceB, 0,   8))
    print("Pool3 [0,1,:8]: " .. readRow(WorkspaceB, 64,  8))
    print("Pool3 [0,2,:8]: " .. readRow(WorkspaceB, 128, 8))

    -- Flatten
    local nodes = Flat.Flatten(h, w_dim, 64)
    print(string.format("After flatten: %d nodes", nodes))
    print("First 8: " .. readRow(WorkspaceB, 0, 8))
    print("Last  8: " .. readRow(WorkspaceB, nodes - 8, 8))

    -- Dense 1
    b, w, bias = DataLoader.GetLayerOffsets(6)
    Dense.Dense(WorkspaceB, WorkspaceA, b, b, w, bias, nodes, 512, false)
    for i = 0, 511 do
        local val = buffer.readf32(WorkspaceA, i * 4)
        buffer.writef32(WorkspaceA, i * 4, math.tanh(val))
    end

    -- Dense 2
    b, w, bias = DataLoader.GetLayerOffsets(7)
    Dense.Dense(WorkspaceA, WorkspaceB, b, b, w, bias, 512, 345, false)

    Act.SoftmaxWithBuffer(WorkspaceB, 0, 345)

    local bestIdx = 0
    local maxProb = -math.huge
    for i = 0, 344 do
        local prob = buffer.readf32(WorkspaceB, i * 4)
        if prob > maxProb then
            maxProb = prob
            bestIdx = i
        end
    end

    print(string.format("Winning Probability Score: %.6f", maxProb))
    return bestIdx
end

return { Predict = PredictDoodle }