--!strict
local DoodleCNNModel = require("../model_interface")
-- local Categories = require("./Categories")

-- ==========================================
-- 1. DUMMY DATA GENERATOR
-- ==========================================
local function CreateTestSquare(): buffer
    local b = buffer.create(28 * 28 * 4)
    -- Fill with 0 (Black)
    for i = 0, (28 * 28) - 1 do
        buffer.writef32(b, i * 4, 0.0)
    end
    
    -- Draw a 10x10 "White" Square (1.0)
    for y = 9, 19 do
        for x = 9, 19 do
            local offset = (y * 28 + x) * 4
            buffer.writef32(b, offset, 1.0)
        end
    end
    return b
end
local function CreateTestCircle(): buffer
    local b = buffer.create(28 * 28 * 4)
    local centerX, centerY = 14, 14
    local radius = 8

    for y = 0, 27 do
        for x = 0, 27 do
            -- Pythagorean theorem to find distance from center
            local dist = math.sqrt((x - centerX)^2 + (y - centerY)^2)
            local val = (dist < radius + 0.5 and dist > radius - 1.5) and 1.0 or 0.0
            buffer.writef32(b, (y * 28 + x) * 4, val)
        end
    end
    return b
end

-- ==========================================
-- 2. CONSOLE VISUALIZER
-- ==========================================
-- This prints a small ASCII version of your buffer so you can verify the shape
local function VisualCheck(b: buffer)
    print("\n--- BUFFER VISUAL CHECK (28x28) ---")
    for y = 0, 27 do
        local row = ""
        for x = 0, 27 do
            local val = buffer.readf32(b, (y * 28 + x) * 4)
            row ..= (val > 0.5 and "██" or "  ")
        end
        print(row)
    end
    print("-----------------------------------\n")
end

-- ==========================================
-- 3. EXECUTION & BENCHMARK
-- ==========================================
local testDrawing = CreateTestCircle()
VisualCheck(testDrawing)

print("Starting Inference Benchmark (100 runs)...")

local totalTime = 0

local start = os.clock()
local doodlePrediction = DoodleCNNModel.Predict(testDrawing)
totalTime += (os.clock() - start)

-- ==========================================
-- 4. RESULTS
-- ==========================================

print("===================================")
print(string.format("Average Speed:  %.2f ms per doodle", totalTime))
print("-----------------------------------")
print("AI Classification: " .. "???" .. " (Index: " .. doodlePrediction .. ")")
print("===================================")