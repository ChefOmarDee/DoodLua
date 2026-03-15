--!strict
local DoodleCNNModel = require("../model_interface")

-- ==========================================
-- 1. DETERMINISTIC DATA GENERATORS
-- ==========================================
local function CreateTestSquare(): buffer
    local b = buffer.create(28 * 28 * 4)
    for i = 0, (28 * 28) - 1 do buffer.writef32(b, i * 4, 0.0) end
    
    -- Draw 10x10 square (Matching Python 9:19 range)
    for y = 9, 18 do
        for x = 9, 18 do
            buffer.writef32(b, (y * 28 + x) * 4, 1.0)
        end
    end
    return b
end

-- ==========================================
-- 2. CONSOLE VISUALIZER
-- ==========================================
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
local testDrawing = CreateTestSquare()
VisualCheck(testDrawing)

print("Starting Inference Benchmark (50 runs)...")
local start = os.clock()

local winningIdx = DoodleCNNModel.Predict(testDrawing)

local elapsed = os.clock() - start

-- ==========================================
-- 4. RESULTS
-- ==========================================
print("===================================")
print(string.format("Avg Speed: %.2f ms", (elapsed / 50) * 1000))
print("Winning Index: " .. tostring(winningIdx))
print("===================================")