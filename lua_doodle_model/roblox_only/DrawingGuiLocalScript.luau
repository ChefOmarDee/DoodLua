local UserInputService = game:GetService("UserInputService")
local ReplicatedStorage = game:GetService("ReplicatedStorage")

local canvas = script.Parent:WaitForChild("CanvasFrame")
local sendDrawing = ReplicatedStorage:WaitForChild("SendDrawing")
local CanvasDraw = require(ReplicatedStorage:WaitForChild("CanvasDraw"))

local BRUSH_SIZE = 12

local drawing = false
local lastPosition = nil

task.wait(3)

local screenSize = canvas.AbsoluteSize
print("Screen size:", screenSize.X, screenSize.Y)

local GRID_WIDTH = 448
--local GRID_WIDTH = math.min(math.floor(screenSize.X), 448)
--local GRID_HEIGHT = math.min(math.floor(screenSize.Y), 448)
local GRID_HEIGHT = 448
print(screenSize.Y)

canvas.Size = UDim2.new(0, GRID_WIDTH, 0, GRID_HEIGHT)

task.wait()

print("Grid size:", GRID_WIDTH, GRID_HEIGHT)

local grid = {}
local function resetGrid()
	for i = 1, GRID_WIDTH * GRID_HEIGHT do
		grid[i] = 255.0
	end
end
resetGrid()

local myCanvas = CanvasDraw.new(canvas, Vector2.new(GRID_WIDTH, GRID_HEIGHT), Color3.new(1, 1, 1))

local function isInsideCanvas(position)
	local canvasPos = canvas.AbsolutePosition
	local canvasSz = canvas.AbsoluteSize
	return position.X >= canvasPos.X and position.X <= canvasPos.X + canvasSz.X
		and position.Y >= canvasPos.Y and position.Y <= canvasPos.Y + canvasSz.Y
end

local function markGrid(relX, relY)
	local centerCol = math.clamp(math.floor(relX * GRID_WIDTH) + 1, 1, GRID_WIDTH)
	local centerRow = math.clamp(math.floor(relY * GRID_HEIGHT) + 1, 1, GRID_HEIGHT)

	for dy = -BRUSH_SIZE, BRUSH_SIZE do
		for dx = -BRUSH_SIZE, BRUSH_SIZE do
			if dx * dx + dy * dy <= BRUSH_SIZE * BRUSH_SIZE then
				local col = math.clamp(centerCol + dx, 1, GRID_WIDTH)
				local row = math.clamp(centerRow + dy, 1, GRID_HEIGHT)
				grid[(row - 1) * GRID_WIDTH + col] = 0.0
			end
		end
	end
end

local function drawDot(position)
	local canvasPos = canvas.AbsolutePosition
	local canvasSz = canvas.AbsoluteSize
	local relX = (position.X - canvasPos.X) / canvasSz.X
	local relY = (position.Y - canvasPos.Y) / canvasSz.Y
	if relX < 0 or relX > 1 or relY < 0 or relY > 1 then return end

	markGrid(relX, relY)

	local px = math.clamp(math.floor(relX * GRID_WIDTH) + 1, 1, GRID_WIDTH)
	local py = math.clamp(math.floor(relY * GRID_HEIGHT) + 1, 1, GRID_HEIGHT)
	myCanvas:DrawCircle(Vector2.new(px, py), BRUSH_SIZE, Color3.new(0, 0, 0), 1)
end

local function drawLine(from, to)
	local canvasPos = canvas.AbsolutePosition
	local canvasSz = canvas.AbsoluteSize

	local fromPx = Vector2.new(
		math.clamp(math.floor((from.X - canvasPos.X) / canvasSz.X * GRID_WIDTH) + 1, 1, GRID_WIDTH),
		math.clamp(math.floor((from.Y - canvasPos.Y) / canvasSz.Y * GRID_HEIGHT) + 1, 1, GRID_HEIGHT)
	)
	local toPx = Vector2.new(
		math.clamp(math.floor((to.X - canvasPos.X) / canvasSz.X * GRID_WIDTH) + 1, 1, GRID_WIDTH),
		math.clamp(math.floor((to.Y - canvasPos.Y) / canvasSz.Y * GRID_HEIGHT) + 1, 1, GRID_HEIGHT)
	)

	local steps = math.ceil((toPx - fromPx).Magnitude)
	for i = 0, steps do
		local t = if steps == 0 then 0 else i / steps
		local relX = ((fromPx.X + (toPx.X - fromPx.X) * t) - 1) / GRID_WIDTH
		local relY = ((fromPx.Y + (toPx.Y - fromPx.Y) * t) - 1) / GRID_HEIGHT
		markGrid(relX, relY)
	end

	myCanvas:DrawLine(fromPx, toPx, Color3.new(0, 0, 0), BRUSH_SIZE)
end

UserInputService.InputBegan:Connect(function(input)
	if input.UserInputType == Enum.UserInputType.MouseButton1 or
		input.UserInputType == Enum.UserInputType.Touch then
		if isInsideCanvas(input.Position) then
			drawing = true
			lastPosition = input.Position
			drawDot(input.Position)
		end
	end
end)

UserInputService.InputEnded:Connect(function(input)
	if input.UserInputType == Enum.UserInputType.MouseButton1 or
		input.UserInputType == Enum.UserInputType.Touch then
		drawing = false
		lastPosition = nil
	end
end)

UserInputService.InputChanged:Connect(function(input)
	if drawing and (input.UserInputType == Enum.UserInputType.MouseMovement or
		input.UserInputType == Enum.UserInputType.Touch) then
		if lastPosition then
			drawLine(lastPosition, input.Position)
		end
		lastPosition = input.Position
	end
end)

local clearButton = script.Parent:WaitForChild("ClearButton")
clearButton.Activated:Connect(function()
	resetGrid()
	myCanvas:Fill(Color3.new(1, 1, 1))
end)

local sendButton = script.Parent:WaitForChild("SendButton")
sendButton.Activated:Connect(function()
	local buf = buffer.create(GRID_WIDTH * GRID_HEIGHT)
	for i = 1, GRID_WIDTH * GRID_HEIGHT do
		buffer.writeu8(buf, i - 1, grid[i])
	end
	sendDrawing:FireServer(buf, GRID_WIDTH, GRID_HEIGHT)
end)