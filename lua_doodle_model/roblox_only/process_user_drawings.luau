local GRID    = 28
local PADDING = 30 

local ProcessUserDrawingModuleScript = {}

-- eead a pixel from the source buffer as a 0-255 ink value.
local function getInk(buf, col, row, width, height): number
	if col < 1 or col > width or row < 1 or row > height then
		return 0   -- out-of-bounds → blank
	end
	local raw = buffer.readu8(buf, (row - 1) * width + (col - 1))
	return 255 - raw   -- invertpixel color
end

-- this approximates LANCZOS by averaging the source region that maps to each pixel
local function sampleBox(buf, cx, cy, radius, width, height): number
	local r   = math.max(1, math.floor(radius))
	local sum = 0
	local count = 0
	for dy = -r, r do
		for dx = -r, r do
			local sx = math.clamp(math.floor(cx + 0.5) + dx, 1, width)
			local sy = math.clamp(math.floor(cy + 0.5) + dy, 1, height)
			sum   = sum   + getInk(buf, sx, sy, width, height)
			count = count + 1
		end
	end
	return count > 0 and (sum / count) or 0
end

-- write a float32 (little-endian) into a buffer at byte offset.
local function writeFloat32(buf, offset, value)
	buffer.writef32(buf, offset, value)
end

function ProcessUserDrawingModuleScript.DrawingToModelInput(
	buf: buffer,
	width: number,
	height: number
): buffer

	--get bounding box
	local minCol, minRow = width + 1, height + 1
	local maxCol, maxRow = 0, 0
	local hasInk = false

	for row = 1, height do
		for col = 1, width do
			if getInk(buf, col, row, width, height) > 0 then
				hasInk = true
				if col < minCol then minCol = col end
				if col > maxCol then maxCol = col end
				if row < minRow then minRow = row end
				if row > maxRow then maxRow = row end
			end
		end
	end

	--thi scode is for if canvas is empty
	local outBuf = buffer.create(GRID * GRID * 4)   
	if not hasInk then
		for i = 0, GRID * GRID - 1 do
			writeFloat32(outBuf, i * 4, 0.0)
		end
		return outBuf
	end

	-- padding
	local left   = minCol - PADDING
	local top    = minRow - PADDING
	local right  = maxCol + PADDING
	local bottom = maxRow + PADDING

	--make square 
	local bw = right  - left
	local bh = bottom - top
	local sz = math.max(bw, bh)
	local cx = (left + right)  / 2
	local cy = (top  + bottom) / 2

	local sqLeft   = cx - sz / 2
	local sqTop    = cy - sz / 2
	local sqRight  = cx + sz / 2
	local sqBottom = cy + sz / 2
	local scaleX = (sqRight  - sqLeft)  / GRID
	local scaleY = (sqBottom - sqTop)   / GRID
	local radius = math.max(scaleX, scaleY) / 2

	for outRow = 1, GRID do
		for outCol = 1, GRID do
			--normalised position within the square crop to a float that is hbetween 0 and 1,
			local u = (outCol - 0.5) / GRID
			local v = (outRow  - 0.5) / GRID

			local srcX = sqLeft + u * (sqRight  - sqLeft)
			local srcY = sqTop  + v * (sqBottom - sqTop)

			local inkVal    = sampleBox(buf, srcX, srcY, radius, width, height)
			local normalized = math.clamp(inkVal / 255.0, 0.0, 1.0)

			local idx = (outRow - 1) * GRID + (outCol - 1)
			writeFloat32(outBuf, idx * 4, normalized)
		end
	end

	return outBuf
end

return ProcessUserDrawingModuleScript