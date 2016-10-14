local M = {}

local function convertToTensor(file)
   local data, labels
   for line in io.lines(file) do 
      local tmp = {}
      for word in string.gmatch(line, '([^,]+)') do
         word = tonumber(word) 
         tmp[#tmp+1] = word
      end
      label = table.remove(tmp, 1)
      tmp = torch.Tensor(tmp):byte()
      label = label + 1
      -- this part you should use int() to output test filenames. 
      -- byte() will convert file names to 0-255
      label = torch.Tensor({label}):int()
      if not data then
         data = tmp
         labels = label
      else
         data = torch.cat(data, tmp, 1)
         labels = torch.cat(labels, label)
      end
      collectgarbage()
   end
   return {
      data = data:contiguous():view(-1, 3, 32, 32),
      labels = labels,
   }
end
--------------------------------Modify this part 
function M.exec(opt, cacheFile)
   -- the path should be your training and test data
   local trainData = convertToTensor('gen/testHead100')
   local testData = convertToTensor('gen/testPixelData3')

   print(" | saving CIFAR-10 dataset to " .. cacheFile)
   torch.save(cacheFile, {
      train = trainData,
      val = testData,
   })
end
------------------------------------------------
return M
