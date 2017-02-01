require 'nn'

local ReQU, Parent = torch.class('nn.ReQU', 'nn.Module')

function ReQU:__init()
  Parent.__init(self)
end

function ReQU:updateOutput(input)
  self.output:resizeAs(input):copy(input)
  self.output[self.output:le(0)] = 0
  self.output:pow(2)
  return self.output
end

function ReQU:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  local jacobian = torch.Tensor()
  jacobian:resizeAs(input):copy(input)
  jacobian[jacobian:le(0)] = 0
  jacobian:mul(2)
  self.gradInput:cmul(jacobian)
  return self.gradInput
end
