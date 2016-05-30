require 'acdc'

local LSTM = {}
function LSTM.lstm(input_size, rnn_size, n, dropout)
  dropout = dropout or 0 

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  acdcopt = {}
  acdcopt.rand_init = true
  acdcopt.sign_init = true
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then 
      x = OneHot(input_size)(inputs[1])
      input_size_L = input_size
      i2h_1 = nn.Linear(input_size_L, rnn_size)(x)
      i2h_2 = nn.Linear(input_size_L, rnn_size)(x)
      i2h_3 = nn.Linear(input_size_L, rnn_size)(x)
      i2h_4 = nn.Linear(input_size_L, rnn_size)(x)
    else 
      x = outputs[(L-1)*2] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
      i2h_1 = acdc.FastACDC(rnn_size, acdcopt)(x)
      i2h_2 = acdc.FastACDC(rnn_size, acdcopt)(x)
      i2h_3 = acdc.FastACDC(rnn_size, acdcopt)(x)
      i2h_4 = acdc.FastACDC(rnn_size, acdcopt)(x)
    end
    -- local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
    -- local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L}
    -- local all_input_sums = nn.CAddTable()({i2h, h2h})
    -- local i2h_1 = acdc.FastACDC(rnn_size)(x)
    local h2h_1 = acdc.FastACDC(rnn_size, acdcopt)(prev_h)
    local h2h_2 = acdc.FastACDC(rnn_size, acdcopt)(prev_h)
    local h2h_3 = acdc.FastACDC(rnn_size, acdcopt)(prev_h)
    local h2h_4 = acdc.FastACDC(rnn_size, acdcopt)(prev_h)

    -- decode the gates
    local pre_gates_1 = nn.CAddTable()({i2h_1, h2h_1})
    local pre_gates_2 = nn.CAddTable()({i2h_2, h2h_2})
    local pre_gates_3 = nn.CAddTable()({i2h_3, h2h_3})
    local pre_gates_4 = nn.CAddTable()({i2h_4, h2h_4})
    local gates_1 = acdc.Permutation(rnn_size)(pre_gates_1)
    local gates_2 = acdc.Permutation(rnn_size)(pre_gates_2)
    local gates_3 = acdc.Permutation(rnn_size)(pre_gates_3)
    local gates_4 = acdc.Permutation(rnn_size)(pre_gates_4)
    -- local gates_1 = nn.CAddTable()({i2h_1, h2h_1})
    -- local gates_2 = nn.CAddTable()({i2h_2, h2h_2})
    -- local gates_3 = nn.CAddTable()({i2h_3, h2h_3})
    -- local gates_4 = nn.CAddTable()({i2h_4, h2h_4})
    local in_gate          = nn.Sigmoid()(gates_1)
    local in_transform     = nn.Tanh()(gates_2)
    local forget_gate      = nn.Sigmoid()(gates_3)
    local out_gate         = nn.Sigmoid()(gates_4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnn_size, input_size)(top_h):annotate{name='decoder'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

return LSTM

