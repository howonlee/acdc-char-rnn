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
    curr_h2h1, curr_h2h2, curr_h2h3, curr_h2h4 = prev_h, prev_h, prev_h, prev_h
    for i =1,3 do
      curr_h2h1 = acdc.FastACDC(rnn_size, acdcopt)(curr_h2h1)
      curr_h2h2 = acdc.FastACDC(rnn_size, acdcopt)(curr_h2h2)
      curr_h2h3 = acdc.FastACDC(rnn_size, acdcopt)(curr_h2h3)
      curr_h2h4 = acdc.FastACDC(rnn_size, acdcopt)(curr_h2h4)
      -- if i > 6 then
      --   curr_h2h1 = nn.Dropout(dropout)(curr_h2h1)
      --   curr_h2h2 = nn.Dropout(dropout)(curr_h2h2)
      --   curr_h2h3 = nn.Dropout(dropout)(curr_h2h3)
      --   curr_h2h4 = nn.Dropout(dropout)(curr_h2h4)
      -- end
      -- curr_h2h1 = nn.ReLU()(curr_h2h1)
      -- curr_h2h2 = nn.ReLU()(curr_h2h2)
      -- curr_h2h3 = nn.ReLU()(curr_h2h3)
      -- curr_h2h4 = nn.ReLU()(curr_h2h4)
      -- curr_h2h1 = acdc.Permutation(rnn_size)(curr_h2h1)
      -- curr_h2h2 = acdc.Permutation(rnn_size)(curr_h2h2)
      -- curr_h2h3 = acdc.Permutation(rnn_size)(curr_h2h3)
      -- curr_h2h4 = acdc.Permutation(rnn_size)(curr_h2h4)
      -- experiment harder on adding the relus and things
    end

    -- decode the gates
    local gates_1 = nn.CAddTable()({i2h_1, curr_h2h1})
    local gates_2 = nn.CAddTable()({i2h_2, curr_h2h2})
    local gates_3 = nn.CAddTable()({i2h_3, curr_h2h3})
    local gates_4 = nn.CAddTable()({i2h_4, curr_h2h4})
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

