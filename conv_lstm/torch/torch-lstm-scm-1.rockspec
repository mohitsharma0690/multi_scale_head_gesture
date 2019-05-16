package = "torch-lstm"
version = "scm-1"
description = {
  summary = "Efficient, reusable RNNs and LSTMs for Torch.",
  detailed = [[
  torch-lstm provides efficient torch/nn modules implementing LSTMs and RNNs.
  ]],
}
dependencies = {
  "torch >= 7.0",
  "nn >= 1.0",
}
build = {
  type = "builtin",
  modules = {
    ["torch-lstm.init"] = "init.lua",
    ["torch-lstm.model.LSTM"] = "./model/LSTM.lua",
    ["torch-lstm.model.VanillaLSTM"] = "./model/VanillaLSTM.lua",
  }
}
