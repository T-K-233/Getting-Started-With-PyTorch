import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import json

from omegaconf import OmegaConf
import torch
import numpy as np

from dataloader import Dataloader
from model import EncoderRNN


conf = OmegaConf.load('./configs/train.yaml')

dataloader = Dataloader(conf.dataloader.path)
model = EncoderRNN(dataloader.n_letters, conf.model.hidden_layer_size, dataloader.n_categories)
model.load("./logs/model.pth")

print("shape of network:", dataloader.n_letters, conf.model.hidden_layer_size, dataloader.n_categories)


lib_code = """
def nn_linear(input, weight_T, bias):
    return np.matmul(input, weight_T) + bias

def nn_logsoftmax(input):
    return np.log(np.exp(input) / np.sum(np.exp(input), axis=1))


"""




output_code = ""
output_weights = ""

INDENT = "    "

output_code += "def forward(input_out, hidden):\n"

# this is a hack to get RNN working
output_code += INDENT+"input_out = np.concatenate((input_out, hidden), axis=1)\n"

prev_layer_name = "input"
for layer_name, module in model.named_modules():
    
    print("Find network:", layer_name, type(module))
    if type(module) == type(model):
        # this is just wrapper
        continue
    if type(module) == torch.nn.Linear:
        for key in module.state_dict():
            array = module.state_dict()[key].numpy()
            
            if key == "weight":
                print("  ", key, "\t:", array.shape[0], "x", array.shape[1])
                # store the transposed array in advance
                array = array.T
                output_weights += "{layer_name}_weight_rows = {value}\n".format(layer_name=layer_name, value=array.shape[0])
                output_weights += "{layer_name}_weight_cols = {value}\n".format(layer_name=layer_name, value=array.shape[1])
                output_weights += "{layer_name}_weight_T = np.array([".format(layer_name=layer_name)
            elif key == "bias":
                print("  ", key, "\t:", array.shape[0])
                output_weights += "{layer_name}_bias_rows = {value}\n".format(layer_name=layer_name, value=1)
                output_weights += "{layer_name}_bias_cols = {value}\n".format(layer_name=layer_name, value=array.shape[0])
                output_weights += "{layer_name}_bias = np.array([".format(layer_name=layer_name)
                
            flat_array = np.ndarray.flatten(array)
            for i in range(flat_array.shape[0]-1):
                output_weights += "{value}, ".format(value=flat_array[i])
            output_weights += "{value}".format(value=flat_array[flat_array.shape[0]-1])
            output_weights += "]).reshape({layer_name}_{key}_rows, {layer_name}_{key}_cols)\n".format(layer_name=layer_name, key=key)
        output_weights += "\n"

        output_code += INDENT+"# Linear\n"
        output_code += INDENT+"{layer_name}_out = nn_linear({prev_layer_name}_out, {layer_name}_weight_T, {layer_name}_bias)\n".format(layer_name=layer_name, prev_layer_name=prev_layer_name)
        
    if type(module) == torch.nn.LogSoftmax:
        output_code += INDENT+"# Log Softmax\n"
        output_code += INDENT+"{layer_name}_out = nn_logsoftmax({prev_layer_name}_out)\n".format(layer_name=layer_name, prev_layer_name=prev_layer_name)

        
    prev_layer_name = layer_name
    
output_code += INDENT+"return {prev_layer_name}_out\n".format(prev_layer_name=prev_layer_name)


with open("test.py", "w") as f:
    f.write("import numpy as np\n\n")
    
    f.write(lib_code)

    f.write(output_weights)
    f.write("\n\n")

    f.write(output_code)
    f.write("\n\n")

    f.write("print(forward(np.zeros((1,57)), np.zeros((1,32))))")


quit()



model_dict = dict(model.state_dict())


with open("weights.h", "w") as f:
    f.write("#include <float.h>\n")
    f.write("#include <stddef.h>\n")
    f.write("\n")
    

    for key in model_dict:
        print(key)
        array = model_dict[key].numpy()
        layer_name = key.split(".")[0]

        if ".weight" in key:
            # store the transposed array in advance
            array = array.T
            f.write("const static size_t {layer_name}_WEIGHT_T_ROWS = {value};\n".format(layer_name=layer_name.upper(), value=array.shape[0]))
            f.write("const static size_t {layer_name}_WEIGHT_T_COLS = {value};\n".format(layer_name=layer_name.upper(), value=array.shape[1]))
            f.write("const static float {layer_name}_WEIGHT_DATA[] = {{".format(layer_name=layer_name.upper()))
        elif ".bias" in key:
            f.write("const static size_t {layer_name}_BIAS_ROWS = {value};\n".format(layer_name=layer_name.upper(), value=1))
            f.write("const static size_t {layer_name}_BIAS_COLS = {value};\n".format(layer_name=layer_name.upper(), value=array.shape[0]))
            f.write("const static float {layer_name}_BIAS_DATA[] = {{".format(layer_name=layer_name.upper()))
            
        flat_array = np.ndarray.flatten(array)
        for i in range(flat_array.shape[0]-1):
            f.write("{value}, ".format(value=flat_array[i]))
        f.write("{value}".format(value=flat_array[flat_array.shape[0]-1]))
            
        f.write("};\n")
        f.write("\n")

    f.write("\n\n")



for key in model_dict:
    model_dict[key] = model_dict[key].tolist()


json.dump(model_dict, open("./logs/model.json", "w"))
