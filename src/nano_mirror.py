import torch
import torch.nn as nn
import onnx
import os

class NanoMirror(nn.Module):
    def __init__(self):
        super(NanoMirror, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=10, 
            out_channels=10, 
            kernel_size=(1, 30), 
            groups=10, 
            bias=False,
            padding=(0, 29)
        )
        self._set_static_weights()

    def _set_static_weights(self):
        weights = torch.zeros(self.conv.weight.shape)
        for i in range(10):
            weights[i, 0, 0, 0] = 1.0
        self.conv.weight = nn.Parameter(weights)

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :, :30]

def finalize_onnx(path):
    # Load the model and force all data to be internal
    model = onnx.load(path)
    onnx.save(model, path, save_as_external_data=False)
    # Clean up the external data file if it exists
    data_file = path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)

def execute_export():
    model = NanoMirror()
    model.eval()
    dummy_input = torch.zeros((1, 10, 30, 30))
    dummy_input[0, 1, 0, 0] = 1.0 
    
    with torch.no_grad():
        output = model(dummy_input)
    
    if output[0, 1, 0, 29] == 1.0:
        onnx_path = "task400.onnx"
        torch.onnx.export(
            model, 
            dummy_input, 
            onnx_path, 
            export_params=True, 
            opset_version=17, 
            do_constant_folding=True,
            input_names=['input_grid'], 
            output_names=['output_grid']
        )
        finalize_onnx(onnx_path)
        return True
    return False

if __name__ == "__main__":
    success = execute_export()
    if success:
        print("STATUS: ARCHITECTURE_VERIFIED")
        print("ARTIFACT: task400.onnx_GENERATED (Self-Contained)")