import torch
import torch.nn as nn
import onnx
import os

class ARCLogicEngine(nn.Module):
    def __init__(self, task_type="mirror_h"):
        super(ARCLogicEngine, self).__init__()
        self.task_type = task_type
        self.is_color_task = "color" in task_type
        groups = 1 if self.is_color_task else 10
        
        # 30x30 kernel for spatial, 1x1 for color mapping
        k_size = (1, 1) if self.is_color_task else (30, 30)
        padding = 0 if self.is_color_task else 29
        
        self.conv = nn.Conv2d(10, 10, k_size, groups=groups, bias=False, padding=padding)
        self._initialize_logic()

    def _initialize_logic(self):
        weights = torch.zeros(self.conv.weight.shape)
        with torch.no_grad():
            if self.task_type == "identity":
                for c in range(10): weights[c, 0, 29, 29] = 1.0
            elif self.task_type == "mirror_h":
                for c in range(10): weights[c, 0, 29, 0] = 1.0
            elif self.task_type == "color_2_to_1":
                for c in range(10):
                    if c == 1: weights[1, 2, 0, 0] = 1.0
                    elif c != 2: weights[c, c, 0, 0] = 1.0
            elif self.task_type == "rotate_90":
                for c in range(10): weights[c, 0, 0, 0] = 1.0
        self.conv.weight = nn.Parameter(weights)

    def forward(self, x):
        x = self.conv(x)
        if self.is_color_task: return x
        if self.task_type == "rotate_90":
            return x[:, :, :30, :30].transpose(2, 3).flip(3)
        return x[:, :, :30, :30]

def batch_export(task_list):
    for task_id, logic in task_list.items():
        model = ARCLogicEngine(task_type=logic)
        model.eval()
        dummy_input = torch.zeros((1, 10, 30, 30))
        filename = f"task_{task_id}.onnx"
        
        torch.onnx.export(model, dummy_input, filename, opset_version=17, do_constant_folding=True)
        
        m = onnx.load(filename)
        onnx.save(m, filename, save_as_external_data=False)
        if os.path.exists(filename + ".data"):
            os.remove(filename + ".data")
        print(f"EXPORTED: {filename} | LOGIC: {logic}")

if __name__ == "__main__":
    tasks_to_solve = {
        "400": "mirror_h",
        "401": "identity",
        "402": "rotate_90",
        "403": "color_2_to_1"
    }
    batch_export(tasks_to_solve)