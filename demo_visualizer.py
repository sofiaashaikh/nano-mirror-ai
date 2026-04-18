import numpy as np
import onnxruntime as ort

def print_grid(grid, title):
    print(f"\n--- {title} ---")
    for row in range(10):
        line = ""
        for col in range(10):
            val = int(grid[row, col])
            line += "██ " if val > 0 else ".  "
        print(line)

def run_demo():
    print("Loading Nano-Mirror AI (task_400.onnx)...")
    session = ort.InferenceSession("task_400.onnx")
    
    input_tensor = np.zeros((1, 10, 30, 30), dtype=np.float32)
    
    input_tensor[0, 1, 1, 1] = 1.0
    input_tensor[0, 1, 2, 1] = 1.0
    input_tensor[0, 1, 3, 1] = 1.0
    input_tensor[0, 1, 3, 2] = 1.0
    input_tensor[0, 1, 3, 3] = 1.0

    input_2d = np.argmax(input_tensor[0], axis=0)
    print_grid(input_2d, "INPUT GRID (Before AI)")

   
    inputs = {session.get_inputs()[0].name: input_tensor}
    output_tensor = session.run(None, inputs)[0]

    output_2d = np.argmax(output_tensor[0], axis=0)
    
    print(f"\n--- OUTPUT GRID (After AI Reflection) ---")
    for row in range(10):
        line = ""
        for col in range(20, 30):
            val = int(output_2d[row, col])
            line += "██ " if val > 0 else ".  "
        print(line)

if __name__ == "__main__":
    run_demo()