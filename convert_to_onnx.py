import torch
import numpy as np

# Suppress YOLO logging if needed
from ultralytics import YOLO

# Import custom models from the user's files safely
# (Assuming execution code is guarded by if __name__ == '__main__')
from finalfacerecognition import ImprovedDeepSortWideResNet
from mediapipe_inference import EfficientGCN_B0

def convert_yolo():
    print("\n--- Converting YOLOv8 ---")
    model = YOLO("yolo26n.pt")
    # Ultralytics built-in export to ONNX with dynamic shapes
    model.export(format="onnx", dynamic=True)
    print("✓ yolo26n.onnx saved.")

def convert_reid():
    print("\n--- Converting BoT-SORT ReID (ImprovedDeepSortWideResNet) ---")
    model = ImprovedDeepSortWideResNet(num_features=128, num_classes=625, dropout=0.3, use_se=True, use_gem=True)
    
    # Load PyTorch weights
    checkpoint = torch.load('best_model.pth', map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    # The input shape is [Batch, 3, 128, 64]
    dummy_input = torch.randn(1, 3, 128, 64)
    
    # Wrap model to enforce return_feature=True during export
    class FeatureWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m(x, return_feature=True)
    
    wrapper = FeatureWrapper(model)
    
    torch.onnx.export(
        wrapper,
        dummy_input,
        "best_model.onnx",
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['features'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'features': {0: 'batch_size'}
        }
    )
    print("✓ best_model.onnx saved.")

def convert_action():
    print("\n--- Converting EfficientGCN-B0 ---")
    model = EfficientGCN_B0(num_classes=45)
    
    # Load PyTorch weights
    ckpt = torch.load('best_efficientgcn_b0_resumed_again.pth', map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt
        
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    # Create dummy inputs [Batch, Channels, Frames, Joints, Persons]
    dummy_joint = torch.randn(1, 6, 90, 25, 2)
    dummy_velocity = torch.randn(1, 6, 90, 25, 2)
    dummy_bone = torch.randn(1, 3, 90, 25, 2)

    torch.onnx.export(
        model,
        (dummy_joint, dummy_velocity, dummy_bone),
        "best_efficientgcn_b0_(2).onnx",
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['joint', 'velocity', 'bone'],
        output_names=['output'],
        dynamic_axes={
            'joint': {0: 'batch_size', 4: 'num_persons'},
            'velocity': {0: 'batch_size', 4: 'num_persons'},
            'bone': {0: 'batch_size', 4: 'num_persons'},
            'output': {0: 'batch_size'}
        }
    )
    print("✓ best_efficientgcn_b0(2).onnx saved.")

if __name__ == "__main__":
    import os
    print(f"Current Working Directory: {os.getcwd()}")
    convert_yolo()
    convert_reid()
    convert_action()
    print("\n--- All Conversions Complete! ---")
