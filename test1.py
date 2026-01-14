import torch
import time
import numpy as np
from thop import profile as thop_profile
from models.gated_fusion_model import GatedFusionModel
from models.baseline_model import BaselineModel

def quick_profile(model, input_size=(3, 512, 512), device='cuda'):
    """快速获取模型关键指标"""
    model = model.to(device).eval()
    
    # 参数量
    params = sum(p.numel() for p in model.parameters()) / 1e6
    
    # FLOPs (需要thop)
    try:
        dummy = torch.randn(1, *input_size).to(device)
        
        class Wrapper(torch.nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x): 
                out = self.m(x)
                return out[0] if isinstance(out, tuple) else out
        
        flops, _ = thop_profile(Wrapper(model), inputs=(dummy,), verbose=False)
        gflops = flops / 1e9
    except:
        gflops = None
    
    # FPS
    dummy = torch.randn(1, *input_size).to(device)
    with torch.no_grad():
        for _ in range(10): model(dummy)  # warmup
        if device == 'cuda': torch.cuda.synchronize()
        
        times = []
        for _ in range(50):
            if device == 'cuda': torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(dummy)
            if device == 'cuda': torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    
    fps = 1.0 / np.mean(times)
    latency = np.mean(times) * 1000
    
    print(f"Parameters: {params:.2f}M")
    print(f"GFLOPs: {gflops:.2f}" if gflops else "GFLOPs: N/A (install thop)")
    print(f"FPS: {fps:.1f}")
    print(f"Latency: {latency:.1f}ms")
    
    return {'params_m': params, 'gflops': gflops, 'fps': fps, 'latency_ms': latency}


# 使用示例
if __name__ == "__main__":
    model = BaselineModel(num_classes=5, pretrained=True)
    quick_profile(model, input_size=(3, 512, 512))