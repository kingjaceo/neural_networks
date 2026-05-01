"""Count trainable parameters in CompactNet for both head sizes."""
import sys
sys.path.insert(0, '.')

from models import get_model

for num_classes, label in [(200, 'Tiny ImageNet-200'), (100, 'CIFAR-100')]:
    for act in ['relu', 'gelu', 'rswish']:
        m = get_model(num_classes=num_classes, activation=act)
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        if act == 'relu':  # same architecture for all activations
            print(f'{label} ({num_classes} classes): {trainable:,} trainable params')
        break
