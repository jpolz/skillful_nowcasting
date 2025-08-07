from dgmr import DGMR
import torch.nn.functional as F
import torch

model = DGMR(
        forecast_steps=4,
        input_channels=1,
        output_shape=128,
        latent_channels=384,
        context_channels=192,
        num_samples=3,
        grid_lambda=1.0,  # Use lower value
        gen_lr=5e-5,
        disc_lr=2e-4,
    )
x = torch.rand((2, 4, 1, 128, 128))
out = model(x)
print(out.shape)  # Should print torch.Size([2, 4, 1, 128, 128])
y = torch.rand((2, 4, 1, 128, 128))
loss = F.mse_loss(y, out)
print(loss.item())  # Should print a scalar loss value
loss.backward()
