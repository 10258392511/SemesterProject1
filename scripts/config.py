# U-Net
u_net_params = dict(num_down_blocks=4, target_channels=4, in_channels=1)
u_net_optimzer_params = dict(lr=1e-3)

# Normalizer
normalizer_params = dict(num_layers=3, kernel_size=1, in_channels=1, intermediate_channels=16)
normalizer_optimizer_params = dict(lr=1e-3)
