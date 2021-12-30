# U-Net
# u_net_params = dict(num_down_blocks=4, target_channels=4, in_channels=1, attention=False)
u_net_params = dict(num_down_blocks=4, target_channels=4, in_channels=1, attention=True)
# u_net_params = dict(num_down_blocks=4, target_channels=4, in_channels=16)
u_net_optimzer_params = dict(lr=1e-4)

# Normalizer
normalizer_params = dict(num_layers=3, kernel_size=1, in_channels=1, out_channels=1, intermediate_channels=16)
# normalizer_params = dict(num_layers=3, kernel_size=1, in_channels=1, out_channels=16, intermediate_channels=16)
normalizer_optimizer_params = dict(lr=1e-4)


# data augmentation
data_aug_defaults = dict(data_aug_ratio=0.5, sigma=20, alpha=1000, trans_min=-10, trans_max=10, rot_min=-10, rot_max=10,
                         scale_min=0.9, scale_max=1.1, gamma_min=0.5, gamma_max=2.0, brightness_min=0.0,
                         brightness_max=0.1, noise_min=0.0, noise_max=0.1)
