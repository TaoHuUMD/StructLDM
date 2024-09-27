import torch

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(
                    lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim

xyz_res = 10
view_res = 4
pose_res = 1

xyz_embedder, xyz_dim = get_embedder(xyz_res)
view_embedder, view_dim = get_embedder(view_res)
pose_embedder, pose_dim = get_embedder(pose_res, input_dims=72)
shape_embedder, shape_dim = get_embedder(2, input_dims=10)

uvhn_embedder, uvhn_dim = get_embedder(6, input_dims=6)
uvh_embedder, uvh_dim = get_embedder(6, input_dims=3)
uv_embedder, uv_dim = get_embedder(6, input_dims=2)

#pose_embedder_2, pose_dim_2 = get_embedder(2, input_dims=72)
