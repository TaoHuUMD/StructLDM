from .options import BaseOptions

from .project_option import import_project_opt


class ProjectOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        import_project_opt(self.parser)

    def parse_(self):
        device = "cuda"
        opt = self.parse()
        opt.subdir = opt.dataset.dataset_name

        if opt.data_step != -1:
            opt.test_step_size = opt.data_step


        opt.model.freeze_renderer = True
        opt.training.camera = opt.camera
        opt.training.renderer_output_size = opt.model.renderer_spatial_output_dim
        opt.training.style_dim = opt.model.style_dim
        opt.training.with_sdf = not opt.rendering.no_sdf
        if opt.training.with_sdf and opt.training.min_surf_lambda > 0:
            opt.rendering.return_sdf = True
        opt.rendering.no_features_output = True
        opt.training.sphere_init = False
        opt.isTrain = False

        return opt



        return opt
