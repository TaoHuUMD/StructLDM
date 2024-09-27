from .test_options import TestOptions

class EvaluateOptions(TestOptions):
    def initialize(self):
        TestOptions.initialize(self)
        
        #self.parser.add_argument('--model_name', type=str, default=None)
        self.parser.add_argument('--l1', action='store_true')
        self.parser.add_argument('--ssim', action='store_true')
        self.parser.add_argument('--fid', action='store_true')
        self.parser.add_argument('--lpip', action='store_true')
        self.parser.add_argument('--psnr', action='store_true')
        
        self.parser.add_argument('--make_video', action='store_true')
        
        self.parser.add_argument('--eval_dp', action='store_true')
        self.parser.add_argument('--dp_dir', type=str, default='')

        self.parser.add_argument('--use_gpu', type=int, default=1)
        self.parser.add_argument('-v', '--version', type=str, default='0.1')

        self.parser.add_argument('--crop_bbox', action='store_true')
        
        self.isTrain = False
        print("evaluations")
        #if opt.subdir != '':
        #    opt.results_dir = os.path.join(opt.results_dir, opt.subdir)
