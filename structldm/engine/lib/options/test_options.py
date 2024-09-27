from .base_options import BaseOptions

from uvm_lib.engine.thutil.files import get_test_epoch

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='../data/result/result', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=500000, help='how many test images to run')       
        self.parser.add_argument('--cluster_path', type=str, default='features_clustered_010.npy', help='the path for clustered results of encoded features')
        self.parser.add_argument('--use_encoded_image', action='store_true', help='if specified, encode the real image to get the feature map')
        self.parser.add_argument("--export_onnx", type=str, help="export ONNX model to a given file")
        self.parser.add_argument("--engine", type=str, help="run serialized TRT engine")
        self.parser.add_argument("--onnx", type=str, help="run ONNX model via TRT")

        self.parser.add_argument("--pid1", type=str, help="pairid1")
        self.parser.add_argument("--pid2", type=str, help="pairid1")

        self.parser.add_argument("--test_epoch", type=str, help="which epoch to test", default='')
        self.parser.add_argument("--save_name", type=str, help="default project name", default='')

        #self.parser.add_argument('--fps', type=int, default=25, help='fps to make video')

        self.isTrain = False
        
        #if opt.subdir != '':
        #    opt.results_dir = os.path.join(opt.results_dir, opt.subdir)
