from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=50, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')        
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        
        self.parser.add_argument('--eva_epoch_freq', type=int, help='evaluation freq')
        self.parser.add_argument('--save_epoch_freq', type=int, default=2, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--save_latest_epoch_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--display_epoch_freq', type=int, default=0, help='frequency of showing training results on screen')

        # for training
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        

        # for discriminators        

        self.isTrain = True
