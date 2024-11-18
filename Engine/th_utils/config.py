import pathlib
import os
cur_path = str(pathlib.Path(__file__).parent.absolute())

cfg_smpl_paths = {
    "neutral": './data/asset/smpl_data/SMPL_NEUTRAL.pkl',
    "male":    './data/asset/smpl_data/SMPL_MALE.pkl',
    "female": './data/asset/smpl_data/SMPL_FEMALE.pkl'
}