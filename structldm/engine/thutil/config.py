import pathlib
cur_path = str(pathlib.Path(__file__).parent.absolute())

cfg_smpl_paths = {
    "neutral": cur_path + '/../../asset/smpl_data/SMPL_NEUTRAL.pkl',
    "male":    cur_path + '/../../asset/smpl_data/SMPL_MALE.pkl',
    "female": cur_path + '/../../asset/smpl_data/SMPL_FEMALE.pkl'
}