import os

def matched_checkpoint(dir, epoch, step = -1, clip_value = 0):
    
    pts = os.listdir(dir)
    max_pt = -1
    print(dir)
    pth_format = ""
    for pt in pts:
        if os.path.isdir(os.path.join(dir, pt)) and pt.count("x") == 3:
            if pt.startswith(f'{epoch}') and ((step != -1 and pt.find(f"_{step}") !=-1) or step == -1):
                if clip_value != 0 and pt.endswith(f"_{clip_value}"):
                    return pt
                elif clip_value == 0: return pt                 

    if max_pt == -1: 
        print("diff model not trained, %s" % dir)
        raise NotImplementedError()
        return -1
    return "%d_%s" % (max_pt, pth_format)


    sample_dirs, self.opt.df.diffusion_epoch

def get_diff_epoch(opt):
    if opt.df.load_diffused_sample:
        sample_dirs = f'../data/result/nr/trained_model/DF2/diffusion/{opt.dataset.dataset_name}/{opt.df.diffusion_name}/samples'

        diff_pth_dir = find_diffusion_checkpoint(sample_dirs)
        
        if opt.df.diffusion_epoch != -1:
            #ep = diff_pth_dir.split("_")[0]
            #diff_pth_dir = diff_pth_dir.replace(ep, f'{self.opt.df.diffusion_epoch}')
            diff_pth_dir = matched_checkpoint(sample_dirs, opt.df.diffusion_epoch)
        else:
            opt.df.diffusion_epoch = int(diff_pth_dir.split("_")[0])
        return opt.df.diffusion_epoch
    return None


def find_diffusion_checkpoint(dir):
    pts = os.listdir(dir)
    max_pt = -1
    print(dir)
    pth_format = ""
    for pt in pts:
        #print(pt, pt.count("x"))
        if os.path.isdir(os.path.join(dir, pt)) and pt.count("x") == 3:
        #if pt.startswith("samples") and pt.endswith(".npz"):            
            format_ = pt[pt.find('_') + 1:]
            iters = pt.split("_")[0]
            try:
                iters = int(iters)
            except: continue
            if iters > max_pt: 
              max_pt = iters
              pth_format = format_

    if max_pt == -1: 
        print("diff model not trained, %s" % dir)
        raise NotImplementedError()
        return -1
    return "%d_%s" % (max_pt, pth_format)
        
def find_diffusion_checkpoint_file(dir):
    pts = os.listdir(dir)
    max_pt = -1
    print(dir)
    pth_format = ""
    for pt in pts:
        if pt.startswith("samples") and pt.endswith(".npz"):
            pth_format = pt.split("_")[-1]
            iters = pt.split("_")[1]
            iters = int(iters)
            if iters > max_pt: max_pt = iters
    if max_pt == -1: 
        print("diff model not trained, %s" % dir)
        raise NotImplementedError()
        return -1
    return "samples_%d_%s" % (max_pt, pth_format)
        
