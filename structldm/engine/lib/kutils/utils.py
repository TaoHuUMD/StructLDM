from __future__ import print_function
import os
import os.path
import glob
import subprocess
#from PIL import Image
import numpy as np
#import matplotlib.pyplot as plt
import fnmatch

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def getSplitDir(dirname):
    return  os.path.split(os.path.normpath(dirname))

def getDirAndFile(filename):
    return os.path.split(filename)

def splitExt(filename):
    return os.path.splitext(filename)

def getFileNameWoExt(filename):
    return getDirAndFile(os.path.splitext(filename)[0])[1]

def executShellCommand(command):
    #print command
    return subprocess.call(command, shell = True)

def executShellCommandArray(commandname, args):
    #print command
    args = map(str, args)
    command = commandname + ' ' + ' '.join(args)
    #print command
    return subprocess.call(command, shell = True)

def replaceExtension(filename, newext):
    return os.path.splitext(filename)[0] + newext

def executeAndGetOutput(command):
    #print command
    return subprocess.check_output(command, shell = True)

def getImmediateSubDirectoryNames(a_dir):
    if not os.path.exists(a_dir): return []
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

def getImmediateSubDirectories(a_dir):
    return [(a_dir + "/" + name) for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

def getFiles(pattern):
    return glob.glob(pattern)

def findFilesRec(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

def getFirstFile(pattern):
    l = glob.glob(pattern)
    if(len(l) >=1): return l[0]
    return ""

def getFileNos(pattern):
    return len(glob.glob(pattern))

def createDirectory(directory):
    
    if not os.path.exists(directory):
        #os.makedirs(directory)
        os.makedirs(directory)

def removeDir(directory):
    if not os.path.exists(directory): return
    import shutil
    shutil.rmtree(directory)

def copy(src, dst):
    import shutil
    shutil.copy(src, dst)

def changeDir(dirname):
    os.chdir(os.path.expanduser(dirname))

def isDir(pathname):
    return os.path.isdir(pathname)

CPP_BINARY_DIR = '/home/sarkar/.CLion2016.1/system/cmake/generated/snapshot-753b6f02/753b6f02/Debug/'
def execppcommand(command, args):
    if (command[0] != '/'):
        command = CPP_BINARY_DIR + "/" + command + " " + args
    else:
        command = command + " " + args
    print (command)
    executShellCommand(command)


def kLoadImage(filename):
    im = Image.open(filename)
    image = np.asarray(im).astype(np.float32).copy()
    image /= 256
    return image


def kShowImage(pimage):
    import matplotlib.pyplot as plt
    plt.ion()
    plt.imshow(pimage)
    plt.show()


def vis_square(data, wait = False):
    import matplotlib.pyplot as plt

    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    normscale = data.max() - data.min()
    if normscale == 0: normscale = 1
    data = (data - data.min()) / normscale

    #data = 1 - data

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=0)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    #plt.imshow(data)
    plt.figure()
    plt.imshow(data, cmap='gray', interpolation="nearest")
    plt.imshow(data)
    plt.axis('off')

    if not wait:
        plt.show()

def plottogether(array1, array2):
    import matplotlib.pyplot as plt

    maxint = array1.size / array2.size

    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(np.arange(array1.size), array1)
    ax2.plot(maxint * np.arange(array2.size), array2, 'r')

    ax1.set_xlabel('iteration')
    ax1.set_ylabel('train loss')
    ax2.set_ylabel('test loss')
    ax2.set_title('Final Test Loss: {:.2f}'.format(array2[-1]))
    # pl.plot(pl.randn(100))
    # display.display(plt.gcf())
    # display.clear_output(wait=True)



def distanceP2P(cloudfrom, cloudto):
    #cloud: nx3 mats
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=1).fit(cloudto)
    distances, indices = nbrs.kneighbors(cloudfrom)
    return distances

def convertTo3Channel(image, masks = None, scaleint = False):
    if len(image.shape) == 3:
        return image

    # import cv2
    # newimage = cv2.cvtColor((image*255).astype('uint8'), cv2.COLOR_BAYER_GR2BGR)
    vmin = image.min()
    vmax = image.max()
    normimage = (image - vmin)/(vmax - vmin)
    newimage = np.zeros(image.shape + (3,))
    for i in range(3):
        newimage[:,:,i] = normimage

    if masks is not None:
        masks = np.squeeze(masks)
        #newimage[masks] = [0.0, 1, 0]
        newimage[masks] = [0, 1, 1]

    if scaleint:
        newimage = (255*newimage).astype(dtype=np.uint8)

    return newimage


if __name__ == "__main__":
    # print getDirAndFile("/home/sarkar/work/py/mylibs/kutils/engine.py")
    # print getImmediateSubDirectories("/home/sarkar/models/MODELS_ALL")
    # print getImmediateSubDirectoryNames("/home/sarkar/models/MODELS_ALL")
    # print getFiles("/home/sarkar/models/MODELS_ALL/Lion/*.obj")
    # print getFirstFile("/home/sarkar/models/MODELS_ALL/Lion/*.obj")
    # print executShellCommand("cp lkjlkjlkj lkjlkjl")
    a = [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]
    b = [[1.1, 0.999], [2.1, 0.999]]
    print (distanceP2P(b, a))
