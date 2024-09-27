#import cv2
from lxml import etree as ET
import numpy as np
import kengine.thutils as utils
import skimage.io

def write_opencv_data(data, filename):

        #data format is 'nodename': numpy array/data

        root = ET.Element ("opencv_storage")


        for nodename in data:
            data_node = ET.SubElement(root, nodename)

            #for matrices
            if (isinstance(data[nodename], np.ndarray)):
                nparray = data[nodename]
                data_node.set("type_id", "opencv-matrix")
                subnode = ET.SubElement(data_node, "rows"); subnode.text = str(nparray.shape[0])
                subnode = ET.SubElement(data_node, "cols"); subnode.text = "1" if len(nparray.shape) == 1 else str(nparray.shape[1])

                subnode = ET.SubElement(data_node, "dt")
                if (nparray.dtype.name == 'float64'): subnode.text = 'd'
                else: subnode.text = 'f'
                subnode = ET.SubElement(data_node, "data"); subnode.text = ' '.join(map(str, nparray.ravel().tolist()))

            else:
                #for elementary data
                data_node.text = str(data[nodename])


        tree = ET.ElementTree(root)
        textout = ET.tostring(tree, xml_declaration = True, pretty_print=True).replace("'", '"')

        file = open(filename, "w")
        file.write(textout)

        #tree.write(filename, pretty_print=True, xml_declaration = True)
        #tree.write(filename, pretty_print=True)


def findSiftGPU(imagefile, siftfile = "temp_siftfile.sift"):
    command = "findsift_g " + imagefile + " " + siftfile
    print(command)
    engine.executShellCommand(command)
    return skimage.io.load_sift(open(siftfile))

def loadVSFMSIFT(siftfile):
    asciifile = siftfile+".ascii"
    command = "readsift_vsfm " + siftfile + " " + asciifile
    engine.executShellCommand(command)
    return skimage.io.load_sift(asciifile)

if __name__ == "__main__":
    cam_mat = np.random.random((3, 3))
    dist_co = np.random.random((1, 5))
    rms = 0.45

    write_opencv_data({"Camera_Matrix": cam_mat, "Distortion_Coefficients": dist_co, "Avg_Reprojection_Error": rms }, "camera_matrix.xml")
