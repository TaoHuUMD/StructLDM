from kutils import utils as ku
import uuid
import numpy as np
import os



cacheddir = '/data_sarkar/cloudcompare_cache/'

def samplePointsCloudCompare(meshname, outputpcdfile = None, nopoints = 1000000):
    if outputpcdfile is None:
        outputpcdfile = meshname[:-4] + ".pcd"

    command = "CloudCompare -SILENT -o " + meshname + " -C_EXPORT_FMT PCD -SAMPLE_MESH POINTS " + str(nopoints)

    #command = "CloudCompare -SILENT -o " + meshname + " -M_EXPORT_FMT OBJ -SAMPLE_MESH POINTS " + str(nopoints)
    print command
    ku.executShellCommand(command)

    clouddir, _ = ku.getDirAndFile(meshname)
    print clouddir
    #print "ls -t1 " + clouddir + "|grep " + ku.getFileNameWoExt(meshname) + "| head -n1"
    outputname = ku.executeAndGetOutput("ls -t1 " + clouddir + "|grep " + ku.getFileNameWoExt(meshname) + "| head -n1").strip()
    print outputname
    # creating concrete output
    print "mv " + clouddir + "/" + outputname + " " + outputpcdfile
    ku.executShellCommand("mv " + clouddir + "/" + outputname + " " + outputpcdfile)
    return outputpcdfile


def samplePointsCloudCompareSafe(meshname, outputpcdfile, nopoints=1000000):
    print 'params: ', meshname, outputpcdfile

    import os
    meshext = ku.splitExt(meshname)[1]
    meshid = ku.getFileNameWoExt(meshname)

    #meshid = uuid.uuid4().hex
    safemesh = cacheddir + "/meshname_" + meshid + meshext

    #safeout = cacheddir + "/output_" + meshid + ".pcd"

    command = 'cp ' + meshname + " " + safemesh
    ku.executShellCommand(command)

    outfile = samplePointsCloudCompare(safemesh, outputpcdfile, nopoints)
    os.remove(safemesh)

    return outputpcdfile


def samplePointsCloudCompareSafeOld(meshname, outputpcdfile, nopoints=1000000):

    meshext = ku.splitExt(meshname)[1]
    meshid = uuid.uuid4().hex
    safemesh = cacheddir + "/meshname_" + meshid + meshext

    safeout = cacheddir + "/output_" + meshid + ".pcd"

    command = 'cp ' + meshname + " " + safemesh
    ku.executShellCommand(command)

    outfile = samplePointsCloudCompare(safemesh, safeout, nopoints)

    command = 'cp ' + outfile + " " + outputpcdfile
    ku.executShellCommand(command)

    os.remove(safemesh)
    os.remove(safeout)

    return outputpcdfile

def compareCloudToMesh(cloudname, meshname):

    clouddir, _ = ku.getDirAndFile(cloudname)
    command = 'CloudCompare -SILENT -o ' + cloudname + " -o " + meshname + " -C_EXPORT_FMT ASC -C2M_DIST"
    print command
    ku.executShellCommand(command)
    outputname = ku.executeAndGetOutput("ls -t1 " + clouddir + "| head -n1").strip()

    #creating concrete output
    errorfilename = cloudname[:-4] + "_error.asc"
    ku.executShellCommand("mv " + clouddir + "/" + outputname + " " + errorfilename)

    error = np.loadtxt(errorfilename)
    os.remove(errorfilename)

    return error[:, -1]

def compareCloudToMeshPymesh(cloudname, meshname):
    import pcl
    pointcloud = np.asarray(pcl.load(cloudname))
    mesh = pymesh.load_mesh(meshname)
    distances = pymesh.distance_to_mesh(mesh, pointcloud)
    return distances[0]

def compareCloudToCloud(pcfrom, pcto):
    import pcl
    from scipy import spatial

    pcfrom = np.asarray(pcl.load(pcfrom))
    pcto = np.asarray(pcl.load(pcto))

    tree = spatial.cKDTree(pcto)
    mindist, minid = tree.query(pcfrom)
    return mindist


def comparePointsToMeshPymesh(points, meshname):
    import pymesh
    pointcloud = points
    mesh = pymesh.load_mesh(meshname)
    distances = pymesh.distance_to_mesh(mesh, pointcloud)
    return distances[0]


def convertToUnitCube(meshin, meshout = None):
    if meshout is None:
        meshout = meshin

    import pymesh

    mesh = pymesh.load_mesh(meshin)
    pts = mesh.vertices

    # pts = pts - pts.mean(axis=0)
    pts = pts / np.max(pts.max(axis=0) - pts.min(axis=0))
    bbox = (pts.max(axis=0) + pts.min(axis=0)) / 2
    pts = pts - bbox
    meshnew = pymesh.form_mesh(pts, mesh.faces)
    pymesh.save_mesh(meshout, meshnew)


def translateToCenter(meshin, meshout = None):
    if meshout is None:
        meshout = meshin

    import pymesh

    mesh = pymesh.load_mesh(meshin)
    pts = mesh.vertices

    # pts = pts - pts.mean(axis=0)
    bbox = (pts.max(axis=0) + pts.min(axis=0)) / 2
    pts = pts - bbox
    meshnew = pymesh.form_mesh(pts, mesh.faces)
    pymesh.save_mesh(meshout, meshnew)



def convertToUnitCubePCD(pcdin, pcdout = None):
    import pcl

    if pcdout is None:
        pcdout = pcdin


    pts = np.asarray(pcl.load(pcdin))


    # pts = pts - pts.mean(axis=0)
    pts = pts / np.max(pts.max(axis=0) - pts.min(axis=0))
    bbox = (pts.max(axis=0) + pts.min(axis=0)) / 2
    pts = pts - bbox

    pc = pcl.PointCloud(pts.astype(np.float32))
    pc._to_pcd_file(pcdout, True)


def scalePCD(pcdin, scale, pcdout = None):
    import pcl

    if pcdout is None:
        pcdout = pcdin


    pts = np.asarray(pcl.load(pcdin))


    # pts = pts - pts.mean(axis=0)
    pts = pts / np.max(pts.max(axis=0) - pts.min(axis=0))
    bbox = (pts.max(axis=0) + pts.min(axis=0)) / 2
    pts = pts - bbox

    pts = pts * scale

    pc = pcl.PointCloud(pts.astype(np.float32))
    pc._to_pcd_file(pcdout, True)


def scaleMesh(meshin, scale, meshout = None):
    if meshout is None:
        meshout = meshin

    import pymesh

    mesh = pymesh.load_mesh(meshin)
    pts = mesh.vertices

    # pts = pts - pts.mean(axis=0)
    pts = pts / np.max(pts.max(axis=0) - pts.min(axis=0))
    bbox = (pts.max(axis=0) + pts.min(axis=0)) / 2
    pts = pts - bbox
    pts = pts*scale

    meshnew = pymesh.form_mesh(pts, mesh.faces)
    pymesh.save_mesh(meshout, meshnew)

def convertToPCD(mesh, pcdout):
    import pymesh

    mesh = pymesh.load_mesh(mesh)
    pts = mesh.vertices

    import pcl
    pc = pcl.PointCloud(pts.astype(np.float32))
    pc._to_pcd_file(pcdout, True)


def convertToCGO(pcdfile, cgoout):

    import pcl
    pointcloud = np.asarray(pcl.load(pcdfile))
    f = open(cgoout, 'w')
    print >> f, pointcloud.shape[0]
    for i in range(pointcloud.shape[0]):
        print >> f, pointcloud[i, 0], pointcloud[i, 1], pointcloud[i, 2]



def convertToMesh(pcdin, plyout):
    import pymesh
    import pcl

    pc = pcl.load(pcdin)
    pc._to_ply_file(plyout, True)
