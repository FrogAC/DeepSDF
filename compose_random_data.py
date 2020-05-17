import argparse
import os
import random
import numpy as np
import json
from scipy.spatial.transform import Rotation as scRot

SPLIT

def writePly(xyzn:list, name:str):
    with open(name, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(xyzn)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p in xyzn:
            f.write('{} {} {} {} {} {} 122 122 122\n'.format(p[0], p[1], p[2], p[3], p[4], p[5]))

def writeNpz(xyzn:list, center: list, bbox:list, name:str):
    np.savez(name, xyzn = np.array(xyzn), center = np.array(center), bbox = np.array(bbox))

# standard format:
# ...
# element vertex {}
# ...
# end_header
# x y z nx ny nz r g b
# ...
def readPly(name:str, sample:int) -> list:
    xyzn = []
    num_sample = 0
    with open(name) as f:
        for _, line in enumerate(f) :
            if line.startswith('end_header'):
                    break
            if line.startswith('element vertex'):
                num_sample = int(line.strip().split(' ')[2])
        
        xyzn = np.random.choice(f.readlines()[0:num_sample], min(sample, num_sample))
        xyzn = [np.array([float(x) for x in line.strip().split(' ')[0:6]]) for line in xyzn]
    return xyzn

def readScaleAndBbox(name):
    with open(name, 'rb') as f:
        data = np.load(f)
        scale = data['scale']
        bboxmax = data['bboxmax']
        # assert the bbox is centered
        # assert((bboxmax == -bboxmin).all())
        return  (scale, bboxmax)


def generateData(outputDir:str , id:int , surfaceNormFiles:list,  n:int, numSample:int, boxSize:float):
    ITER_MAX = 1000

    centers = []
    for _ in range(n):
        validSdf = False
        iterLeft = ITER_MAX
        while not validSdf and iterLeft > 0 :
            iterLeft -= 1
            boxLen = (boxSize-2)/2
            newCenter = np.array([
                random.uniform(-boxLen,boxLen),
                0,
                random.uniform(-boxLen,boxLen)
            ])
            validSdf = True
            for center in centers:
                validSdf &= np.linalg.norm(center - newCenter) > 2.01
            if validSdf: 
                centers.append(newCenter)
                
    # gen sdf
    print('#{} : {}/{} objs'.format(id, len(centers),n))
    combinedXyzn = []
    bboxs = []
    selectedSurfNorms = random.sample(surfaceNormFiles, len(centers))
    rot_x90 = scRot.from_euler('x', 90, True)
    centers = rot_x90.apply(centers)
    for center, (surfF, normF) in zip(centers, selectedSurfNorms):
        xyzn = readPly(surfF, numSample)
        (_,bbox) = readScaleAndBbox(normF)
        # rot = scRot.from_euler(
        #     'y',
        #     random.uniform(0.0,360.0),
        #     True
        # )
        # sdf = [x for x in rot.apply(sdf)]

        # Rotate to Z-UP
        xyzn = [np.append(rot_x90.apply(x[0:3]) + center,rot_x90.apply(x[3:6])) for x in xyzn]
        # sdf = rot_x90.apply(sdf)
        combinedXyzn += xyzn
        bboxs.append(rot_x90.apply(bbox))

    plyFile = os.path.join(outputDir, 'scene_{}.ply'.format(id))
    npzFile = os.path.join(outputDir, 'scene_{}.npz'.format(id))
    infoFile = os.path.join(outputDir, 'scene_{}_info.json'.format(id))
    writePly(combinedXyzn, plyFile)
    writeNpz(combinedXyzn, centers, bboxs , npzFile)
    infoData = [{'mesh': os.path.basename(surf)[:-4]} for (surf,_) in selectedSurfNorms]
    with open(infoFile,'w') as f:
        f.write(json.dumps(infoData, indent=4))

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Generate synthetic scene from preproceesed ply dataset",
    )
    arg_parser.add_argument(
        "--data_dir",
        "-d",
        dest="data_dir",
        required=True,
        help="The experiment data directory",
    )
    arg_parser.add_argument(
        "--name",
        "-n",
        dest="source_name",
        default=None,
        help="The name to use for the data source",
    )
    arg_parser.add_argument(
        "--split",
        dest="split_filename",
        required=True,
        help="A split filename defining the shapes to be processed.",
    )
    arg_parser.add_argument(
        "--nobj",
        dest='n',
        required = True,
        help = "Number of objects in each scene"
    )
    arg_parser.add_argument(
        "--nscene",
        dest='s',
        required = True,
        help = 'Number of scenes'
    )
    arg_parser.add_argument(
        "--sample",
        dest='sample',
        required = True,
        help = 'number of sample surf points per object'
    )
    arg_parser.add_argument(
        "--bbox",
        dest='boxSize',
        required = False,
        default= 4.0,
        help = 'size of the bounding box'
    )
    args =  arg_parser.parse_args()
    normParamBase = os.path.join(args.data_dir, 'NormalizationParameters', args.source_name)
    surfaceBase = os.path.join(args.data_dir, 'SurfaceSamples', args.source_name)
    outputBase = os.path.join(args.data_dir, 'ComposedScene', args.source_name)

    surfaceFiles = []
    normParamFiles = []
    with open(args.split_filename, "r") as f:
        split = json.load(f)
    for folder in split[args.source_name]:
        outputP = os.path.join(outputBase, folder)
        os.makedirs(outputP, exist_ok=True)

        surfP = os.path.join(surfaceBase, folder)
        normP = os.path.join(normParamBase, folder)
        surfF = os.listdir(surfP)
        surfaceFiles += [os.path.join(surfP,f) for f in surfF]
        normParamFiles += [os.path.join(normP,f[:-4]+'.npz') for f in surfF]  # assume surf files < norm files
        for i in range(int(args.s)):
            generateData(
                outputP,
                i,
                list(zip(surfaceFiles, normParamFiles)),
                int(args.n),
                int(args.sample),
                float(args.boxSize))