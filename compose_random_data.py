import argparse
import os
import random
import numpy as np
import json
from scipy.spatial.transform import Rotation as scRot



def writePly(xyz:list, name:str):
    with open(name, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(xyz)))
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
        for p in xyz:
            f.write('{} {} {} 0 0 0 122 122 122\n'.format(p[0], p[1], p[2]))

def writeNpz(xyz:list, name:str):
    sdf = np.zeros((len(xyz),1))
    xyz = np.append(np.array(xyz),sdf, axis=1)
    np.savez(name, neg = xyz)

def readPly(name:str, sample:int) -> list:
    xyz = []
    num_sample = 0

    with open(name) as f:
        for _, line in enumerate(f) :
            if line.startswith('end_header'):
                    break
            if line.startswith('element vertex'):
                num_sample = int(line.strip().split(' ')[2])
        
        xyz = np.random.choice(f.readlines()[0:num_sample], min(sample, num_sample))
        xyz = [np.array([float(x) for x in line.strip().split(' ')[0:3]]) for line in xyz]
    return xyz

def readScale(name):
    with open(name, 'rb') as f:
        return np.load(f)['scale']


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
                validSdf &= np.linalg.norm(center - newCenter) > 2
            if validSdf: 
                centers.append(newCenter)
                
    # gen sdf
    print('#{} : {}/{} objs'.format(id, len(centers),n))
    combinedXyz = []
    selectedSurfNorms = random.sample(surfaceNormFiles, len(centers))
    for center, (surfF, normF) in zip(centers, selectedSurfNorms):
        xyz = readPly(surfF, numSample)
        scale = readScale(normF)
        # rot = scRot.from_euler(
        #     'y',
        #     random.uniform(0.0,360.0),
        #     True
        # )
        # sdf = [x for x in rot.apply(sdf)]
        xyz = [x * scale + center for x in xyz]
        # sdf = rot.apply(sdf)
        combinedXyz += xyz

    plyFile = os.path.join(outputDir, 'scene_{}.ply'.format(id))
    npzFile = os.path.join(outputDir, 'scene_{}.npz'.format(id))
    infoFile = os.path.join(outputDir, 'scene_{}_info.json'.format(id))
    writePly(combinedXyz, plyFile)
    writeNpz(combinedXyz, npzFile)
    infoData = [{'mesh': os.path.basename(surf)[:-4], 'center': center.tolist()} for center, (surf,_) in zip(centers, selectedSurfNorms)]
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