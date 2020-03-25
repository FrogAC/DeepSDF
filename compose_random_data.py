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

def generateData(dstDir, id:int , srcFiles:list, n:int, sample:int):
    ITER_MAX = 1000

    centers = []

    for _ in range(n):
        validSdf = False
        iterLeft = ITER_MAX
        while not validSdf and iterLeft > 0 :
            iterLeft -= 1
            newCenter = np.array([
                random.uniform(-0.75,0.75),
                0,
                random.uniform(-0.75,0.75)
            ])
            validSdf = True
            for center in centers:
                validSdf &= np.linalg.norm(center - newCenter) > 1.0
            if validSdf: 
                centers.append(newCenter)
                
    # gen sdf
    print(len(centers))
    combinedXyz = []
    meshes = np.random.choice(srcFiles, len(centers))
    for center, mesh in zip(centers, meshes):
        xyz = readPly(mesh, sample)
        # rot = scRot.from_euler(
        #     'y',
        #     random.uniform(0.0,360.0),
        #     True
        # )
        # sdf = [x for x in rot.apply(sdf)]
        xyz = [x +center for x in xyz]
        # sdf = rot.apply(sdf)
        combinedXyz += xyz

    plyFile = os.path.join(dstDir, '{}_sdf.ply'.format(i))
    npzFile = os.path.join(dstDir, '{}_sdf.npz'.format(i))
    infoFile = os.path.join(dstDir, '{}_info.json'.format(i))
    writePly(combinedXyz, plyFile)
    writeNpz(combinedXyz, npzFile)
    infoData = [{'mesh': mesh, 'center': center.tolist()} for center, mesh in zip(centers, meshes)]
    print(json.dumps(infoData, indent=4))
    with open(infoFile,'w') as f:
        f.write(json.dumps(infoData))

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
        help="The directory which will hold generated data.",
    )
    arg_parser.add_argument(
        "--source",
        "-s",
        dest="source_dir",
        required=True,
        help="The directory which holds preprocessed data and append.",
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
    args =  arg_parser.parse_args()

    sdfFiles = [os.path.join(args.source_dir,f) for f in os.listdir(args.source_dir)]

    os.makedirs(args.data_dir, exist_ok=True)

    for i in range(int(args.s)):
        generateData(
            args.data_dir,
            i,
            sdfFiles,
            int(args.n),
            int(args.sample))