import os
import argparse
import random
import numpy as np
import json
from plyfile import PlyData, PlyListProperty
from scipy.spatial.transform import Rotation as scRot

ITER_MAX = 1000

class compose_scene_util:
    def __init__(self, data_dir, split_dir, split_list, label_list, size_scene, num_objects, num_points, corruption01, use_normalize):
        split_json = [json.load(open(os.path.join(split_dir, split))) for split in split_list]
        self.label_list = label_list
        self.file_label_pairs = [ \
            (os.path.join(data_dir,'SurfaceSamples','ShapeNetV2', outfolder, '{}.ply'.format(data_id)), \
             class_id) \
            for class_id, split in enumerate(split_json) \
            for outfolder in split['ShapeNetV2'] \
            for data_id in split['ShapeNetV2'][outfolder] \
            ]
        # scene detail
        self.size_scene = size_scene
        self.size_center = size_scene - 2.2
        self.num_objects = num_objects
        self.num_points = num_points

        self.use_corruption = corruption01 < 1.0 and corruption01 > 0.0
        self.corruption01 = corruption01

        self.use_normalize = use_normalize

    def get_label_map(self):
        return self.label_list

    def get_next_scene(self):
        '''
            point_cloud : [N,6] x y z nx ny nz
            label: [N,1] label
        '''
        def rnd_center():
            return random.random() * self.size_center - self.size_center / 2.0

        # ------------------------------------------------------------------------
        # pick center
        centers = [np.array([rnd_center(), rnd_center(), 0])]
        for _ in range(self.num_objects - 1):
            iterLeft = ITER_MAX
            while iterLeft > 0 :
                iterLeft -= 1
                newCenter = np.array([rnd_center(),rnd_center(), 0])
                if np.all(np.linalg.norm(np.array(centers) - newCenter, axis=1) > 2):
                    centers.append(newCenter)
                    break

        print('#{}/{} objs'.format(len(centers),self.num_objects))

        # fill-in objs
        ret_pc = np.empty((0,6))
        ret_label = np.empty((0,))
        rot_x90 = scRot.from_euler('x', 90, True)
        normalize_offset = np.array([0.5,0.5,0.5])
        for center in centers:
            objply, label = random.choice(self.file_label_pairs)
            vertex = np.array(np.random.choice(PlyData.read(objply)['vertex'].data,self.num_points).tolist())[:,0:6]
            
            # set rotation
            rot_z = scRot.from_euler('z', np.random.random() * 360, degrees = True)
            rot_vertex = rot_z * rot_x90
            # pos
            floor_offset = np.array([0,0,np.max(vertex[:,1])])
            if self.use_normalize:  # non-optimal branch
                # normalizing to 0-1 cube centered at 0.5,0.5
                vertex[:,0:3] = np.apply_along_axis(rot_vertex.apply, 1, vertex[:,0:3]) + center + floor_offset
                vertex[:,0:3] = vertex[:,0:3] / self.size_scene + normalize_offset  # inplace
            else:
                vertex[:,0:3] = np.apply_along_axis(rot_vertex.apply, 1, vertex[:,0:3]) + center + floor_offset
            # normal
            vertex[:,3:6] = np.apply_along_axis(rot_vertex.apply, 1, vertex[:,3:6])
            
            label = np.full((vertex.shape[0],),label)
            ret_pc = np.append(ret_pc, vertex, axis = 0)
            ret_label =  np.append(ret_label, label)

        if self.use_corruption :
            pc_c, label_c = __corrupt_scene(ret_pc, ret_label, self.corruption01)
            return ret_pc, ret_label, pc_c, label_c
        else:
            return ret_pc, ret_label, None, None

    def __corrupt_scene(pc, label, corruption01):
        # remove all points that is close to random centers

        return pc, label


def writePly(f_out:str, vertex: list):
    with open(f_out, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(vertex)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("end_header\n")
        for i in range(len(vertex)):
            v = vertex[i]
            # col =[50 * (x + 1) for x in [ label[i] % 3, label[i] % 4, label[i] % 5 ]]
            f.write('{} {} {} {} {} {}\n'
                .format(v[0], v[1], v[2],
                    v[3], v[4], v[5]))


if __name__ == "__main__":
    
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Util to compose scenes from surface data",
    )

    # arg_parser.add_argument(
    #     dest="data_dir",
    #     required=True,
    #     help="Deep SDF data dir")
    # arg_parser.add_argument(
    #     dest="split_dir",
    #     required=True,
    #     help="Deep SDF split json dir")
    arg_parser.add_argument(
        'num_scenes',
        type=int,
        help="number of scenes to generate"
    )
    arg_parser.add_argument(
        'size_scene',
        type=float,
        help="size of generating plane (before normalizing, obj as unit size)"
    )
    arg_parser.add_argument(
        'num_objects',
        type=int,
        help="number of objects per scene"
    )
    arg_parser.add_argument(
        'num_points',
        type=int,
        help="number of points per"
    )
    arg_parser.add_argument(
        'corruption01',
        type=float,
        help="(0.0,1.0) -- percent of data missing; Other -- no corruption, equal to 0.0"
    )
    arg_parser.add_argument(
        'out_dir',
        help="dir for output ply files"
    )

    args = arg_parser.parse_args()

    # defauly composer
    composer = compose_scene_util(
        data_dir = 'data/',
        split_dir = 'examples/splits',
        split_list = ['sv2_chairs_train.json',
                      #'sv2_lamps_train.json',
                      #'sv2_planes_train.json',
                      #'sv2_sofas_train.json',
                      #'sv2_tables_train.json'
                      ],
        label_list = ['chair',
                      #'lamp',
                      #'plane',
                      #'sofa',
                      #'table'
                      ],    
        size_scene = args.size_scene,
        num_objects = args.num_objects,
        num_points = args.num_points,
        corruption01 = args.corruption01,
        use_normalize = True
    )

    # setup output dir
    os.makedirs(args.out_dir, mode=0o777, exist_ok=True)

    for i in range(args.num_scenes):
        pc_gd,_, pc_corrupted, _ = composer.get_next_scene()
        writePly(os.path.join(args.out_dir,'compose_scene_{}.ply'.format(i)),pc_gd)
        if pc_corrupted is not None:
            writePly(os.path.join(args.out_dir,'compose_scene_corrupted_{}.ply'.format(i)),pc_corrupted)
        
