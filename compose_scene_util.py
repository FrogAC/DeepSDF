import os
import random
import numpy as np
import json
from plyfile import PlyData, PlyListProperty
from scipy.spatial.transform import Rotation as scRot

ITER_MAX = 1000

LABEL_LIST = [#'chair',
               'lamp',
              #'plane',
               'sofa',
               'table']
SPLIT_FILE = [#'sv2_chairs_train.json',
              'sv2_lamps_train.json',
             #'sv2_planes_train.json',
              'sv2_sofas_train.json',
              'sv2_tables_train.json']

class compose_scene_util:
    def __init__(self, splits_dir, data_dir):
        split_json = [json.load(open(os.path.join(splits_dir, split))) for split in SPLIT_FILE]
        self.file_label_pairs = [ \
            (os.path.join(data_dir,'SurfaceSamples','ShapeNetV2', outfolder, '{}.ply'.format(data_id)), \
             class_id) \
            for class_id, split in enumerate(split_json) \
            for outfolder in split['ShapeNetV2'] \
            for data_id in split['ShapeNetV2'][outfolder] \
            ]

    def get_label_map(self):
        return LABEL_LIST

    def get_scene(self, size_x, size_y, num_objects, num_points):
        '''
            point_cloud : [N,6] x y z nx ny nz
            label: [N,1] label
        '''
        # pick center
        size_x = (size_x - 2) / 2
        size_y = (size_y - 2) / 2
        centers = [np.array([ random.uniform(-size_x,size_x), random.uniform(-size_y,size_y), 0])]
        for _ in range(num_objects - 1):
            iterLeft = ITER_MAX
            while iterLeft > 0 :
                iterLeft -= 1
                newCenter = np.array([ random.uniform(-size_x,size_x), random.uniform(-size_y,size_y), 0])
                if np.all(np.linalg.norm(np.array(centers) - newCenter, axis=1) > 2):
                    centers.append(newCenter)
                    break

        print('#{}/{} objs'.format(len(centers),num_objects))

        # fill-in objs
        ret_pc = np.empty((0,6))
        ret_label = np.empty((0,))
        rot_x90 = scRot.from_euler('x', 90, True)
        for center in centers:
            has_found_model = False
            while not has_found_model:
                try:
                    objply, label = random.choice(self.file_label_pairs)
                    vertex = np.array(np.random.choice(PlyData.read(objply)['vertex'].data,num_points).tolist())[:,0:6]
                    
                    has_found_model = True
                except:
                    has_found_model = False

            assert vertex.shape[1] == 6
            
            # pc = [np.append(rot_x90.apply(x[0:3]) + center , rot_x90.apply(x[3:6])) for x in pc]
            # pos
            # floor_offset = abs(normparams['bboxmin'][1])
            # center[2] += floor_offset
            floor_offset = np.max(vertex[:,1])
            vertex[:,0:3] = np.apply_along_axis(rot_x90.apply, 1, vertex[:,0:3]) + center + floor_offset
            # normal
            vertex[:,3:6] = np.apply_along_axis(rot_x90.apply, 1, vertex[:,3:6])
            label = np.full((vertex.shape[0],),label)
            ret_pc = np.append(ret_pc, vertex, axis = 0) 
            ret_label =  np.append(ret_label, label)
        
        return np.array(ret_pc), np.array(ret_label)


def writePly(f_out:str, vertex: list, label: list):
    with open(f_out, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(label)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("end_header\n")
        for i in range(len(label)):
            v = vertex[i]
            col =[50 * (x + 1) for x in [ label[i] % 3, label[i] % 4, label[i] % 5 ]]
            f.write('{} {} {} {} {} {} {} {} {}\n'
                .format(v[0], v[1], v[2],
                    v[3], v[4], v[5],
                    col[0], col[1], col[2]))


if __name__ == "__main__":
    import time

    start_timer = time.perf_counter()
    
    # example / benchmark
    composer = compose_scene_util(
        splits_dir = 'examples/splits',
        data_dir = 'data/')
    test_size = 5
    for i in range(test_size):
        pc,label = composer.get_scene(6,6,8,4000)
        writePly('tmp/tmp_compose_scene{}.ply'.format(i),pc, label)
    print('Time Per Scene', (time.perf_counter() - start_timer)/test_size)