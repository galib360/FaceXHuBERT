import math
import trimesh
import numpy as np
import cv2
import io
from PIL import Image
import os
import ffmpeg
import gc
import pyrender
import pymeshlab as pmlab

quantfilename = "quantitative_metric.txt"
render_folder = "renders/"
gt_folder = "GroundTruth/"
pred_folder = "../result/"
audio_folder = "../BIWI/wav/"
video_woA_folder = render_folder + "videos_no_audio/"
video_wA_folder = render_folder + "videos_with_audio/"
meshes_folder = render_folder+ "temp/meshes/"
frames_folder = render_folder+ "temp/frames/"

mean_face_vertex_error = 0

gt_seqs = os.listdir(gt_folder)
pred_seqs = os.listdir(pred_folder)

fps = 25
fourcc = cv2.VideoWriter_fourcc(*'MP4V')

cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
camera_pose = np.array([[1.0, 0,   0.0,   0.00],
                        [0.0,  -1.0, 0.0, 0.00],
                        [0.0,  0.0,   1.0, -2.0],
                        [0.0,  0.0, 0.0, 1.0]])


light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)

r = pyrender.OffscreenRenderer(640, 480)

print("Evaluation started")
for gt_seq in gt_seqs:
    if gt_seq.endswith('.npy'):
        video_woA_path = video_woA_folder + gt_seq.split('.')[0] + '.mp4'
        video_wA_path = video_wA_folder + gt_seq.split('.')[0] + '.mp4'
        video = cv2.VideoWriter(video_woA_path, fourcc, fps, (640, 480))
        gt_seq_path = gt_folder + gt_seq
        pred_seq_path = pred_folder + gt_seq.split('.')[0] + "_condition_" + gt_seq.split('_')[0] + ".npy"
        print("Now evaluating sequence: ", gt_seq)
        subject_template_path = "../BIWI/templates/"+ gt_seq.split('_')[0] + ".obj"
        audio = gt_seq.split('.')[0].split('_')[0]+'_'+gt_seq.split('.')[0].split('_')[1]+'.wav'
        audio_path = audio_folder + audio
#         print(gt_seq_path)
#         print(pred_seq_path)
#         print(audio_path)
#         print(subject_template_path)
#         ref_mesh = trimesh.load_mesh(subject_template_path, process=False)
        
        gt_seq = np.load(gt_seq_path)
        pred_seq = np.load(pred_seq_path)
        
        if(gt_seq.shape[0]>pred_seq.shape[0]):
            gt_seq = gt_seq[:pred_seq.shape[0]]
            
        if(pred_seq.shape[0]>gt_seq.shape[0]):
            pred_seq = pred_seq[:gt_seq.shape[0]]
        
        
        gt_seq = np.reshape(gt_seq,(-1,70110//3,3))
        pred_seq = np.reshape(pred_seq,(-1,70110//3,3))
        sequence_mean_face_vertex_error = 0
        
        
#         seq = np.reshape(seq,(-1,70110//3,3))
#         ref_mesh.vertices = seq[0,:,:]
#         py_mesh = pyrender.Mesh.from_trimesh(ref_mesh)
        for f in range(pred_seq.shape[0]):
            ms = pmlab.MeshSet()
            ms.load_new_mesh(subject_template_path)
            template_mesh= ms.current_mesh()
            
            gt_mesh = pmlab.Mesh(gt_seq[f,:,:], template_mesh.face_matrix(), template_mesh.vertex_normal_matrix())
            ms.add_mesh(gt_mesh)
#             print(ms.current_mesh_id())

            pred_mesh = pmlab.Mesh(pred_seq[f,:,:], template_mesh.face_matrix(), template_mesh.vertex_normal_matrix())
            ms.add_mesh(pred_mesh)
#             print(ms.current_mesh_id())
            
            ms.apply_filter('distance_from_reference_mesh', measuremesh=2, refmesh=1, signeddist = False)
            ms.set_current_mesh(2)
            vertex_quality = ms.current_mesh().vertex_quality_array()
            #ms.apply_filter('colorize_by_vertex_quality', minval=ms.current_mesh().vertex_quality_array().min(), maxval=ms.current_mesh().vertex_quality_array().max(),zerosym=True)
            ms.apply_filter('quality_mapper_applier', minqualityval=ms.current_mesh().vertex_quality_array().min(), maxqualityval=ms.current_mesh().vertex_quality_array().max(),tfslist= 1)
            ms.save_current_mesh( meshes_folder + str(f) +".obj", save_vertex_color=True)
            sequence_mean_face_vertex_error = sequence_mean_face_vertex_error + vertex_quality.mean(axis=None)
            ms.set_current_mesh(2)
            ms.delete_current_mesh()
            ms.set_current_mesh(1)
            ms.delete_current_mesh()
            ms.set_current_mesh(0)
            
            render_mesh = trimesh.load_mesh(meshes_folder + str(f) +".obj", process=False)
            py_mesh = pyrender.Mesh.from_trimesh(render_mesh)
            
#             ref_mesh.vertices = seq[f,:,:]
#             py_mesh = pyrender.Mesh.from_trimesh(ref_mesh)
            scene = pyrender.Scene()
            scene.add(py_mesh)
            
            scene.add(cam, pose=camera_pose)
            scene.add(light, pose=camera_pose)
            color, _ = r.render(scene)
            
            output_frame = frames_folder + "frame" + str(f) + ".jpg"
            image_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_frame, image_bgr)
            frame = cv2.imread(output_frame)
            video.write(frame)
        video.release()
        sequence_mean_face_vertex_error = sequence_mean_face_vertex_error/pred_seq.shape[0]
        mean_face_vertex_error = mean_face_vertex_error + sequence_mean_face_vertex_error
        
        input_video = ffmpeg.input(video_woA_path)
        input_audio = ffmpeg.input(audio_path)
        ffmpeg.concat(input_video, input_audio, v=1, a=1).output(video_wA_path).run()
        del video
        gc.collect()
        
        
mean_face_vertex_error = mean_face_vertex_error/len(gt_seqs)

file = open(quantfilename, "w")
 
#convert variable to string
str = repr(mean_face_vertex_error)
file.write("mean_face_vertex_error = " + str + "\n")

file.close()
print("Done!")