import trimesh
import numpy as np
import cv2
import os
import ffmpeg
import gc
import pyrender

train_folder = "render_folder/"
results_folder = "result/"
audio_folder = "BIWI/wav/"
video_woA_folder = "renders/"+ train_folder+ "videos_no_audio/"
video_wA_folder = "renders/"+ train_folder+ "videos_with_audio/"
frames_folder = "renders/"+ train_folder+ "temp/frames/"

seqs = os.listdir(results_folder)

fps = 25
fourcc = cv2.VideoWriter_fourcc(*'MP4V')

cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
camera_pose = np.array([[1.0, 0,   0.0,   0.00],
                        [0.0,  -1.0, 0.0, 0.00],
                        [0.0,  0.0,   1.0, -1.6],
                        [0.0,  0.0, 0.0, 1.0]])


light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)

r = pyrender.OffscreenRenderer(640, 480)

for seq in seqs:
    if seq.endswith('.npy'):
        video_woA_path = video_woA_folder + seq.split('.')[0] + '.mp4'
        video_wA_path = video_wA_folder + seq.split('.')[0] + '.mp4'
        video = cv2.VideoWriter(video_woA_path, fourcc, fps, (640, 480))
        seq_path = results_folder + seq
        subject_template_path = "BIWI/templates/"+ seq.split('_')[0] + ".obj"
        audio = seq.split('_')[0]+'_'+seq.split('_')[1]+'.wav'
        audio_path = audio_folder + audio
        ref_mesh = trimesh.load_mesh(subject_template_path, process=False)
        
        seq = np.load(seq_path)
        seq = np.reshape(seq,(-1,70110//3,3))
        ref_mesh.vertices = seq[0,:,:]
        py_mesh = pyrender.Mesh.from_trimesh(ref_mesh)
        for f in range(seq.shape[0]):
            ref_mesh.vertices = seq[f,:,:]
            py_mesh = pyrender.Mesh.from_trimesh(ref_mesh)
            scene = pyrender.Scene()
            scene.add(py_mesh)
            
            scene.add(cam, pose=camera_pose)
            scene.add(light, pose=camera_pose)
            color, _ = r.render(scene)
            
            output_frame = frames_folder + "frame" + str(f) + ".jpg"
            cv2.imwrite(output_frame, color)
            frame = cv2.imread(output_frame)
            video.write(frame)
        video.release()
        
        input_video = ffmpeg.input(video_woA_path)
        input_audio = ffmpeg.input(audio_path)
        ffmpeg.concat(input_video, input_audio, v=1, a=1).output(video_wA_path).run()
        del video, seq, ref_mesh
        gc.collect()
