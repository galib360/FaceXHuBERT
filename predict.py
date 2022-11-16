import librosa
import argparse
import pickle
import torch
import trimesh
import numpy as np
import cv2
import os
import ffmpeg
import gc
import pyrender
from faceXhubert import FaceXHuBERT
from transformers import Wav2Vec2Processor
import time


def test_model(args):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    model = FaceXHuBERT(args)
    model.load_state_dict(torch.load('pretrained_model/{}.pth'.format(args.model_name)))
    model = model.to(torch.device(args.device))
    model.eval()

    template_file = os.path.join(args.dataset, args.template_path)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')

    train_subjects_list = [i for i in args.train_subjects.split(" ")]

    one_hot_labels = np.eye(len(train_subjects_list))
    emo_one_hot_labels = np.eye(2)
    if args.emotion == 1:
        emo_one_hot = torch.FloatTensor(emo_one_hot_labels[1]).to(device=args.device)
        emo_label = "emotional"
    else:
        emo_one_hot = torch.FloatTensor(emo_one_hot_labels[0]).to(device=args.device)
        emo_label = "neutral"

    iter = train_subjects_list.index(args.condition)
    one_hot = one_hot_labels[iter]
    one_hot = np.reshape(one_hot,(-1,one_hot.shape[0]))
    one_hot = torch.FloatTensor(one_hot).to(device=args.device)

    temp = templates[args.subject]
             
    template = temp.reshape((-1))
    template = np.reshape(template,(-1,template.shape[0]))
    template = torch.FloatTensor(template).to(device=args.device)

    wav_path = args.wav_path
    test_name = os.path.basename(wav_path).split(".")[0]
    start_time = time.time()
    speech_array, sampling_rate = librosa.load(os.path.join(wav_path), sr=16000)
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-xlarge-ls960-ft")
    audio_feature = processor(speech_array, return_tensors="pt", padding="longest", sampling_rate=sampling_rate).input_values
    audio_feature = np.reshape(audio_feature,(-1,audio_feature.shape[0]))
    audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)

    prediction = model.predict(audio_feature, template, one_hot, emo_one_hot)
    prediction = prediction.squeeze()
    elapsed = time.time() - start_time
    print("Inference time for ", prediction.shape[0], " frames is: ", elapsed, " seconds.")
    print("Inference time for 1 frame is: ", elapsed/prediction.shape[0], " seconds.")
    print("Inference time for 1 second of audio is: ", ((elapsed * 25) / prediction.shape[0]), " seconds.")
    out_file_name = test_name + "_" + emo_label + "_" + args.subject + "_Condition_" + args.condition
    np.save(os.path.join(args.result_path, out_file_name), prediction.detach().cpu().numpy())


def render(args):
    fps = args.fps
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    render_path = "demo/render/"
    frames_folder = render_path + "frames/"
    video_woA_folder = render_path + "video_wo_audio/"
    video_wA_folder = render_path + "video_with_audio/"
    emo_label = "emotional"
    if args.emotion == 0:
        emo_label = "neutral"

    wav_path = args.wav_path
    test_name = os.path.basename(wav_path).split(".")[0]
    out_file_name = test_name + "_" + emo_label + "_" + args.subject + "_Condition_" + args.condition
    predicted_vertices_path = os.path.join(args.result_path,out_file_name+".npy")
    if args.dataset == "BIWI":
        template_file = os.path.join(args.dataset, args.render_template_path + "/BIWI_topology.obj")

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
    camera_pose = np.array([[1.0, 0, 0.0, 0.00],
                            [0.0, -1.0, 0.0, 0.00],
                            [0.0, 0.0, 1.0, -1.6],
                            [0.0, 0.0, 0.0, 1.0]])

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)

    # r = pyrender.OffscreenRenderer(640, 480)
    r = pyrender.OffscreenRenderer(1920, 1440)

    print("rendering the predicted sequence: ", test_name)

    video_woA_path = video_woA_folder + out_file_name + '.mp4'
    video_wA_path = video_wA_folder + out_file_name + '.mp4'
    # video = cv2.VideoWriter(video_woA_path, fourcc, fps, (640, 480))
    video = cv2.VideoWriter(video_woA_path, fourcc, fps, (1920, 1440))

    ref_mesh = trimesh.load_mesh(template_file, process=False)
    seq = np.load(predicted_vertices_path)
    seq = np.reshape(seq, (-1, 70110 // 3, 3))
    ref_mesh.vertices = seq[0, :, :]
    py_mesh = pyrender.Mesh.from_trimesh(ref_mesh)

    for f in range(seq.shape[0]):
        ref_mesh.vertices = seq[f, :, :]
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
    input_audio = ffmpeg.input(wav_path)

    ffmpeg.concat(input_video, input_audio, v=1, a=1).output(video_wA_path).run()
    del video, seq, ref_mesh
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description='FaceXHuBERT: Text-less Speech-driven E(X)pressive 3D Facial Animation Synthesis using Self-Supervised Speech Representation Learning')
    parser.add_argument("--model_name", type=str, default="FaceXHuBERT")
    parser.add_argument("--dataset", type=str, default="BIWI", help='name of the dataset folder. eg: BIWI')
    parser.add_argument("--fps", type=float, default=25, help='frame rate - 25 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=256, help='GRU Vertex Decoder hidden size')
    parser.add_argument("--vertice_dim", type=int, default=70110, help='number of vertices - 23370*3 for BIWI')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_subjects", type=str, default="F1 F2 F3 F4 F5 F6 F7 F8 M1 M2 M3 M4 M5 M6")
    parser.add_argument("--test_subjects", type=str, default="F1 F2 F3 F4 F5 F6 F7 F8 M1 M2 M3 M4 M5 M6")
    parser.add_argument("--wav_path", type=str, default="demo/wav/test.wav", help='path of the input audio signal in .wav format')
    parser.add_argument("--result_path", type=str, default="demo/result", help='path of the predictions in .npy format')
    parser.add_argument("--condition", type=str, default="M3", help='select a conditioning subject from train_subjects')
    parser.add_argument("--subject", type=str, default="M1", help='select a subject from test_subjects or train_subjects')
    parser.add_argument("--template_path", type=str, default="templates_scaled.pkl", help='path of the personalized templates')
    parser.add_argument("--render_template_path", type=str, default="templates", help='path of the mesh in BIWI topology')
    parser.add_argument("--input_fps", type=int, default=50, help='HuBERT last hidden state produces 50 fps audio representation')
    parser.add_argument("--output_fps", type=int, default=25, help='fps of the visual data, BIWI was captured in 25 fps')
    parser.add_argument("--emotion", type=int, default="1", help='style control for emotion, 1 for expressive animation, 0 for neutral animation')
    args = parser.parse_args()   

    test_model(args)
    render(args)


if __name__=="__main__":
    main()
