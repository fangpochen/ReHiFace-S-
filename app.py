import gradio as gr
import cv2
import os
import numpy as np
import numexpr as ne
from concurrent.futures import ThreadPoolExecutor

from face_feature.hifi_image_api import HifiImage
from HifiFaceAPI_parallel_trt_roi_realtime_sr_api import HifiFaceRealTime
from face_lib.face_swap import HifiFace
from face_restore.gfpgan_onnx_api import GFPGAN
from face_restore.xseg_onnx_api import XSEG
from face_detect.face_align_68 import face_alignment_landmark
from face_detect.face_detect import FaceDetect
from options.hifi_test_options import HifiTestOptions
from color_transfer import color_transfer

opt = HifiTestOptions().parse()
processor = None

def initialize_processor():
    global processor
    if processor is None:
        processor = FaceSwapProcessor(crop_size=opt.input_size)

class FaceSwapProcessor:
    def __init__(self, crop_size=256):
        self.hi = HifiImage(crop_size=crop_size)
        self.xseg = XSEG(model_type='xseg_0611', provider='gpu')
        self.hf = HifiFace(model_name='er8_bs1', provider='gpu')
        self.scrfd_detector = FaceDetect(mode='scrfd_500m', tracking_thres=0.15)
        self.face_alignment = face_alignment_landmark(lm_type=68)
        self.gfp = GFPGAN(model_type='GFPGANv1.4', provider='gpu')
        self.crop_size = crop_size

    def reverse2wholeimage_hifi_trt_roi(self, swaped_img, mat_rev, img_mask, frame, roi_img, roi_box):
        target_image = cv2.warpAffine(swaped_img, mat_rev, roi_img.shape[:2][::-1], borderMode=cv2.BORDER_REPLICATE)[
                       ...,
                       ::-1]
        local_dict = {
            'img_mask': img_mask,
            'target_image': target_image,
            'roi_img': roi_img,
        }
        img = ne.evaluate('img_mask * (target_image * 255)+(1 - img_mask) * roi_img', local_dict=local_dict,
                          global_dict=None)
        img = img.astype(np.uint8)
        frame[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]] = img
        return frame

    def process_frame(self, frame, image_latent, use_gfpgan, sr_weight, use_color_trans, color_trans_mode):
        _, bboxes, kpss = self.scrfd_detector.get_bboxes(frame, max_num=0)
        rois, faces, Ms, masks = self.face_alignment.forward(
            frame, bboxes, kpss, limit=5, min_face_size=30,
            crop_size=(self.crop_size, self.crop_size), apply_roi=True
        )

        if len(faces) == 0:
            return frame
        elif len(faces) == 1:
            face = np.array(faces[0])
            mat = Ms[0]
            roi_box = rois[0]
        else:
            max_index = np.argmax([roi[2] * roi[3] for roi in rois])  # Get the largest face
            face = np.array(faces[max_index])
            mat = Ms[max_index]
            roi_box = rois[max_index]

        roi_img = frame[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        mask_out, swap_face_out = self.hf.forward(face, image_latent[0].reshape(1, -1))
        mask_out = self.xseg.forward(swap_face_out)[None, None]

        mask = cv2.warpAffine(mask_out[0][0].astype(np.float32), mat, roi_img.shape[:2][::-1])
        mask[mask > 0.2] = 1
        mask = mask[:, :, np.newaxis].astype(np.uint8)
        swap_face = swap_face_out[0].transpose((1, 2, 0)).astype(np.float32)
        target_face = (face.copy() / 255).astype(np.float32)

        if use_gfpgan:
            sr_face = self.gfp.forward(swap_face)
            if sr_weight != 1.0:
                sr_face = cv2.addWeighted(sr_face, sr_weight, swap_face, 1.0 - sr_weight, 0)
            if use_color_trans:
                transed_face = color_transfer(color_trans_mode, (sr_face + 1) / 2, target_face)
                swap_face = (transed_face * 2) - 1
            else:
                swap_face = sr_face
        elif use_color_trans:
            transed_face = color_transfer(color_trans_mode, (swap_face + 1) / 2, target_face)
            swap_face = (transed_face * 2) - 1

        swap_face = ((swap_face + 1) / 2)

        frame_out = self.reverse2wholeimage_hifi_trt_roi(
            swap_face, mat, mask,
            frame, roi_img, roi_box
        )

        return frame_out

def process_image_video(image, video_path, use_gfpgan, sr_weight, use_color_trans, color_trans_mode):
    global processor
    initialize_processor()

    src_latent, _ = processor.hi.get_face_feature(image)
    image_latent = [src_latent]

    video = cv2.VideoCapture(video_path)
    video_fps = video.get(cv2.CAP_PROP_FPS)
    video_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    output_dir = 'data/output/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    swap_video_path = output_dir + 'temp.mp4'
    videoWriter = cv2.VideoWriter(swap_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, video_size)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            future = executor.submit(processor.process_frame, frame, image_latent, use_gfpgan, sr_weight,
                                     use_color_trans, color_trans_mode)
            futures.append(future)

        for future in futures:
            processed_frame = future.result()
            if processed_frame is not None:
                videoWriter.write(processed_frame)

    video.release()
    videoWriter.release()

    add_audio_to_video(video_path, swap_video_path)

    return swap_video_path


def add_audio_to_video(original_video_path, swapped_video_path):
    audio_file_path = original_video_path.split('.')[0] + '.wav'
    if not os.path.exists(audio_file_path):
        os.system(f'ffmpeg -y -hide_banner -loglevel error -i "{original_video_path}" -f wav -vn "{audio_file_path}"')

    temp_output_path = swapped_video_path.replace('.mp4', '_with_audio.mp4')
    os.system(
        f'ffmpeg -y -hide_banner -loglevel error -i "{swapped_video_path}" -i "{audio_file_path}" -c:v copy -c:a aac "{temp_output_path}"')

    os.remove(swapped_video_path)
    os.rename(temp_output_path, swapped_video_path)


# Gradio interface setup
iface = gr.Interface(
    fn=process_image_video,
    inputs=[
        gr.Image(type="pil", label="Source Image"),
        gr.Video(label="Input Video"),
        gr.Checkbox(label="Use GFPGAN [Super-Resolution]"),
        gr.Slider(minimum=0.1, maximum=1.0, step=0.1, label="SR Weight [only support GFPGAN enabled]", value=1.0),
        gr.Checkbox(label="Use Color Transfer"),
        gr.Dropdown(choices=["rct", "lct", "mkl", "idt", "sot"],
                    label="Color Transfer Mode [only support Color-Transfer enabled]", value="rct")
    ],
    outputs=gr.Video(label="Output Video"),
    title="Video Generation",
    description="Upload an image and a video, and the system will generate a new video based on the input."
)

if __name__ == "__main__":
    iface.launch()