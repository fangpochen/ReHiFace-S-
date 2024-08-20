import cv2
import numpy as np
from PIL import Image
from .LandmarksProcessor import get_transform_mat_all
import onnxruntime as ort
def drawLandmark_multiple(img, bbox, landmark):
    '''
    Input:
    - img: gray or RGB
    - bbox: type of BBox
    - landmark: reproject landmark of (5L, 2L)
    Output:
    - img marked with landmark and bbox
    '''
    cv2.rectangle(img, (bbox['left'], bbox['top']), (bbox['right'], bbox['bottom']), (0,0,255), 2)
    for x, y in landmark:
        cv2.circle(img, (int(x), int(y)), 2, (0,255,0), -1)
    return img

def drawLandmark_multiple_list(img, bbox, landmark):
    '''
    Input:
    - img: gray or RGB
    - bbox: type of BBox
    - landmark: reproject landmark of (5L, 2L)
    Output:
    - img marked with landmark and bbox
    '''
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 2)
    for x, y in landmark:
        cv2.circle(img, (int(x), int(y)), 2, (0,255,0), -1)
    return img
REFERENCE_FACIAL_POINTS = [
    [30.29459953,  51.69630051],
    [65.53179932,  51.50139999],
    [48.02519989,  71.73660278],
    [33.54930115,  92.3655014],
    [62.72990036,  92.20410156]
]
landmarks_2D_new = np.array([
    [0.000213256, 0.106454],  # 17
    [0.0752622, 0.038915],  # 18
    [0.18113, 0.0187482],  # 19
    [0.29077, 0.0344891],  # 20
    [0.393397, 0.0773906],  # 21
    [0.586856, 0.0773906],  # 22
    [0.689483, 0.0344891],  # 23
    [0.799124, 0.0187482],  # 24
    [0.904991, 0.038915],  # 25
    [0.98004, 0.106454],  # 26
    [0.490127, 0.203352],  # 27
    [0.490127, 0.307009],  # 28
    [0.490127, 0.409805],  # 29
    [0.490127, 0.515625],  # 30
    [0.36688, 0.587326],  # 31
    [0.426036, 0.609345],  # 32
    [0.490127, 0.628106],  # 33
    [0.554217, 0.609345],  # 34
    [0.613373, 0.587326],  # 35
    [0.121737, 0.216423],  # 36
    [0.187122, 0.178758],  # 37
    [0.265825, 0.179852],  # 38
    [0.334606, 0.231733],  # 39
    [0.260918, 0.245099],  # 40
    [0.182743, 0.244077],  # 41
    [0.645647, 0.231733],  # 42
    [0.714428, 0.179852],  # 43
    [0.793132, 0.178758],  # 44
    [0.858516, 0.216423],  # 45
    [0.79751, 0.244077],  # 46
    [0.719335, 0.245099],  # 47
    [0.254149, 0.780233],  # 48
    [0.726104, 0.780233],  # 54
], dtype=np.float32)
mesh_33=[70,63,105,66,107,336,296,334,293,300,168,197,5,4,240,99,2,328,460,33,160,158,133,153,144,362,385,387,263,373,380,57,287]


def convert98to68(list_info):
    points = list_info[0,0:196]
    info_68 = []
    for j in range(17):
        x = points[j * 2 * 2 + 0]
        y = points[j * 2 * 2 + 1]
        info_68.append(x)
        info_68.append(y)
    for j in range(33, 38):
        x = points[j * 2 + 0]
        y = points[j * 2 + 1]
        info_68.append(x)
        info_68.append(y)
    for j in range(42, 47):
        x = points[j * 2 + 0]
        y = points[j * 2 + 1]
        info_68.append(x)
        info_68.append(y)
    for j in range(51, 61):
        x = points[j * 2 + 0]
        y = points[j * 2 + 1]
        info_68.append(x)
        info_68.append(y)
    point_38_x = (float(points[60 * 2 + 0]) + float(points[62 * 2 + 0])) / 2.0
    point_38_y = (float(points[60 * 2 + 1]) + float(points[62 * 2 + 1])) / 2.0
    point_39_x = (float(points[62 * 2 + 0]) + float(points[64 * 2 + 0])) / 2.0
    point_39_y = (float(points[62 * 2 + 1]) + float(points[64 * 2 + 1])) / 2.0
    point_41_x = (float(points[64 * 2 + 0]) + float(points[66 * 2 + 0])) / 2.0
    point_41_y = (float(points[64 * 2 + 1]) + float(points[66 * 2 + 1])) / 2.0
    point_42_x = (float(points[60 * 2 + 0]) + float(points[66 * 2 + 0])) / 2.0
    point_42_y = (float(points[60 * 2 + 1]) + float(points[66 * 2 + 1])) / 2.0
    point_44_x = (float(points[68 * 2 + 0]) + float(points[70 * 2 + 0])) / 2.0
    point_44_y = (float(points[68 * 2 + 1]) + float(points[70 * 2 + 1])) / 2.0
    point_45_x = (float(points[70 * 2 + 0]) + float(points[72 * 2 + 0])) / 2.0
    point_45_y = (float(points[70 * 2 + 1]) + float(points[72 * 2 + 1])) / 2.0
    point_47_x = (float(points[72 * 2 + 0]) + float(points[74 * 2 + 0])) / 2.0
    point_47_y = (float(points[72 * 2 + 1]) + float(points[74 * 2 + 1])) / 2.0
    point_48_x = (float(points[68 * 2 + 0]) + float(points[74 * 2 + 0])) / 2.0
    point_48_y = (float(points[68 * 2 + 1]) + float(points[74 * 2 + 1])) / 2.0
    info_68.append((point_38_x))
    info_68.append((point_38_y))
    info_68.append((point_39_x))
    info_68.append((point_39_y))
    info_68.append(points[64 * 2 + 0])
    info_68.append(points[64 * 2 + 1])
    info_68.append((point_41_x))
    info_68.append((point_41_y))
    info_68.append((point_42_x))
    info_68.append((point_42_y))
    info_68.append(points[68 * 2 + 0])
    info_68.append(points[68 * 2 + 1])
    info_68.append((point_44_x))
    info_68.append((point_44_y))
    info_68.append((point_45_x))
    info_68.append((point_45_y))
    info_68.append(points[72 * 2 + 0])
    info_68.append(points[72 * 2 + 1])
    info_68.append((point_47_x))
    info_68.append((point_47_y))
    info_68.append((point_48_x))
    info_68.append((point_48_y))
    for j in range(76, 96):
        x = points[j * 2 + 0]
        y = points[j * 2 + 1]
        info_68.append(x)
        info_68.append(y)
    for j in range(len(list_info[196:])):
        info_68.append(list_info[196 + j])
    return np.array(info_68)
def crop(image, center, scale, resolution=256.0):
    ul = transform([1, 1], center, scale, resolution).astype(np.int)
    br = transform([resolution, resolution], center, scale, resolution).astype(np.int)

    if image.ndim > 2:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0], image.shape[2]], dtype=np.int32)
        newImg = np.zeros(newDim, dtype=np.uint8)
    else:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
        newImg = np.zeros(newDim, dtype=np.uint8)
    ht = image.shape[0]
    wd = image.shape[1]
    newX = np.array([max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
    newY = np.array([max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
    oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
    oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
    newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1]] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]

    newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)), interpolation=cv2.INTER_LINEAR)
    return newImg

def get_pts_from_predict(a, center, scale):
    a_ch, a_h, a_w = a.shape

    b = a.reshape((a_ch, a_h * a_w))
    c = b.argmax(1).reshape((a_ch, 1)).repeat(2, axis=1).astype(np.float)
    c[:, 0] %= a_w
    c[:, 1] = np.apply_along_axis(lambda x: np.floor(x / a_w), 0, c[:, 1])

    for i in range(a_ch):
        pX, pY = int(c[i, 0]), int(c[i, 1])
        if pX > 0 and pX < 63 and pY > 0 and pY < 63:
            diff = np.array([a[i, pY, pX + 1] - a[i, pY, pX - 1], a[i, pY + 1, pX] - a[i, pY - 1, pX]])
            c[i] += np.sign(diff) * 0.25

    c += 0.5

    return np.array([transform(c[i], center, scale, a_w) for i in range(a_ch)])
def transform(point, center, scale, resolution):
    pt = np.array([point[0], point[1], 1.0])
    h = 200.0 * scale
    m = np.eye(3)
    m[0, 0] = resolution / h
    m[1, 1] = resolution / h
    m[0, 2] = resolution * (-center[0] / h + 0.5)
    m[1, 2] = resolution * (-center[1] / h + 0.5)
    m = np.linalg.inv(m)
    return np.matmul(m, pt)[0:2]

class pfpld():
    def __init__(self,cpu=True):
        onnx_path = "./pretrain_models/pfpld_robust_sim_bs1_8003.onnx"
        try:
            self.ort_session = ort.InferenceSession(onnx_path)
        except Exception as e:
            raise e("load onnx failed")
        # e.g. ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] ordered by priority
        # self.ort_session.get_providers()
        if cpu:
            self.ort_session.set_providers(['CPUExecutionProvider'])
        else:
            self.ort_session.set_providers(['CUDAExecutionProvider'])
        self.input_name = self.ort_session.get_inputs()[0].name

    def forward(self, input):
        size = input.shape
        ort_inputs = {self.input_name: (cv2.resize(input, (112, 112)) / 255).astype(np.float32).transpose(2, 0, 1)[None]}
        pred = self.ort_session.run(None, ort_inputs)
        pred = convert98to68(pred[1])
        return pred.reshape(-1, 68, 2) * size[:2][::-1]

class face_alignment_landmark():
    def __init__(self,lm_type=68,method='affine'):
        self.lm_type=lm_type
        self.method=method
        self.frame_index = 0

        if lm_type==68:
            #self.fan = fan()
            self.fan = pfpld(cpu=False)
            self.refrence = landmarks_2D_new

        else:
            raise Exception('landmark shape error')

        if method=='similarity':
            self.refrence=self.refrence

    def forward(self,img, boxes,kpss,limit=None, min_face_size=64.0, crop_size=(112, 112), apply_roi=False, multi_sample=True):
        if limit:
            boxes = boxes[:limit]
        # cv2.imshow('img', cv2.resize(img, (512, 512)))
        # if cv2.waitKey(1) == ord('q'):  # 按Q退出
        #     pass

        faces = []
        Ms = []
        rois = []
        masks = []
        for i,box in enumerate(boxes):
            if apply_roi:
                box = np.round(np.array(boxes[i])).astype(int)[:4]
                roi_pad_w = int(0.6 * max([box[2]-box[0],box[3]-box[1]]))
                roi_pad_h= int(0.4 * max([box[2]-box[0],box[3]-box[1]]))

                roi_box = np.array([
                    max(0, box[0] - roi_pad_w),
                    max(0, box[1] - roi_pad_h),
                    min(img.shape[1], box[2] + roi_pad_w),
                    min(img.shape[0], box[3] + roi_pad_h)
                ])
                rois.append(roi_box)
                roi = img[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]].copy()
                # cv2.imwrite("data/test/roi_{}_{}.jpg".format(self.frame_index, i), roi)
                self.frame_index += 1
                # mrow1 = roi_box[1]
                # mcol1 = roi_box[0]
                # roi_facial5points = facial5points.copy()

                # cv2.imshow('roi', roi)
                # if cv2.waitKey(1) == ord('q'):  # 按Q退出
                #     break
                mrow1 = roi_box[1]
                mcol1 = roi_box[0]
                facial5points=kpss[i]
                facial5points[:, 0] -= mcol1
                facial5points[:, 1] -= mrow1
                if multi_sample :
                    roi_facial5points_list=[]
                    move_list=[[0,0],[-1,-1],[1,1],[-1,1],[1,-1]]
                    distance = int(0.01 * max([box[2]-box[0],box[3]-box[1]]))
                    x1, y1, x2, y2 = box
                    w = x2 - x1 + 1
                    h = y2 - y1 + 1
                    size_w = int(max([w, h]) * 0.9)
                    size_h = int(max([w, h]) * 0.9)
                    height, width = img.shape[:2]
                    for i in range(1):
                        move=move_list[i]
                        cx = (x1 + x2) // 2+move[1]*distance
                        cy = (y1 + y2) // 2+move[0]*distance
                        x1 = cx - size_w // 2
                        x2 = x1 + size_w
                        y1 = cy - int(size_h * 0.4)
                        y2 = y1 + size_h
                        left = 0
                        top = 0
                        bottom = 0
                        right = 0
                        if x1 < 0:
                            left = -x1
                        if y1 < 0:
                            top = -y1
                        if x2 >= width:
                            right = x2 - width
                        if y2 >= height:
                            bottom = y2 - height

                        x1 = max(0, x1)
                        y1 = max(0, y1)

                        x2 = min(width, x2)
                        y2 = min(height, y2)
                        roi_2 = img[y1:y2, x1:x2]
                        roi_2 = cv2.copyMakeBorder(roi_2, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)
                        roi_box_2=[x1,y1,x2,y2]
                        # roi_pad_2 = int(0.2 * max([box[2] - box[0], box[3] - box[1]]))
                        # roi_box_2 = np.array([
                        #     max(0, box[0] - roi_pad_2+move[0]*distance),
                        #     max(0, box[1] - roi_pad_2+move[1]*distance),
                        #     min(img.shape[1], box[2] + roi_pad_2+move[0]*distance),
                        #     min(img.shape[0], box[3] + roi_pad_2+move[1]*distance)
                        # ])
                        # roi_2 = img[roi_box_2[1]:roi_box_2[3], roi_box_2[0]:roi_box_2[2]].copy()


                        if self.lm_type==68:
                            landmarks=self.fan.forward(roi_2)

                            # print("landmarks:", landmarks.shape)
                            # image = img.copy()
                            # point_size = 5
                            # point_color = (0, 0, 255)  # BGR
                            # thickness = -1
                            # for index in range(landmarks.shape[1]):
                            #     x = round(landmarks[0][index][0]) + x1
                            #     y = round(landmarks[0][index][1]) + y1
                            #     point = (x, y)
                            #     cv2.circle(image, point, point_size, point_color, thickness)
                            # cv2.imwrite('data/source/source_lm_066_68.png', image)



                            bbox={'left':0,'top':0,'bottom':roi_2.shape[1],'right':roi_2.shape[0]}
                            # a = drawLandmark_multiple(roi_2,bbox ,landmarks[0] )
                            # cv2.imshow('landmark',a,)
                            # cv2.waitKey(1)
                            if len(landmarks) >= 1:
                                roi_facial5points_tmp = landmarks[0]
                                roi_facial5points_tmp[:, 0] -= roi_box[0] - roi_box_2[0] + left
                                roi_facial5points_tmp[:, 1] -= roi_box[1] - roi_box_2[1] + top
                                roi_facial5points_list.append(roi_facial5points_tmp)
                        elif self.lm_type==468:
                            results = self.mp_m_landmark.process(roi)
                            if results.multi_face_landmarks is None:
                                landmarks=[]
                            else:
                                landmarks = np.array(
                                    [(lm.x, lm.y, lm.z) for lm in results.multi_face_landmarks[0].landmark])[:, :2]
                                landmarks = (landmarks * np.array([roi.shape[1], roi.shape[0]])).astype(np.int)[None,...]

                            if len(landmarks) >= 1:
                                roi_facial5points_tmp = landmarks[0]
                                roi_facial5points_list.append(roi_facial5points_tmp)

                    if len(roi_facial5points_list)>0:
                        # cv2.imshow('succsess', cv2.resize(roi_2,(512,512)))
                        # if cv2.waitKey(1) == ord('q'):  # 按Q退出
                        #     break
                        roi_facial5points=np.mean(roi_facial5points_list,axis=0)
                        if self.lm_type==68:
                            roi_facial5points = np.concatenate([roi_facial5points[17:49], roi_facial5points[54:55]])
                        elif self.lm_type == 468:
                            roi_facial5points = roi_facial5points[mesh_33]

                        if self.method=='affine':
                            if self.lm_type==468:
                                mat = get_transform_mat_all(roi_facial5points, self.refrence, output_size=crop_size[0],
                                                            scale=1.04,gcx=-0.02,gcy=0.25)
                            elif self.lm_type==68:
                                mat = get_transform_mat_all(roi_facial5points, self.refrence, output_size=crop_size[0],
                                                            scale=1.06,gcx=-0.02,gcy=0.21) # 1.06 0.9
                            warped_face=cv2.warpAffine(roi, mat, crop_size)
                            M = cv2.invertAffineTransform(mat)


                        # cv2.imshow('warped_face', warped_face)
                        # if cv2.waitKey(1) == ord('q'):  # 按Q退出
                        #     break
                        # cv2.imshow('warped_face_5',  warped_face_5)
                        # if cv2.waitKey(1) == ord('q'):  # 按Q退出
                        #     break

                        face = Image.fromarray(warped_face)
                        faces.append(face)
                        Ms.append(M)
                        # mask = np.full(crop_size, 255, dtype=float)
                        # mask = cv2.warpAffine(mask, M, roi.shape[:2][::-1])
                        # mask[mask > 20] = 255
                        mask= np.array([0,1])
                        masks.append(mask)
                    else:
                        # cv2.imshow('failure',  cv2.resize(roi,(512,512)))
                        # if cv2.waitKey(1) == ord('q'):  # 按Q退出
                        #     break
                        pass



        if apply_roi:
            return rois, faces, Ms, masks
        else:
            return boxes, faces, Ms
