from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import cv2
import yaml
import pyrealsense2 as rs

def get_model_and_labels(model_path='yolov8n.pt',
                         labels_path='./ultralytics/datasets/coco.yaml'):
    model = YOLO(model_path)
    with open(labels_path) as f:
        labels = yaml.load(f, Loader=yaml.FullLoader)['names']
    return model, labels

def inference_image(model, image):
    bbox_list = []
    prob_list = []
    label_list = []
    
    results = model(image)

    preds = results[0].boxes.data.cpu().numpy()
    bbox_list = preds[..., :4]
    prob_list = preds[..., 4]
    label_list = preds[..., 5].astype(np.int32)

    return bbox_list, prob_list, label_list

def draw_image(image, bbox_list, prob_list, label_list, labels):
    for bbox, prob, label in zip(bbox_list, prob_list, label_list):
        p1 = bbox[:2].astype(np.int32)
        p2 = bbox[2:].astype(np.int32)
        text = f'{labels[label]}: {prob:.3}'
        try:
            cv2.rectangle(image, p1, p2, (255,0,0), 2)
            cv2.putText(image, text, [p1[0], p1[1]-5], cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255,0,0), 1)
        except:
            pass


def main2():
    # load weights model and labels
    model, labels = get_model_and_labels()
    # video writer set
    #(path, type, fps, (width, height))
    writer = cv2.VideoWriter('./output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (1280, 720))

    try:
        #realsense pipeline connection
        rs.pipeline()
        pipeline = rs.pipeline()
        pipeline.start()

        while(True):
            # receive 1 frame
            frames = pipeline.wait_for_frames()
            # split color, depth binary data frame
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            # binary data to numpy.array
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # inference
            #inference_result = inference_image(model, color_image)
            # draw result to img, * denote unwrap
            # draw_image(color_image, *inference_result, labels)
            # image show, [..., ::-1] denote rgb <-> bgr 
            cv2.imshow('real_time', color_image[..., ::-1])
            # open window, insert key
            key = cv2.waitKey(1)
            if key == 27:
                break

            # video write
            writer.write(color_image[..., ::-1])
    finally:
        writer.release()
        pipeline.stop()

main2()
