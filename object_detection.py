from ultralytics import YOLO
import pyrealsense2 as rs
import cv2
import numpy as np
import yaml

def get_model_and_labels(model_path='yolov8x.pt',
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

def main():

    model, labels = get_model_and_labels()
    cap = cv2.VideoCapture('./test12.avi')
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret:
            inference_result = inference_image(model, frame[..., ::-1])
            print(inference_result)
        
            draw_image(frame, *inference_result, labels)
            print(inference_result)
            cv2.imshow('frame', frame)
            if cv2.waitKey(30) == ord('q'):
               break
        
    cap.release()
    cv2.destroyAllWindows()

main()