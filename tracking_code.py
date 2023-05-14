from ultralytics import YOLO
import pyrealsense2 as rs
import cv2
import numpy as np
import frames_
import frame_
import object_
import yaml

#create class list that includes r_id
#class list will get included after checking with object.class 

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
    #print(results)
    preds = results[0].boxes.data.cpu().numpy()
    bbox_list = preds[..., :4]
    #print(bbox_list)
    #corresponds to the probability/confidence score of each predicted bounding box
    prob_list = preds[..., 4]
    #confidence score of the bounding box
    #print(prob_list)
    label_list = preds[..., 5].astype(np.int32)
    #print(label_list)
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


def crop_image(frame, bbox_list):
    #convert bounding box to integers
    x = bbox_list[0].astype(np.int32)
    y = bbox_list[1].astype(np.int32)
    w = bbox_list[2].astype(np.int32)
    h = bbox_list[3].astype(np.int32)
   
    #should this change?
    roi = frame[y:y+h, x:x+w]
   
    return roi


def main():
    
    frame_count = 0
    fs = frames_()
    r_id_matrix = []

    def r_id_add(self, object):
        self.r_id = object.centerpoint
        self.class_type = object.class_type
        self.keypoint = object.keypoint
        self.feature = object.features
        r_id_matrix.append(self)

    #load weights model and labels
    model, labels = get_model_and_labels()
    #open video file and process inference
    cap = cv2.VideoCapture('./test12_trim&crop.mp4')
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        #create frame_ instance 
        f = frame_()
        if ret:
            inference_result = inference_image(model, frame[..., ::-1])
            #only get bbox_list in  return
            bbox_list = inference_result[0]
            print(bbox_list) 
            #get all the bbox list of the current frame and crop
            crop_list = []
            
            if len(bbox_list) == 0:
                continue
            
            if len(bbox_list) >= 1:
                len_bbox_list = len(bbox_list)
                #print(bbox_list[0])
                for i in range(len_bbox_list):
                    #check how it crops
                    cv2.imshow('cropped image', crop_image(frame, bbox_list[i]))
                    #why can't it save all the outcome?
                    crop_list.append(crop_image(frame, bbox_list[i]))
                    cv2.imwrite(f'./output_image{i}.jpg', crop_image(frame, bbox_list[i]))

            #calculate keypoints of crop_list
            orb = cv2.ORB_create()
            keypoints_list = []
            features_list = []
            for i in range(len(crop_list)):
                keypoints, descriptors = orb.detectAndCompute(crop_list[i], None)
                keypoints_list.append(keypoints)
                features_list.append(descriptors)

            #calculate centerpoint of the crop_list
            centerpoint_list = []
            #first convert to gray scale
            for i in range(len(crop_list)):
                gray_img = cv2.cvtColor(crop_list[i], cv2.COLOR_BGR2GRAY)
                moment = cv2.moments(gray_img)
                x = int(moment["m10"]/moment["m00"])
                y = int(moment["m01"]/moment["m00"])
                tuple = (x,y)
                centerpoint_list.append(tuple)

            #set all the values of the object instance
            for i in range(len(crop_list)):
                #create object instance
                ob = object_()
                #update properly!
                ob.id = ob.update_id()
                ob.keypoints = keypoints_list[i]#put in the keypoints that have been calculated
                ob.features = features_list[i]#put in the features that have been found
                ob.centerpoint = centerpoint_list[i]#put in the centerpoint that has been found
                ob.class_type = inference_result[2]#put in the resulting class type from yolov8 inference
                #add to frame_ instance
                f.add(ob)
                r_id_add(ob)

            #append frame in frames
            fs.add(f)
            frame_count += 1

            #compare between objects within current frame with the previous frame's object
            #TODO:compare between current object's centerpoint and previous object's centerpoint

            #--------------- search function within the frames' frame list
            #how to search for prevprev frame?
            def search_frame(frames, frame_count):
                #how to know the index of the frame within frames list?            
                for i, element in enumerate(frames):
                    if i == 0:
                        continue
                    if i == (frame_count):
                        #finding previous frame from frames list
                        previous_element = frames[i-1]
                        return previous_element
            #-----------
            for i in range(len(f.objects_lists)):
                #check the objects have same class type
                if frame_count == 1:
                    continue
                else:
                    prev = search_frame(fs, frame_count)
                    for j in prev.objects_lists[j]:
                        #class type check
                        if f.objects_lists[i].class_type == prev.objects_lists[j].class_type #every object from previous frame
                            #get euclidean distance and if within threshold..
                            temp_cp = f.objects_lists[i].centerpoint
                            temp_cpprev = prev.object_lists[j].centerpoint
                            distance = ((temp_cp[0]-temp_cpprev[0])**2 + (temp_cp[1]-temp_cpprev[1])**2)
                            if distance > 10:
                                continue
                            else:
                                #if key/feature similar - use bf matching algorithm
                                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                                matches = matcher.match(queryDescriptors, trainDescriptors, mask)
                                    # Detect the keypoints and compute the descriptors for the query and train images
                                    query_kp, query_desc = detector.detectAndCompute(query_img, None)
                                    train_kp, train_desc = detector.detectAndCompute(train_img, None)
                                    if matches
                                    #set same id
                                    
                        else:
                             #if class type differ

                            
                #draw bounding box and id of the tracked object



            if cv2.waitKey(30) == ord('q'):
                break
            #draw_image(frame, *inference_result, labels)

            #cv2.imshow('frame', frame)
            #if cv2.waitKey(30) == ord('q'):
            #    break
        
    cap.release()
    cv2.destroyAllWindows()




'''

#get class information from Yolov8 detection outcome


---------------------------------
    class detected_class ():
        def __init__(self):
        self.list = []
        def add(self, tuple):
        self.list.append(tuple)
        def get_r_id(self):
        return self.list 
---------------------------------
    #connect to camera


    #create frames.instance
    fs1 = frames()
    #from realsense streaming get 1 frame to conduct the following

        # conduct inference. Get Yolov8 weight
        #<check output of Yolov8>

        #cropped image of objects temp list
        temp_objects = []
        #conduct cropping of objects in image
        #create frame.instance
        f1 = frame()

        
        #conduct finding keypoint and ORB algorithm for the cropped image
        temp_keypoints = []
    
        #get class type

        #get centerpoint

        #create object.instance for every object detected in the current frame
        for i in range(len(temp_objects)):
            i = object()
            #add object instance to 'cropped' object list in 1 frame.
            f1.add(i)

        #set all the values in object instance
        for i in range(len(f1.objects.lists)):
            i.id = i.update_id()
            i.keypoints = temp_keypoints[i]#put in the keypoints that have been calculated
            i.features = #put in the features that have been found
            i.centerpoint = #put in the centerpoint that has been found
            i.class_type = #put in the resulting class type from yolov8 inference
            

        #check with class list which contains accumulation of r_ids ultimately for tracking
        #check with the previously saved r_id

            #if the object belongs to class 'ball'
                #check which r_id list it belongs to by calculating euclidean distance
                #using last r_id value and object's centerpoint
----------------------------------
for all rows in r_id_matrix:
    if different class_type
        create new row in r_id_matrix

    if same 
        compare the two objects by using centerpoint and check if within the threshold value
            append to r_id row if same
            if len(self.r_id) > self.max_length:
            self.r_id = self.r_id[31:] 
------------------------------------

                    #if out of bound, create new r_id list


                    #if exist, calculate similarity between newly added object's keypoint value
                    #and previous object's keypoint value

                        #if out of bound, create new r_id list


                        #if in bound, append to r_id list


            #if no class exist:
                #create new class and creat new r_id list and append
    
        #append the frame instance to frames list.
        fs1.add(f1)



    #tracking object using their id - BF matching



if __name__ == '__main__':
    main()

'''

main()
