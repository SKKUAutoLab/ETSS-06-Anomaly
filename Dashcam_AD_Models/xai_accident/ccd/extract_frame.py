import cv2
import os

video_directory = 'Crash-1500/'
destination = 'positive/'
video_list = os.listdir(video_directory)
count = 0
for i in range(len(video_list)):
    filename = os.path.join(video_directory, video_list[i])
    vname_slice = video_list[i].split('.')
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print("Could not open video", filename)
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('Extracting frames from video:', video_list[i])
        for loop in range(total_frames):
            cap = cv2.VideoCapture(filename)
            cap.set(1, loop)
            success = cap.grab()
            ret, image = cap.retrieve()
            try:
                image = cv2.resize(image, (224, 224))
            except:
                continue
            frame_name = vname_slice[0] + '_frame_%d.jpg' % loop
            saved_path = '/' + vname_slice[0]
            destination_2 = destination + saved_path
            if not os.path.exists(destination_2):
                os.makedirs(destination_2)
            destination_dir = os.path.join(destination_2, frame_name)
            cv2.imwrite(destination_dir, image)
    count = count + 1
    cap.release()
    cv2.destroyAllWindows()
