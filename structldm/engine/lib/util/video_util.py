import numpy as np
import cv2
import os
import shutil



def extrat_frames_by_frame_start_end(video_name, start_frame, end_frame, total_extracted_frames, img_id, pre_name,
                                     out_dir):
    cap = cv2.VideoCapture(video_name)

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    os.makedirs(out_dir, exist_ok=True)

    num = total_extracted_frames

    ratio = (end_frame - start_frame + 1) * 1.0 / num

    for i in range(num):
        which_frame = int(start_frame + ratio * i)

        cap.set(1, which_frame)
        res, frame = cap.read()

        w, h = frame.shape[:-1]

        my_video_name = os.path.join(out_dir, '%d_mv_%s_%d_%d.png' % (i, pre_name, img_id, which_frame))
        cv2.imwrite(my_video_name, frame)

    return 0

def extrat_frames_by_frame_start_num(video_name, start_frame, total_extracted_frames, img_id, pre_name, out_dir):
    cap = cv2.VideoCapture(video_name)

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    os.makedirs(out_dir, exist_ok=True)

    num = total_extracted_frames

    for i in range(num):
        which_frame = int(start_frame + i)

        cap.set(1, which_frame)
        res, frame = cap.read()

        w, h = frame.shape[:-1]

        my_video_name = os.path.join(out_dir, '%d_mv_%s_%d_%d.png' % (i, pre_name, img_id, which_frame))
        cv2.imwrite(my_video_name, frame)

    return 0


def extrat_frames_by_frame_set_ego_mv(ego_name, mv_name, mv_start_frame, mv_end_frame, ego_start_frame,
                                      total_extracted_frames, view_id, train_test, out_dir, ego_rate=2):
    if False:
        mv_name1 = 'D:/dataset/videos/fc2_save_2017-07-27-122826-0000.avi'
        mv_name2 = 'D:/dataset/videos/fc2_save_2017-07-27-122826-0001.avi'
        mv_name3 = 'D:/dataset/videos/fc2_save_2017-07-27-122826-0002.avi'

        mvlist = [mv_name1, mv_name2, mv_name3]
        for m in mvlist:
            mv_cap = cv2.VideoCapture(m)
            fps = mv_cap.get(cv2.CAP_PROP_FPS)
            frame_num = mv_cap.get(cv2.CAP_PROP_FRAME_COUNT)
            print('mv ', fps, frame_num)

    mv_cap = cv2.VideoCapture(mv_name)
    ego_cap = cv2.VideoCapture(ego_name)

    fps = mv_cap.get(cv2.CAP_PROP_FPS)
    frame_num = mv_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('mv ', fps, frame_num)

    fps = ego_cap.get(cv2.CAP_PROP_FPS)
    frame_num = ego_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('ego ', fps, frame_num)

    out_dir = out_dir + '/%s' % train_test
    os.makedirs(out_dir + '/ego', exist_ok=True)
    os.makedirs(out_dir + '/mv', exist_ok=True)

    num = total_extracted_frames
    mv_ratio = (mv_end_frame - mv_start_frame + 1) * 1.0 / num
    ego_ratio = mv_ratio * ego_rate

    for i in range(num):
        ego_which_frame = int(ego_start_frame + i * ego_ratio)
        ego_cap.set(1, ego_which_frame)
        res, frame = ego_cap.read()
        ego_frame = os.path.join(out_dir, 'ego', '%d_%d.png' % (i, ego_which_frame))
        cv2.imwrite(ego_frame, frame)

        mv_which_frame = int(mv_start_frame + i * mv_ratio)
        mv_cap.set(1, mv_which_frame)
        res, frame = mv_cap.read()
        my_video_name = os.path.join(out_dir, 'mv', '%d_%d_%d.png' % (i, view_id, mv_which_frame))
        cv2.imwrite(my_video_name, frame)

    ego_cap.release()
    mv_cap.release()
    return 0


def split_outdoor_mv(mv_name, mv_start_frame, mv_end_frame, ego_start_frame, out_dir, view_id, pre=''):
    cap = cv2.VideoCapture(mv_name)

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cnt = 0

    test_out_dir = out_dir + '/mv/test/img/%d' % view_id
    train_out_dir = out_dir + '/mv/train/img/%d' % view_id
    os.makedirs(test_out_dir, exist_ok=True)
    os.makedirs(train_out_dir, exist_ok=True)
    training_test_ratio = 0.85
    boundary = int(training_test_ratio * (mv_end_frame - mv_start_frame + 1) + mv_start_frame)

    print('mv ', mv_start_frame , mv_end_frame)

    for i in range(mv_start_frame, mv_end_frame):

        which_frame = i

        cap.set(1, which_frame)
        res, frame = cap.read()

        w, h = frame.shape[:-1]

        if i < boundary:
            tgt_file = os.path.join(train_out_dir,
                                    '%s_%d_%d_%d.jpg' % (pre, ego_start_frame + cnt, which_frame, view_id))
        else:
            tgt_file = os.path.join(test_out_dir,
                                    '%s_%d_%d_%d.jpg' % (pre, ego_start_frame + cnt, which_frame, view_id))

        cv2.imwrite(tgt_file, frame)

        cnt += 1

    cap.release()

    train_file_num = len(os.listdir(train_out_dir))
    test_file_num = len(os.listdir(test_out_dir))
    print('mv %d finish traing %d test %d' % (view_id, train_file_num, test_file_num))

    return 0


def split_outdoor_ego(ego_set_dir, start_frame, end_frame, out_dir, pre=''):
    test_out_dir = out_dir + '/ego/img/test'
    train_out_dir = out_dir + '/ego/img/train'

    os.makedirs(test_out_dir, exist_ok=True)
    os.makedirs(train_out_dir, exist_ok=True)

    training_test_ratio = 0.85
    boundary = int(training_test_ratio * (end_frame - start_frame + 1) + start_frame)

    print('ego ', start_frame, end_frame)

    for i in range(start_frame, end_frame):

        ego_which_frame = i
        ego_src_file = os.path.join(ego_set_dir, 'fc2_save_2017-10-11-135418-%04d.jpg' % ego_which_frame)

        if i < boundary:
            ego_tgt_file = os.path.join(train_out_dir, '%s_%d.jpg' % (pre, ego_which_frame))
        else:
            ego_tgt_file = os.path.join(test_out_dir, '%s_%d.jpg' % (pre, ego_which_frame))

        shutil.copy(ego_src_file, ego_tgt_file)

    train_file_num= len(os.listdir(train_out_dir))
    test_file_num = len(os.listdir(test_out_dir))

    print('ego finish traing %d test %d' % (train_file_num, test_file_num))



