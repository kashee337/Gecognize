import argparse
import os
import pickle
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from dp_matching import DPMatching


def get_landmarks(mp_landmarks, size=(720, 1280)):
    h, w = size
    landmarks = np.asarray(
        [[l.x * w, l.y * h] for l in mp_landmarks],
        dtype=np.int32,
    )
    return landmarks


def normalize_traj(data):
    data = np.asarray(data, dtype=np.float32)
    min_x = np.min(data.T[0])
    max_x = np.max(data.T[0])
    min_y = np.min(data.T[1])
    max_y = np.max(data.T[1])
    data -= np.array([min_x, min_y])
    data /= np.array([max_x - min_x, max_y - min_y])
    return data


class Canvas:
    def __init__(self, pos=(700, 100), size=(500, 500), lifespan=50):
        self.pt1 = pos
        self.pt2 = [pos[0] + size[0], pos[1] + size[1]]
        self.lifespan = lifespan
        self.time = lifespan
        self.text = ""

    def set_text(self, text):
        self.text = text
        self.time = 0

    def draw(self, image, isDrawing):

        if isDrawing:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        cv2.rectangle(image, self.pt1, self.pt2, color=color, thickness=3)
        if self.time < self.lifespan:
            cv2.rectangle(image, (10, 10), (200, 200), (255, 255, 255), -1)
            cv2.putText(
                image,
                self.text,
                (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                5,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

            self.time += 1


def main(cfg):

    # init
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    drawFace = False
    drawHand = True
    drawPose = False
    isRecord = False
    isDrawing = False
    holistic = mp_holistic.Holistic(
        min_detection_confidence=cfg.min_detection_confidence,
        min_tracking_confidence=cfg.min_tracking_confidence,
        # upper_body_only=cfg.upper_body_only,
    )
    cap = cv2.VideoCapture(cfg.device)
    dp_matcher = DPMatching(cfg.template_dir)
    canvas = Canvas()

    while cap.isOpened():
        ret, image = cap.read()

        if not ret:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)

        # draw
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.face_landmarks is not None:
            landmarks = get_landmarks(results.face_landmarks.landmark, image.shape[:2])

        if drawFace:
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS
            )
        if drawHand:
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )
        if drawPose and not cfg.upper_body_only:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
            )

        key = cv2.waitKey(5)
        if key & 0xFF == 27:
            break
        elif key == ord("f"):
            drawFace = not drawFace
        elif key == ord("h"):
            drawHand = not drawHand
        elif key == ord("r"):
            isRecord = not isRecord
            outdir = datetime.now().strftime("%Y%m%d%H%M%S")
            cnt = 0
            if isRecord:
                print("strat!")
            else:
                print("finish!")

        elif key == ord("d"):
            isDrawing = not isDrawing
            if isDrawing:
                print("draw")
            else:
                if len(traj) > 0:
                    res, score = dp_matcher(normalize_traj(traj))
                    print(res, score)
                    canvas.set_text(res)
            traj = []

        if isRecord:
            os.makedirs(outdir, exist_ok=True)
            print("recog")
            if results.left_hand_landmarks is None:
                print("defect")
            else:
                with open(os.path.join(outdir, f"{cnt:05d}.pickle"), "wb") as f:
                    pickle.dump(results.left_hand_landmarks, f)
                cnt += 1

        if isDrawing:
            if results.left_hand_landmarks is not None:
                pt = results.left_hand_landmarks.landmark[8]
                traj.append([pt.x, pt.y])
            for x, y in traj:
                h, w, _ = image.shape
                cv2.circle(image, (int(w * x), int(h * y)), 5, (255, 0, 0), -1)

        canvas.draw(image, isDrawing)
        cv2.imshow("MediaPipe Holistic", image)

    holistic.close()
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--template_dir", type=str, default="../data/template")
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    args = parser.parse_args()
    try:
        main(args)
    except AssertionError:
        traceback.print_exc()
