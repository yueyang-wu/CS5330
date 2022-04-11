# import the opencv library
import cv2
import numpy as np
import torch

import utils
from utils import MyNetwork


def main():
    # load the model
    model = torch.load('model.pth')
    model.eval()

    # define a video capture object
    vid = cv2.VideoCapture(0)

    while True:

        # Capture the video frame by frame
        ret, original_frame = vid.read()

        if ret:
            # preprocess it
            frame = cv2.resize(original_frame, (28, 28))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = np.expand_dims(frame, 0)
            frame = frame.astype(np.float32)
            frame /= 255.0
            print(frame.dtype)
            frame = torch.from_numpy(frame)

            print(frame.type())

            output = model(frame)
            pred = output.data.max(1, keepdim=True)[1]
            print(pred)

            # Display the resulting frame
            cv2.imshow('original_frame', original_frame)

            # the 'q' button is set as the
            # quitting button
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
