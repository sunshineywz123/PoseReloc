class MovieWriter:
    def __init__(self):
        self.video_out_path = ''
        self.movie_cap = None
        self.id = 0

    def start(self):
        if self.movie_cap is not None:
            self.movie_cap.release()

    def write(self, im_bgr, video_out_path, text_info=[], fps=20):
        import cv2
        if self.movie_cap is None:
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            length = fps
            self.video_out_path = video_out_path
            self.movie_cap = cv2.VideoWriter(video_out_path, fourcc,
                                             length, (im_bgr.shape[1], im_bgr.shape[0]))

        if len(text_info) > 0:
            self.put_text(im_bgr, text_info[0], color=text_info[1])
        self.movie_cap.write(im_bgr)
        self.id += 1

    def put_text(self, img, inform_text, color=None):
        import cv2
        fontScale = 1
        if color is None:
            color = (255, 0, 0)
        org = (50, 50)
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        img = cv2.putText(img, inform_text, org, font,
                          fontScale, color, thickness, cv2.LINE_AA)
        return img

    def end(self):
        if self.movie_cap is not None:
            self.movie_cap.release()
            self.movie_cap = None
            print(f"Output frames:{self.id} to {self.video_out_path}")
            self.id = 0