from mtcnn.mtcnn import MTCNN

class mtcnn_to_face_alignment:
    def __init__(self):
        self.detector = MTCNN()

    def find_bboxes(self, input_img):
        """
        Recieve: an image

        Return:  list of bbox format (x1, y1, x2, y2)
        """
        faces_positions = self.detector.detect_faces(input_img)
        #format: (x1, y1, w, h) y probabilidad pero no interesa
        bbox_list = []
        if (len(faces_positions)!=0):
            for face in faces_positions:
                bbox = face['box']
                bbox_formated = [[bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]]
                bbox_list.append(bbox_formated)
        return bbox_list

    def size_of_bbox(self, bbox):
        """
        return:
            multiplication of width x height
        """
        return bbox[2]*bbox[3]
    #terminar
    def mtcnn_bbox_face_alignment_format_a(self, input_img):
        """
        Recieve: an image

        Return: the biggest bbox format (x1, y1, x2, y2)
        """
        max_size_box = 0
        size_box = 0
        bbox_to_return = None
        faces_positions = self.detector.detect_faces(input_img)
        #format: (x1, y1, w, h) y probabilidad pero no interesa
        bbox_list = []
        if (len(faces_positions)!=0):
            for face in faces_positions:
                bbox = face['box']
                size_box = size_of_bbox(bbox)
                if (size_box > max_size_box):
                    bbox_to_return = bbox
            ###terminar para bbox return  en nuevo formato
            bbox = bbox_to_return
            bbox_formated = [[bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]]
            bbox_list.append(bbox_formated)
        return bbox_list
