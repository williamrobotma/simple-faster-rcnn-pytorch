import os
import numpy as np


class PipeDataset:
    """Bounding box dataset of PIPE landscapes.

    The index corresponds to each landscape.

    this dataset returns a corresponding
    :obj:`img, bbox, label`, a tuple of PIPE landscapes, bounding boxes and 
    labels.

    The bounding boxes are packed into a two dimensional tensor of shape
    :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
    four attributes are coordinates of the top left and the bottom right
    vertices.

    The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
    :math:`R` is the number of bounding boxes in the image.
    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`VOC_BBOX_LABEL_NAMES`.

    The type of the image, the bounding boxes and the labels are as follows.

    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`

    Args:
        data_dir (string): Path to the root of the training data. 
            i.e. "/data/image/voc/VOCdevkit/VOC2007/"
        split ({'train', 'val', 'trainval', 'test'}): Select a split of the
            dataset. :obj:`test` split is only available for
            2007 dataset.
    """

    def __init__(self, data_dir):

        self.label_names = LABEL_NAMES

        # get paths of PIPE landscapes
        self.img_files = glob.glob(os.path.join(folder_path,'features','*.npy'))

        # get names of each pair
        self.pair_names = []
        for img_path in self.img_files:
            self.pair_names.append(os.path.splitext(os.path.basename(img_path))[0])

        # get bboxes
        self.bboxes = {}
        with open(os.path.join(folder_path, 'coords.json'), 'r') as f:
            self.bboxes = json.load(f)

        assert len(self.bboxes) == len(self.pair_names)
        assert len(self.bboxes) == len(self.img_files)

    def __len__(self):
        return len(self.pair_names)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        pair_name = self.pair_names[i]
        img_path = self.img_files[i]
        coords = self.bboxes[pair_name]
        # anno = ET.parse(
            # os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = list()
        label = list()

        # bbox.append([
        #     int(bndbox_anno.find(tag).text) - 1
        #     for tag in ('ymin', 'xmin', 'ymax', 'xmax')])

        # format in json is: 
        # list: [[y1, y2], [x1, x2]]
        # The (x1, y1) position is at the top left corner,
        # the (x2, y2) position is at the bottom right corner
        bbox.append([coords[0][0]-1, coords[1][0]-1,
            coords[0][1]-1, coords[1][1]-1])
        # '[ymin', 'xmin', 'ymax', 'xmax']

        name = 'site'
        label.append(LABEL_NAMES.index(name))

        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)

        # Load PIPE landscapes
        # img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        # img = read_image(img_file, color=True)
        img = np.load(img_path, allow_pickle=False).astype(np.float32)
        
        return img, bbox, label

    __getitem__ = get_example


LABEL_NAMES = (
    'site',
)
