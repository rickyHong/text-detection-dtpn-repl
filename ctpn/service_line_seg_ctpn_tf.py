#encoding=utf8
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, sys, cv2
import glob
import shutil

import random
import json
import base64
import time
import logging
import argparse
import tornado.ioloop
import tornado.web
import tornado.httpserver


CTPN_DIR = '/Users/zhangxin/github/text-detection-ctpn'
sys.path.append(CTPN_DIR)
# sys.path.append(os.getcwd())

from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg,cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg

checkpoints_path = os.path.join(CTPN_DIR, 'checkpoints/')
log_dir = './log_ctpn/'
imgtempdir = './tmp_ctpn/'

def resize_im(im, scale, max_scale=None):
    f=float(scale)/min(im.shape[0], im.shape[1])
    if max_scale!=None and f*max(im.shape[0], im.shape[1])>max_scale:
        f=float(max_scale)/max(im.shape[0], im.shape[1])
    return cv2.resize(im, None,None, fx=f, fy=f,interpolation=cv2.INTER_LINEAR), f



class MainGetHandler(tornado.web.RequestHandler):
    """Main Get Handler
    """
    def recog(self):
        """recive image , return recog result
        """
        logging.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        # 获取图像的 base64 编码
        params = json.loads(self.request.body)

        if 'image_base64' in params:
            image_base64 = params["image_base64"]
        else:
            return {'message':'no image_base64', 'returncode':10001}
        # 从base64 编码得到图像文件
        try:
            imgfile = base64.b64decode(image_base64)
        except:
            return {'message':'base64.b64decode error', 'returncode':10002}

        # 将图片解码
        # try:
        #     file_bytes = np.asarray(bytearray(imgfile), dtype=np.uint8)
        #     img = cv2.imdecode(file_bytes, cv2.CV_LOAD_IMAGE_UNCHANGED)
        # except:
        #     return {'message':'cv2.imdecode error', 'returncode':10003}

        # 图片写到本地再读取
        strtime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        imgfilename = os.path.join(imgtempdir, strtime + str(random.randint(10000, 99999)) + '.jpg')
        logging.info('imgfilename : ' + imgfilename)
        try:
            with open(imgfilename, 'w') as obj:
                obj.write(imgfile)
        except:
            return {'message':'write temp img file error', 'returncode':10003}


        im = cv2.imread(imgfilename)
        im, f = resize_im(im, TextLineCfg.SCALE, TextLineCfg.MAX_SCALE)
        # text_lines = text_detector.detect(im)
        # text_lines = text_lines / f
        scores, boxes = test_ctpn(sess, net, im)
        textdetector = TextDetector()
        boxes = textdetector.detect(boxes, scores[:, np.newaxis], im.shape[:2])
        # draw_boxes(img, image_name, boxes, scale)

        result = {}
        result['message'] = 'OK'
        result['returncode'] = 0
        result['result'] = {}
        line_data = []
        for box in boxes:
            item = {}
            b = [0] * 8
            for i in range(8):
                b[i] = int(box[i]/f + 0.5)
            item['box'] = b
            item['conf'] = box[8]
            line_data.append(item)
        result['result']['lines'] = line_data

        logging.info(str(result))
        return result

    def get(self):
        self.write(json.dumps(self.recog()))
    def post(self):
        self.write(json.dumps(self.recog()))
    def data_received(self, chunk):
        """data received"""
        pass


def main(args):
    """main function
    """
    strtime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    if not os.path.isdir(imgtempdir):
        os.mkdir(imgtempdir)

    filename = os.path.join(log_dir,
                            strtime + '.dmocr_textlines_detect.' + str(args.port) + ".log")
    logging.basicConfig(
        filename=filename,
        level=logging.INFO,
        format='[%(levelname)s] (%(process)d) (%(threadName)-10s) %(message)s',
    )
    print("Listen...")
    # tornado.options.parse_command_line()
    application = tornado.web.Application([(r"/ocr_segment_line", MainGetHandler)])
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(args.port)
    tornado.ioloop.IOLoop.instance().start()


def get_args():
    '''get args
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=23331, type=int)
    # parser.add_argument('--gpu', default=0, type=int, help='gpu id, TEST_GPU_ID')
    return parser.parse_args()





if __name__ == "__main__":
    cfg_from_file(os.path.join(CTPN_DIR, 'ctpn/text.yml'))
    # cfg_from_file('ctpn/text.yml')

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()

    # try:
    #     ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
    #     print('Restoring from {}...'.format(cfg.model_checkpoint_path), end=' ')
    #     saver.restore(sess, cfg.model_checkpoint_path)
    #     print('done')
    # except:
    #     raise 'Check your pretrained {:s}'.format(cfg.model_checkpoint_path)
    ckpt = tf.train.get_checkpoint_state(checkpoints_path)
    print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('done')
    main(get_args())
