import myparser
import cv2
import pytesseract 
import telebot
import os, time, sys
import requests
from argparse import Namespace
import time
from pathlib import Path
import cv2
from numpy.lib.function_base import append
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


opt = Namespace(weights=['runs/train/yolov5m_results3/weights/last.pt'], source='images/ims/', img_size=416, conf_thres=0.4, iou_thres=0.45, device='', view_img=False, 
        save_txt=True, save_conf=False, classes=None, agnostic_nms=False, augment=False, 
        update=False, project='runs/detect', name='exp', exist_ok=False)

#pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
bot = telebot.TeleBot(token="1993230425:AAEqbDCNCDGDcAJ00w1nBmk9loenYbMRcbc")
link = 'https://twitter.com/IKudryavtzeff'
xpath = '//article[@data-testid="tweet"]'
image_path = './images/ims/'
driver_path = '/usr/local/bin/geckodriver'
trig = ''

parser = myparser.Parser(link, xpath, image_path, driver_path)

triggers = ['doge', 'shib']
batch_size = 32
image_size = (254, 254)

class Checker():
    def __init__(self, triggers, batch_size, image_size, image_path):
        self.triggers = triggers
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_path = image_path
        self.trig = trig

    def connect(self):
        try:
            parser.connect_driver()
            time.sleep(2)
            parser.start()
            parser.last_time = parser.time_post + '1'
        except Exception as e:
            print('_________________________________________________________________________________________________\n')
            print('                              PRECONNECTING ERROR\n')
            print(f'{e}\n')
            print('_________________________________________________________________________________________________\n')
            bot.send_message('488664136', 'preconnecting error, restart')
            bot.send_message('293125099', 'preconnecting error, restart')
            os.system('pkill firefox')
            time.sleep(0.5)
            os.system('python3 restart.py')
            time.sleep(0.5)
            sys.exit()

    def start_cycle(self):
        try:
            while True:
                parser.get_tweet()
                time.sleep(0.3)
                if parser.last_time != parser.time_post:
                    parser.driver.implicitly_wait(5)
                    time.sleep(0.3)
                    self.tweet_text = parser.get_text()
                    if self.check_text():
                        self.message()
                        parser.last_time = parser.time_post
                        continue 
                    parser.driver.implicitly_wait(5)
                    parser.get_image()
                    time.sleep(0.5)
                    if self.check_image_text():
                        self.message()
                        parser.last_time = parser.time_post
                        continue
                    time.sleep(0.2)
                    if self.check_image():
                        self.message()
                        parser.last_time = parser.time_post
                        continue
                   
                    for elem in os.listdir(self.image_path):
                        os.remove(f'{self.image_path}{elem}')
                        
                parser.last_time = parser.time_post
                parser.driver.refresh()
                parser.driver.implicitly_wait(5)
                time.sleep(1)
        except Exception as e:
            print('_________________________________________________________________________________________________\n')
            print('                              CYCLE ERROR\n')
            print(f'{e}\n')
            print('_________________________________________________________________________________________________\n')
            bot.send_message('488664136', 'cycle error, restart')
            bot.send_message('293125099', 'cycle error, restart')
            os.system('pkill firefox')
            time.sleep(0.5)
            os.system('python3 restart.py')
            time.sleep(0.5)
            sys.exit()
                

    def check_text(self):
        try:
            for trigger in self.triggers:
                if trigger in str(self.tweet_text).lower():
                    self.trig = trigger
                    return True
        except Exception as e:
            print('_________________________________________________________________________________________________\n')
            print('                              TEXT CHECKING ERROR\n')
            print(f'{e}\n')
            print('_________________________________________________________________________________________________\n')

    def detect(self, save_img=False):
        source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://'))

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        if half:
            model.half()  # to FP16

    # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            save_img = True
            dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        
        self.nwords = []

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

        # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

        # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

        # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = Path(path[i]), '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = Path(path), '', im0s

                save_path = str(save_dir / p.name)
                txt_path = str(save_dir / 'labels' / p.stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            self.nwords.append(label.split(' '))
    
    def check_image(self):
        self.detect()
        try:
            for block in self.nwords:
                if 'doge' in block[0] and float(block[1]) >= 0.55:
                    print('SENT')
                    self.trig = 'doge'
                    return True
                elif 'shiba' in block[0] and float(block[1]) >= 0.55:
                    self.trig = 'shiba'
                    print('SENT')
                    return True
        except Exception as e:
            print('-------------------------------------------------------------------------------------------------\n')
            print('                              CHECK IMAGE ERROR\n')
            print(e)
            print('-------------------------------------------------------------------------------------------------\n')

    def check_image_text(self):
        try:
            images = os.listdir(self.image_path)
            for img in images:
                img = cv2.imread(f'{self.image_path}{img}')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = pytesseract.image_to_string(img)
                self.res = res.lower()
                print('-------------------------------------------------------------------------------------------------\n')
                print('                              TESSERACT RESULTS\n')
                print('-------------------------------------------------------------------------------------------------\n')
                for trigger in self.triggers:
                    if trigger in self.res:
                        self.trig = trigger
                        return True
                       
        except Exception as e:
            print('text_img_ERROR')
            print(e)
    
    
    def message(self):
        print(self.trig)
        print('SENT')
        try:
            r = requests.get('http://45.137.64.175:2001/ZldaOUMyTlBiU1hFdWpYRkZUbUFFNjdv/SHIB')
        except:
            bot.send_message('488664136', 'request failed')
            bot.send_message('293125099', 'request failed')
        bot.send_message('488664136', f'{self.trig}')
        bot.send_message('293125099', f'{self.trig}')
        try:
            for elem in os.listdir(self.image_path):
                os.remove(f'{self.image_path}{elem}')
        except:
            pass
        
def main():
    checker = Checker(triggers, batch_size, image_size, image_path)
    checker.connect()
    checker.start_cycle()
    
if __name__ == "__main__":
    main()
