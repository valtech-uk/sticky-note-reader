import requests
import time
from PIL import Image
from io import BytesIO
import queue
import threading
import cv2 as cv

class TextDetector:
    def __init__(self, subscription_key, url, use_confidence = True):
        self.sub_key = subscription_key
        self.url = url
        self.request_queue = queue.Queue()
        self.complete_queue = queue.Queue()
        self.use_confidence = use_confidence

    def batch_infer(self, images):
        retriever = threading.Thread(target = self.retrieve_thread)
        retriever.start()
        for image in images:
            print('request sent')
            url = self.infer(image)
            self.request_queue.put(url)
        self.request_queue.put(None)
        self.request_queue.join()
        results = []
        for i in range(len(images)):
            results.append(self.complete_queue.get())
        retriever.join()
        return results

    def retrieve_thread(self):
        retrieve = True
        while retrieve:
            try:
                url = self.request_queue.get()
            except queue.Empty:
                time.sleep(1)
            else:
                if url is None:
                    retrieve = False
                    self.request_queue.task_done()
                    continue
                else:
                    self.complete_queue.put(self.retrieve(url))
                    print('request retrieved')
                    self.request_queue.task_done()

    def infer(self, image):
        headers = {'Ocp-Apim-Subscription-Key': self.sub_key, 
           'Content-Type': 'application/octet-stream'}
        response = requests.post(self.url, headers=headers, data=image)
        response.raise_for_status()
        operation_url = response.headers["Operation-Location"]
        return operation_url

    def retrieve(self, url):
        headers = {'Ocp-Apim-Subscription-Key': self.sub_key}
        analysis = {}
        poll = True
        while (poll):
            response_final = requests.get(url, headers=headers)
            analysis = response_final.json()
            time.sleep(1)
            if ("recognitionResults" in analysis):
                poll = False
            if ("status" in analysis and analysis['status'] == 'Failed'):
                poll = False
        return self.line_from_json(analysis)

    def line_from_json(self, analysis):
        all_words = []
        for line in analysis['recognitionResults'][0]['lines']:
            for word in line['words']:
                if self.use_confidence:
                    if 'confidence' not in word or word['confidence'] != 'Low':
                        all_words.append(word['text'])
                else:
                    all_words.append(word['text'])
        print(all_words)
        all_text = ' '.join(all_words)
        print(all_text)
        return all_text


if __name__ == "__main__":
    det = TextDetector("26a32a30c1bd44c1b73dd5827bcc5a13")
    images = []
    ret, image = cv.imencode(".jpg", cv.imread("output_images/image949x978.jpg", 1))
    bytesimg = image.tobytes()
    images.append(bytesimg)
    '''
    print(cv.imencode(".jpg", cv.imread("output_images/image949x978.jpg", 1))[1].shape)
    images.append(cv.imencode(".jpg", cv.imread("output_images/image1445x1473.jpg", 1))[1])
    images.append(cv.imencode(".jpg", cv.imread("output_images/image1017x992.jpg", 1))[1])
    '''
    response = det.batch_infer(images)
    print(response)

