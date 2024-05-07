import argparse
import torchvision
import torch
from torchvision.transforms import Resize, RandomCrop
from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation, DetrImageProcessor, DetrForObjectDetection
from os import path 
import scipy
import av
import csv 

def square_crop(image, box):
    width = box[2] - box[0]
    height = box[3] - box[1]
    size = max(height, width)
    if height > width:
        crop = torchvision.transforms.functional.crop(image, top=int(box[1]), left=int((box[2] + box[0])/2 - size/2), height=int(size), width=int(size))
    elif height < width:
        crop = torchvision.transforms.functional.crop(image, top=int((box[3] + box[1])/2 - size/2), left=int(box[0]), height=int(size), width=int(size))
    return crop

def get_mask(frame, image_processor, model):
    inputs = image_processor(images=frame, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    semantic_segmentation = image_processor.post_process_semantic_segmentation(outputs)[0]
    return semantic_segmentation

def get_bbox(image, processor, model):
  # you can specify the revision tag if you don't want the timm dependency
  with torch.no_grad():
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

  # convert outputs (bounding boxes and class logits) to COCO API
  # let's only keep detections with score > 0.8
  target_sizes = torch.tensor([[image.shape[1], image.shape[2]]])
  results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.8)[0]

  mask = (results["labels"] == model.config.label2id['skateboard'])
  boxes = results["boxes"][mask]
  if not boxes.numel(): None

  biggest_box = 0
  best_box = 0
  for i in range(boxes.shape[0]):
    box_size = (boxes[i,2] - boxes[i,0]) * (boxes[i,3] - boxes[i,1])
    if box_size > biggest_box:
      biggest_box = box_size
      best_box = i

  box = torch.round(boxes[best_box,:])
  return box

def process_frame(frame, 
                  frame_id, 
                  vid_id, 
                  frame_num, 
                  bboxs_old, 
                  frames_directory,
                  background_directory, 
                  legs_directory, 
                  bbox_csv, 
                  transform,
                  box_model,
                  box_processor,
                  leg_model,
                  leg_processor
                  ):
    c, h, w = frame.shape
    assert(c == 3)
    # Save two random background images
    background1 = transform(RandomCrop(256)(frame[:,round(h/3.0):,:round(w/3.0)]))  
    background2 = transform(RandomCrop(256)(frame[:,round(h/3.0):,round(w/3.0)*2:]))   
    torchvision.utils.save_image(background1 / 255.0, path.join(background_directory, f'{frame_id}_0.jpg'))
    torchvision.utils.save_image(background2 / 255.0, path.join(background_directory, f'{frame_id}_1.jpg'))

    # find bounding box
    if frame_id in bboxs_old: 
        box = bboxs_old[frame_id]
    else:
        try:
            box = get_bbox(frame, box_processor, box_model)
        except:
            return
    if box == None: return
    try:
        cropped_frame = square_crop(frame, box)
    except:
        return

    # save frame
    torchvision.utils.save_image(transform(cropped_frame) / 255.0, path.join(frames_directory, frame_id + '.jpg'))

    # write bounding box info to csv
    with open(bbox_csv_path, 'a', newline='') as bbox_csv:
        bbox_csv.write(f'{frame_id},{vid_id},{frame_num},{int(box[0].item())},{int(box[1].item())},{int(box[2].item())},{int(box[3].item())}\n')

    # get leg mask
    semantic_segmentation = get_mask(cropped_frame, leg_processor, leg_model)
    mask = semantic_segmentation == 0.0
    if torch.sum(mask) == 0:
        return 
    mask = Resize(cropped_frame.shape[1])(mask.unsqueeze(0))
    
    cropped_frame *= mask
    cropped_frame = transform(cropped_frame)
    mask = transform(mask)
    legs = torch.cat((cropped_frame, mask), dim=0)
    # print(legs.shape)
    # plt.imshow(cropped_frame.permute(1, 2, 0))
    torch.save(legs, path.join(legs_directory, f'{frame_id}_legs.pt'))
    return
    

def process_video(video_directory, 
                  video_path, 
                  frames_directory,
                  background_directory, 
                  legs_directory, 
                  bbox_csv, 
                  transform, 
                  bboxs_old,
                  box_model,
                  box_processor,
                  leg_model,
                  leg_processor
                  ):
    vid = torchvision.io.read_video(path.join(video_directory, video_path))
    vid_id = video_path.split('.')[0]
    print(f'Processing video {vid_id}.')
    frames = vid[0]
    # transform = Compose([Resize(32)])
    frame_id = 0
    for i in range(frames.shape[0]):
        frame = frames[i,:,:,:].permute(2, 0, 1)
        process_frame(frame, 
                      f'{vid_id}_{frame_id}', 
                      vid_id, 
                      frame_id, 
                      bboxs_old, 
                      frames_directory,
                      background_directory, 
                      legs_directory, 
                      bbox_csv, 
                      transform,
                      box_model,
                      box_processor,
                      leg_model,
                      leg_processor
                      )
        frame_id += 1

def get_old_bbox(csv_path):
    bboxs = {}
    if not path.isfile(csv_path):
        return bboxs
    with open(csv_path, 'r') as data:
        count = 0
        for line in csv.reader(data):
            if count == 0:
                count += 1
                continue
            count += 1
            id = line[0].strip()
            bboxs[id] = [int(line[i].strip()) for i in range(3,7)]
            if sum(bboxs[id]) == 0:
                bboxs[id] = None
    return bboxs

if __name__ == '__main__':
    device = 'cuda'
    from glob import glob
    video_directory = 'data/batb1k/videos'
    video_paths = [f.split('/')[-1] for f in glob(path.join(video_directory, '*.mov'))]
    transform = Resize(128)
    frames_directory = 'data/batb1k/frames128'
    backgrounds_directory = 'data/batb1k/backgrounds128'
    legs_directory = 'data/batb1k/leg_masks128'
    bbox_csv_path = 'data/batb1k/new_bboxs.csv'
    old_bbox_csv_path = 'data/batb1k/bounding_box_data.csv'

    # if not path.isfile(bbox_csv_path):
    fields = ['frame_id', 'clip_id', 'frame_num', 'x1', 'y1', 'x2', 'y2']
    with open(bbox_csv_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames = fields)
        writer.writeheader()

    checkpoint_name = 'facebook/maskformer-swin-small-coco'
    leg_model = MaskFormerForInstanceSegmentation.from_pretrained(checkpoint_name)
    leg_processor = MaskFormerImageProcessor.from_pretrained(checkpoint_name)
    box_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    box_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    box_model.eval()
    leg_model.eval()
    for vp in video_paths:
        vp = '18.mov'
        process_video(video_directory, 
                      vp, 
                      frames_directory, 
                      backgrounds_directory, 
                      legs_directory, 
                      bbox_csv_path,
                      transform,
                      get_old_bbox(old_bbox_csv_path),
                      box_model,
                      box_processor,
                      leg_model,
                      leg_processor
                      )
        break