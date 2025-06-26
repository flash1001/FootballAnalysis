import cv2
from utils import save_video, read_video
from tracker import Tracker

def main():
    #Read Video
    video_frames = read_video('input_videos/08fd33_4.mp4')
    
    #Initialize Tracker
    tracker = Tracker('models/best.pt')
    
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
 
    #Save cropped image of player
    for track_id, player in tracks['players'][0].items():
        bbox = player['bbox']
        frame = video_frames[0]
        
        #Crop bbox from frame
        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        #Save the cropped image
        cv2.imwrite(f'output_videos/cropped_image.jpg', cropped_image)
        break
    
    #Draw Output
    ##Draw object Tracks
    output_video_frames = tracker.draw_anotations(video_frames, tracks)
    
    #Save video
    save_video(output_video_frames, 'output_videos/output_video.avi')
    
if __name__ == '__main__':
    main()