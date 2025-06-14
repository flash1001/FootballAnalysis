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
 
    #Draw Output
    ##Draw object Tracks
    output_video_frames = tracker.draw_anotations(video_frames, tracks)
    
    #Save video
    save_video(output_video_frames, 'output_videos/output_video.avi')
    
if __name__ == '__main__':
    main()