import os
import argparse
import ffmpeg
from PIL import Image
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract frames and audio from videos')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing video files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save extracted data')
    parser.add_argument('--segment_length', type=int, default=30.1, help='Segment length in seconds (default: 30)')
    parser.add_argument('--fps', type=int, default=2, help='Frames per second to extract (default: 2)')
    parser.add_argument('--audio_sr', type=int, default=16000, help='Audio sample rate (default: 16000)')
    return parser.parse_args()

def get_video_duration(video_path):
    """Get video duration in seconds"""
    try:
        probe = ffmpeg.probe(video_path)
        duration = float(probe['streams'][0]['duration'])
        return duration
    except:
        print(f"Warning: Could not get duration for {video_path}")
        return None

def get_segments(video_path, segment_length):
    """Calculate segment start/end times"""
    duration = get_video_duration(video_path)
    if duration is None:
        return []
    
    segments = []
    start = 0
    while start < duration:
        end = min(start + segment_length, duration)
        segments.append((start, end))
        start = end
    
    return segments

def extract_frames(video_path, start_time, end_time, fps, output_folder):
    """Extract frames from video segment using ffmpeg"""
    os.makedirs(output_folder, exist_ok=True)
    
    duration = end_time - start_time
    frame_pattern = os.path.join(output_folder, 'frame_%03d.jpg')
    
    try:
        (
            ffmpeg
            .input(video_path, ss=start_time, t=duration)
            .filter('fps', fps=fps)
            .output(frame_pattern, vcodec='mjpeg', pix_fmt='yuvj420p')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
        
        # Count extracted frames
        frame_files = [f for f in os.listdir(output_folder) if f.startswith('frame_')]
        return len(frame_files)
        
    except Exception as e:
        print(f"Error extracting frames: {e}")
        return 0

def extract_audio(video_path, start_time, end_time, audio_sr, output_path):
    """Extract audio from video segment using ffmpeg"""
    duration = end_time - start_time
    
    try:
        (
            ffmpeg
            .input(video_path, ss=start_time, t=duration)
            .output(output_path, acodec='pcm_s16le', ar=audio_sr, ac=1)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
        return True
        
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return False

def process_video(video_path, video_name, output_dir, segment_length, fps, audio_sr):
    """Process one video: segment it and extract frames + audio"""
    print(f"Processing: {video_name}")
    
    # Get segments
    segments = get_segments(video_path, segment_length)
    if not segments:
        print(f"  Skipping - could not process {video_name}")
        return 0, 0, 0
    
    # Create video folder
    video_folder = os.path.join(output_dir, video_name.replace('.mp4', ''))
    
    total_frames = 0
    total_audio_files = 0
    
    for i, (start_time, end_time) in enumerate(segments):
        print(f"  Segment {i+1}/{len(segments)}: {start_time:.1f}s - {end_time:.1f}s")
        
        # Create segment folder
        segment_folder = os.path.join(video_folder, f'segment_{i:03d}')
        frames_folder = os.path.join(segment_folder, 'frames')
        audio_path = os.path.join(segment_folder, 'audio.wav')
        
        # Extract frames
        num_frames = extract_frames(video_path, start_time, end_time, fps, frames_folder)
        total_frames += num_frames
        
        # Extract audio
        audio_success = extract_audio(video_path, start_time, end_time, audio_sr, audio_path)
        if audio_success:
            total_audio_files += 1
        
        print(f"    Extracted {num_frames} frames, audio: {'✓' if audio_success else '✗'}")
    
    return len(segments), total_frames, total_audio_files

def main():
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Video Extraction Script")
    print("=" * 50)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Segment length: {args.segment_length} seconds")
    print(f"Frame rate: {args.fps} FPS")
    print(f"Audio sample rate: {args.audio_sr} Hz")
    print("=" * 50)
    
    # Get video files
    video_files = [f for f in os.listdir(args.input_dir) 
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    video_files = sorted(video_files)
    
    print(f"Found {len(video_files)} video files")
    
    # Process each video
    total_videos = 0
    total_segments = 0
    total_frames = 0
    total_audio_files = 0
    
    for video_name in video_files:
        video_path = os.path.join(args.input_dir, video_name)
        
        # Check if already processed
        video_output_folder = os.path.join(args.output_dir, video_name.replace('.mp4', ''))
        if os.path.exists(video_output_folder):
            print(f"Skipping {video_name} (already processed)")
            continue
        
        try:
            segments, frames, audio_files = process_video(
                video_path, video_name, args.output_dir, 
                args.segment_length, args.fps, args.audio_sr
            )
            
            total_videos += 1
            total_segments += segments
            total_frames += frames
            total_audio_files += audio_files
            
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
            continue
    
    # Final summary
    print("=" * 50)
    print("EXTRACTION SUMMARY")
    print("=" * 50)
    print(f"Videos processed: {total_videos}")
    print(f"Segments created: {total_segments}")
    print(f"Frames extracted: {total_frames}")
    print(f"Audio files saved: {total_audio_files}")
    print(f"Data saved to: {args.output_dir}")

if __name__ == "__main__":
    main()