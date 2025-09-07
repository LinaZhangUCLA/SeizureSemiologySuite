import subprocess
import json
from collections import Counter
from pathlib import Path
import sys

def get_duration(video_path):
    try:
        result = subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', str(video_path)], 
                              capture_output=True, text=True, check=True)
        return round(float(json.loads(result.stdout)['format']['duration']), 1)
    except:
        return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory>")
        sys.exit(1)
    
    directory = Path(sys.argv[1])
    extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v']
    
    durations = []
    for ext in extensions:
        for video in directory.rglob(f'*{ext}'):
            duration = get_duration(video)
            if duration:
                durations.append(duration)
    
    counter = Counter(durations)
    for duration, count in counter.most_common():
        print(f"{duration}s : {count}")

if __name__ == "__main__":
    main()