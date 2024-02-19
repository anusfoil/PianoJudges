import subprocess
import os
import json
import csv
import glob
import pandas as pd
from .. import hook


def should_download(title, duration, filter_keywords, max_duration=600):
    """Check if the video should be downloaded based on the title and duration.
    
    Args:
        title (str): The title of the video.
        duration (int): The duration of the video in seconds.
        filter_keywords (list): A list of keywords to filter out.
        max_duration (int): The maximum duration allowed for a video in seconds.
        
    Returns:
        bool: True if the video should be downloaded, False otherwise.
    """
    # Check title against filter keywords
    for keyword in filter_keywords:
        if keyword.lower() in title.lower():
            return False
    
    # Check duration
    if duration > max_duration:
        return False
    
    return True

def download_channel_videos(url, filter_keywords):
    # Command to fetch video information
    if "channel" in url or "user" in url:
        command = ['yt-dlp', '--dump-json', '--verbose', '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', '--flat-playlist', url]
    else:
        command = ['yt-dlp', '--dump-json', '--verbose', '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', url]

    # Get video information without downloading
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    videos_info = result.stdout.splitlines()

    for video_info in videos_info:
        video_json = json.loads(video_info)
        title = video_json.get('title', '')
        duration = video_json.get('duration', 0)  # Get the duration in seconds
        video_url = 'https://www.youtube.com/watch?v=' + video_json['id']

        # Check if the video should be downloaded based on title and duration
        if should_download(title, duration, filter_keywords):
            # Download command
            download_command = [
                'yt-dlp',
                '--format', 'bestaudio/best',
                '--extract-audio',
                '--audio-format', 'wav',
                '--audio-quality', '16',
                '--write-description',
                '--write-info-json',
                '--write-comments',
                '--ignore-errors',
                '--output', os.path.join(DATA_DIR, '%(id)s.%(ext)s'),
                video_url
            ]
            subprocess.run(download_command)


def write_metadata_to_csv():
    metadata_files = glob.glob(os.path.join(DATA_DIR, '*.info.json'))

    with open(os.path.join(DATA_DIR, 'metadata.csv'), 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'mark_sub_level', 'title', 'upload_date', 'duration', 'view_count', 'like_count', 'dislike_count', 'comment_count', 'description', 'webpage_url', 'channel_url', 'comments'])

        for metadata_file in metadata_files:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

                comments = ''
                if metadata.get('comments'):
                    try:
                        comments = [c['text'] for c in metadata.get('comments')]
                    except:
                        pass
                writer.writerow([
                    metadata.get('id'),
                    '',  # my custom field for my notes
                    metadata.get('title'), 
                    metadata.get('upload_date'), 
                    metadata.get('duration'),
                    metadata.get('view_count'), 
                    metadata.get('like_count'), 
                    metadata.get('dislike_count'),
                    metadata.get('comment_count'), 
                    metadata.get('description'),
                    metadata.get('webpage_url'),
                    metadata.get('channel_url'),
                    comments
                ])
    
    # make a copy of the metadata file
    category = DATA_DIR.split('/')[-1]
    os.system(f"cp {DATA_DIR}/metadata.csv {category}_metadata.csv")
    print(f"{category}_metadata.csv")



def cleanup_artifacts():
    # File extensions to delete
    extensions_to_delete = ['*.webp', '*.jpg', '*.json', '*.description']

    # Delete files with the specified extensions
    for extension in extensions_to_delete:
        for file in glob.glob(os.path.join(DATA_DIR, extension)):
            os.remove(file)




def delete_unreferenced_wav_files(metadata_path, audio_dir):
    # Load the updated metadata
    updated_metadata = pd.read_csv(metadata_path)
    referenced_files = list(updated_metadata['id'])
    referenced_files = set(['/import/c4dm-datasets/PianoJudge/novice/' + path + '.wav' for path in referenced_files])

    # Get all .wav files in the directory
    wav_files = set(glob.glob(os.path.join(audio_dir, '*.wav')))

    # Determine which .wav files are not referenced in the metadata
    unreferenced_files = wav_files - referenced_files

    # Delete the unreferenced .wav files
    for file_path in unreferenced_files:
        os.remove(file_path)
        print(f"Deleted: {file_path}")




if __name__ == "__main__":

    category = 'advanced'

    # DATA_DIR = "/import/c4dm-datasets/ICPC2015-dataset/data/raw/00_preliminary/wav/"
    # url_file = "/import/c4dm-datasets/ICPC2015-dataset/data/raw/00_preliminary/urls_all.list"
    DATA_DIR = f"/import/c4dm-datasets/PianoJudge/{category}/"
    url_file = f"/homes/hz009/Research/PianoJudge/data_collection/{category}_channels.txt"

    # delete_unreferenced_wav_files(DATA_DIR + "metadata.csv", DATA_DIR)
    # channel_url = 'https://www.youtube.com/@nixxpiano'

    # Define keywords to filter out, and download
    filter_keywords = ['concerto', 'duets']
    with open(url_file) as f:
        urls = f.readlines()
    for url in urls:
        if url[0] != '#': # not commented out
            download_channel_videos(url, filter_keywords)

    write_metadata_to_csv()
    cleanup_artifacts()

    
