import subprocess
import os
import json
import csv
import glob
import pandas as pd
import hook


def should_download(title, filter_keywords):
    """ Check if the video should be downloaded based on the title. """
    for keyword in filter_keywords:
        if keyword.lower() in title.lower():
            return False
    return True

def download_channel_videos(channel_url, filter_keywords):
    # Define the yt-dlp command with necessary options
    command = [
        'yt-dlp',
        '--dump-json',
        '--flat-playlist',
        channel_url
    ]

    # Get video information without downloading
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    videos_info = result.stdout.splitlines()

    for video_info in videos_info:
        video_json = json.loads(video_info)
        title = video_json.get('title', '')
        video_url = video_json.get('url', '')

        # Check if the video should be downloaded
        if should_download(title, filter_keywords):
            # Download the video if it passes the filter
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

    with open(os.path.join(DATA_DIR, 'metadata.csv'), 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'title', 'upload_date', 'duration', 'view_count', 'like_count', 'dislike_count', 'comment_count', 'description', 'webpage_url', 'channel_url', 'comments'])

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
    referenced_files = set(updated_metadata['path'])

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

    DATA_DIR = f"/import/c4dm-datasets/PianoJudge/{category}"

    # channel_url = 'https://www.youtube.com/@nixxpiano'

    # Define keywords to filter out
    filter_keywords = ['concerto']

    with open(f"{category}_channels.txt") as f:
        urls = f.readlines()

    # for url in urls:
    #     download_channel_videos(url, filter_keywords)

    write_metadata_to_csv()
    cleanup_artifacts()

    # delete_unreferenced_wav_files(DATA_DIR + "metadata.csv", DATA_DIR)
