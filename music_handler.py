import os
import regex as re
import numpy as np

cwd = os.path.dirname(__file__)

def extract_song_snippet(text):
    pattern = '(^|\n\n)(.*?)\n\n'
    search_results = re.findall(pattern, text, overlapped=True, flags=re.DOTALL)
    songs = [song[1] for song in search_results]
    print("Found {} songs in text".format(len(songs)))
    return songs

def save_song_to_abc(song, filename="tmp"):
    save_name = "{}.abc".format(filename)
    save_path = os.path.join(cwd, 'music_file', save_name)
    with open(save_path, "w") as f:
        f.write(song)
    return filename

def abc2wav(abc_file):
    path_to_tool = os.path.join(cwd, 'bin', 'abc2wav')
    print(path_to_tool)
    cmd = "{} {}".format(path_to_tool, abc_file)
    return os.system(cmd)

def abc2midi(abc_file, remove_abc=True):
    path_to_tool = os.path.join(cwd, 'bin', 'abc2midi.exe')
    input_file_path = os.path.join('music_file', abc_file)
    output_file_path = os.path.join('music_file', abc_file[:-4] + ".mid")
    cmd = "{} {} -o {}".format(path_to_tool, input_file_path, output_file_path)
    ret = os.system(cmd)
    if remove_abc:
        os.remove(input_file_path)
    return ret

def play_song(song):
    basename = save_song_to_abc(song)
    ret = abc2midi(basename+'.abc')
    # ret = abc2wav(basename+'.abc')
    # if ret == 0: #did not suceed
    #     return play_wav(basename+'.wav')
    # return None

def play_generated_song(generated_text):
    songs = extract_song_snippet(generated_text)
    if len(songs) == 0:
        print("No valid songs found in generated text. Try training the \
            model longer or increasing the amount of generated music to \
            ensure complete songs are generated!")

    for song in songs:
        play_song(song)
    print("None of the songs were valid, try training longer to improve \
        syntax.")
