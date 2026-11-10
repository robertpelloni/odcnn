import pickle
import numpy as np
from scipy.signal import argrelmax
from librosa.util import peak_pick
from librosa.onset import onset_detect
from music_processor import *
from chart_exporters import export_osu, export_sm, export_beat_saber, export_midi
from roll_detector import detect_sustained_rolls, merge_onsets_and_rolls

def detection(don_inference, ka_inference, song):
    """detects notes disnotesiresultg don and ka"""

    don_inference = smooth(don_inference, 5)
    ka_inference = smooth(ka_inference, 5)

    don_timestamp = (peak_pick(don_inference, pre_max=1, post_max=2, pre_avg=4, post_avg=5, delta=0.05, wait=3)+7)  # 実際は7フレーム目のところの音
    ka_timestamp = (peak_pick(ka_inference, pre_max=1, post_max=2, pre_avg=4, post_avg=5, delta=0.05, wait=3)+7)
    
    print(don_timestamp)
    print(ka_timestamp)

    song.don_timestamp = don_timestamp[np.where(don_inference[don_timestamp] > ka_inference[don_timestamp])]
    song.timestamp = song.don_timestamp*512/song.samplerate
    # print(len(song.timestamp))
    song.synthesize(diff='don')

    # song.ka_timestamp = song.don_timestamp
    song.ka_timestamp = ka_timestamp[np.where(ka_inference[ka_timestamp] > don_inference[ka_timestamp])]
    song.timestamp=song.ka_timestamp*512/song.samplerate
    # print(len(song.timestamp))
    song.synthesize(diff='ka')

    song.save("./data/inference/inferred_music.wav")


def create_tja(filename, song, don_timestamp, ka_timestamp=None, rolls=None):

    if rolls is None:
        rolls = []

    if ka_timestamp is None:
        timestamp = don_timestamp * 512 / song.samplerate
        with open(filename, "w") as f:
            f.write('TITLE: xxx\nSUBTITLE: --\nBPM: 240\nWAVE:xxx.ogg\nOFFSET:0\n#START\n')
            i = 0
            time = 0
            while(i < len(timestamp)):
                if time / 100 >= timestamp[i]:
                    f.write('1')
                    i += 1
                else:
                    f.write('0')
                if time % 100 == 99:
                    f.write(',\n')
                time += 1
            f.write('#END')

    else:
        don_ticks = np.rint(don_timestamp * 512 / song.samplerate * 100).astype(np.int32)
        ka_ticks = np.rint(ka_timestamp * 512 / song.samplerate * 100).astype(np.int32)

        # Build roll start and end tick maps
        roll_starts = {}
        roll_ends = set()
        for r in rolls:
            s_tick = int(round(r["start_sec"] * 100))
            e_tick = int(round(r["end_sec"] * 100))
            roll_starts[s_tick] = str(r.get("type", 5))
            roll_ends.add(e_tick)

        max_tick = 0
        if len(don_ticks) > 0:
            max_tick = max(max_tick, int(don_ticks[-1]))
        if len(ka_ticks) > 0:
            max_tick = max(max_tick, int(ka_ticks[-1]))
        if len(roll_ends) > 0:
            max_tick = max(max_tick, max(roll_ends))

        with open(filename, "w") as f:
            f.write('TITLE: xxx\nSUBTITLE: --\nBPM: 240\nWAVE:xxx.ogg\nOFFSET:0\n#START\n')
            for time in range(max_tick + 1):
                if time in roll_starts:
                    f.write(roll_starts[time])
                elif time in roll_ends:
                    f.write('8')
                elif np.isin(time, don_ticks):
                    f.write('1')
                elif np.isin(time, ka_ticks):
                    f.write('2')
                else:
                    f.write('0')
                if time % 100 == 99:
                    f.write(',\n')
            f.write('#END')


if __name__ == "__main__":
    
    with open('./data/pickles/test_data.pickle', mode='rb') as f:
        song = pickle.load(f)

    with open('./data/pickles/don_inference.pickle', mode='rb') as f:
        don_inference = pickle.load(f)

    with open('./data/pickles/ka_inference.pickle', mode='rb') as f:
        ka_inference = pickle.load(f)

    detection(don_inference, ka_inference, song)
    # Detect sustained rolls and merge with discrete onsets
    rolls = []
    if hasattr(song, "data") and len(song.data) > 0:
        rolls = detect_sustained_rolls(song.data, sr=song.samplerate)
        if rolls:
            don_sec = song.don_timestamp * 512 / song.samplerate
            ka_sec = song.ka_timestamp * 512 / song.samplerate
            p_don_sec, p_ka_sec = merge_onsets_and_rolls(don_sec, ka_sec, rolls)
            song.don_timestamp = (p_don_sec * song.samplerate / 512).astype(np.int64)
            song.ka_timestamp = (p_ka_sec * song.samplerate / 512).astype(np.int64)

    create_tja("./data/inference/inferred_notes.tja", song, song.don_timestamp, song.ka_timestamp, rolls=rolls)
    export_osu("./data/inference/inferred_notes.osu", song, song.don_timestamp, song.ka_timestamp, rolls=rolls)
    export_sm("./data/inference/inferred_notes.sm", song, song.don_timestamp, song.ka_timestamp)
    export_beat_saber("./data/inference/inferred_notes.dat", song, song.don_timestamp, song.ka_timestamp)
    export_midi("./data/inference/inferred_notes.mid", song, song.don_timestamp, song.ka_timestamp)

