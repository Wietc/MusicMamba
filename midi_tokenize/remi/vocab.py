import pickle
import numpy as np
import miditoolkit
import os
import math
from miditoolkit.midi import parser as mid_parser  
from miditoolkit.midi import containers as ct
import torch

from .constants import (
    DEFAULT_DURATION_BINS,
    DEFAULT_PITCH_BINS,
    DEFAULT_POS_PER_QUARTER,
    DEFAULT_POSITION_BINS,
    DEFAULT_TEMPO_BINS,
    DEFAULT_VELOCITY_BINS,
    MODE_PATTERN
)

class Vocab(object):
    def __init__(self):
        ''' 
        initialize the REMI vocabulary settings
        '''

        # split each beat into 4 subbeats
        self.q_beat = DEFAULT_POS_PER_QUARTER

        # dictionary for matching token ID to name and the other way around.
        self.token2id = {}
        self.id2token = {}

        # midi pitch number : 1 ~ 127 (highest pitch) 
        self._pitch_bins = DEFAULT_PITCH_BINS

        # duration tokens 1~64 of self.q_beat
        self._duration_bins = DEFAULT_DURATION_BINS

        # velocity tokens 1~127
        self._velocity_bins =  DEFAULT_VELOCITY_BINS

        # tempo tokens 17~250 (the min of the tempo and the max of the tempo) #################################################################
        self._tempo_bins = DEFAULT_TEMPO_BINS

        # position(subbeat) tokens 0~15, indicate the relative position with in a bar
        self._position_bins = DEFAULT_POSITION_BINS

        # mode pattern can detecet the existance of mode and the type of mode
        self._mode_pattern = MODE_PATTERN

        self.n_tokens = 0

        self.token_type_base = {}

        self.build_vocab()

        # Event including:
        # Note-On (129) : 0 (padding) ,1 ~ 127(highest pitch) , 128 (rest)
        # Note-Duration : 1 ~ 16 beat * 3
        # min resulution 1/12 notes

    def build_vocab(self):
        self.token2id = {}
        self.id2token = {}

        self.n_tokens = 0

        self.token2id['padding'] = 0
        self.n_tokens += 1

        self.token2id['masking'] = 1
        self.n_tokens += 1

        # Create Note-on tokens:
        # Note-on
        self.token_type_base = {'Note-On' : 2}
        for note_type in ['Normal', 'Mode']:
            for i in self._pitch_bins:
                self.token2id[ 'Note-On-{}_{}'.format(note_type, i) ] = self.n_tokens
                self.n_tokens += 1

        # Create Note-Duration tokens
        # Note-Duration
        self.token_type_base['Note-Duration'] = self.n_tokens
        for duration in self._duration_bins:
            self.token2id[ 'Note-Duration_{}'.format(duration) ] = self.n_tokens
            self.n_tokens += 1
        
        # Create Note-Velocity tokens
        # Note-Velocity
        self.token_type_base['Note-Velocity'] = self.n_tokens
        for vel in self._velocity_bins:
            self.token2id[ 'Note-Velocity_{}'.format(vel) ] = self.n_tokens
            self.n_tokens += 1

        # Tempo
        self.token_type_base['Tempo'] = self.n_tokens
        for tmp in self._tempo_bins:
            self.token2id[ 'Tempo_{}'.format(tmp) ] = self.n_tokens
            self.n_tokens += 1

        # Positions
        self.token_type_base['Position'] = self.n_tokens
        for pos in self._position_bins:
            self.token2id[ 'Position_{}'.format(pos) ] = self.n_tokens
            self.n_tokens += 1

        # Bar
        self.token_type_base['Bar'] = self.n_tokens
        self.token2id[ 'Bar' ] = self.n_tokens
        self.n_tokens += 1

 

        ''' Mode related tokens '''
        self.token_type_base['Mode'] = self.n_tokens
        for type_idx in range(len(self._mode_pattern)):
            self.token2id['Mode_Start_{}'.format(type_idx)] = self.n_tokens
            self.n_tokens += 1
        self.token2id['Mode_End'] = self.n_tokens
        self.n_tokens += 1
            

        for w , v in self.token2id.items():
            self.id2token[v] = w
        
        self.n_tokens = len(self.token2id)

    def detect_mode(self, bar_objs, start_tick):
        notes = [o["obj"] for o in bar_objs if "Note" in o["obj_type"]]
        for mode_idx, intervals in enumerate(self._mode_pattern):
            mode_notes = [notes[0].pitch]
            current_tick = start_tick
            for interval in intervals:
                next_note = next((note for note in notes if note.pitch == mode_notes[-1] + interval and note.start >= current_tick), None)
                if next_note is None:
                    break
                mode_notes.append(next_note.pitch)
                current_tick = next_note.start
            if len(mode_notes) == len(intervals) + 1:
                return mode_idx, current_tick
        return None, 0

    def midi2REMI(self, midi_path, quantize=True, Mode_label=False, verbose=False, bar_first=False):
        '''convert MIDI to tokens representation'''
        MIN_MEL_NOTES = 8
        midi_obj = mid_parser.MidiFile(midi_path)
        # calculate the min step (in ticks) for REMI representation
        min_step = midi_obj.ticks_per_beat * 4 / 16
         
        # quantize
        if quantize:
            for i in range(len(midi_obj.instruments)):
                for n in range(len(midi_obj.instruments[i].notes)):
                    midi_obj.instruments[i].notes[n].start = int(int(midi_obj.instruments[i].notes[n].start / min_step) * min_step)           
                    midi_obj.instruments[i].notes[n].end = int(int(midi_obj.instruments[i].notes[n].end / min_step) * min_step)

        
        if Mode_label:
            Mode_info_track = list(filter(lambda x: x.name=="Mode track",midi_obj.instruments))
            assert len(Mode_info_track) == 1
        
        event_items = []
        midi_obj.instruments[0].notes = sorted(midi_obj.instruments[0].notes,key=lambda x: x.start)
        # extrack Melody Notes
        melody_start = sorted(midi_obj.instruments[0].notes,key=lambda x: x.start)[0].start
        melody_end = sorted(midi_obj.instruments[0].notes,key=lambda x: x.start)[-1].end
        melody_notes = midi_obj.instruments[0].notes
        # add Notes Event
        for t in melody_notes:
            event_items.append({
                                "priority" : 1,
                                "priority_1" : t.pitch,
                                "start_tick" : t.start,
                                "obj_type" : "Note-{}".format(midi_obj.instruments[0].name),
                                "obj" : t
                            })
        # add Tempos Event
        for t in midi_obj.tempo_changes:
            event_items.append({
                            "priority" : 0,
                            "priority_1" : 0,
                            "start_tick" : t.time,
                            "obj_type" : "Tempo",
                            "obj" : t
                        })
        event_items = sorted(event_items,key=lambda x: (x["start_tick"],x["priority"],-x["priority_1"]))

        if Mode_label:
            Mode_info_track = Mode_info_track[0]
            Mode_notes = Mode_info_track.notes
        

        # b_i = 0
        in_mode = False
        mode_end_tick = 0
        # based on Bar
        bar_group = []
        bar_ticks = midi_obj.ticks_per_beat * 4
        if verbose:
            print("Bar tick length: {}".format(bar_ticks))

        for bar_start_tick in range(0,event_items[-1]["start_tick"],bar_ticks):
            if verbose:
                print("Bar {} at tick: {}".format(bar_start_tick // bar_ticks,bar_start_tick))
            bar_end_tick = bar_start_tick + bar_ticks
            current_bar = []
            bar_objs = list(filter(lambda x: x["start_tick"] >=bar_start_tick and x["start_tick"]< bar_end_tick,event_items))
            bar_objs.insert(0,{"start_tick":-1})

            current_bar.append("Bar")

            
            for i,obj in enumerate(bar_objs):
                if obj["start_tick"]==-1 : continue
                if not obj["start_tick"] == bar_objs[i-1]["start_tick"]:
                    # pos events
                    pos = (obj["start_tick"] - bar_start_tick) / midi_obj.ticks_per_beat * self.q_beat
                    pos_index = np.argmin(abs(pos - self._position_bins)) # use the closest position
                    pos = self._position_bins[pos_index]
                    current_bar.append("Position_{}".format(pos))

                if obj["obj_type"].startswith("Note"):
                    # track_name = obj["obj_type"].split('-')[1].upper()
                    if not Mode_label:
                        # add pitch and type
                        note_type = 'Normal'
                        current_bar.append("Note-On-{}_{}".format(note_type, obj["obj"].pitch))
                    else:
                        if not in_mode and obj['obj'] in Mode_notes:
                            note_type = 'Mode'
                            mode_idx, mode_end_tick = self.detect_mode(bar_objs[i:], obj["start_tick"])
                            if mode_idx:
                                in_mode = True
                                current_bar.append('Mode_Start_{}'.format(mode_idx))
                                if verbose:
                                    print("Mode start at {}".format(obj["start_tick"]))
                            current_bar.append("Note-On-{}_{}".format(note_type, obj["obj"].pitch))
                        elif in_mode and obj['obj'] in Mode_notes:
                            note_type = 'Mode'
                            current_bar.append("Note-On-{}_{}".format(note_type, obj["obj"].pitch))
                            # if obj["start_tick"] >= mode_end_tick:
                            #     in_Mode = False
                            #     current_bar.append('Mode_End')
                        else:
                            note_type = 'Normal'
                            current_bar.append("Note-On-{}_{}".format(note_type, obj["obj"].pitch))
                        
                    # add duration
                    dur = (obj["obj"].end - obj["obj"].start) / midi_obj.ticks_per_beat * self.q_beat
                    dur_index = np.argmin(abs(dur - self._duration_bins)) # use the closest position
                    dur = self._duration_bins[dur_index]
                    current_bar.append("Note-Duration_{}".format(dur))
                    # add velocity
                    vel_index = np.argmin(abs(obj["obj"].velocity - self._velocity_bins)) # use the closest position
                    vel = self._velocity_bins[vel_index]
                    current_bar.append("Note-Velocity_{}".format(vel))
                    if in_mode and obj['obj'] in Mode_notes and obj["start_tick"] >= mode_end_tick:
                        in_mode = False
                        current_bar.append('Mode_End')
                        if verbose:
                            print("Mode end at {}".format(obj["start_tick"]))
                elif obj["obj_type"].startswith("Tempo"):
                    # add tempo
                    tmp_index = np.argmin(abs(obj["obj"].tempo - self._tempo_bins)) # use the closest position
                    tmp = self._tempo_bins[tmp_index]
                    current_bar.append(obj["obj_type"] + "_{}".format(tmp))
                else:
                    current_bar.append(obj["obj_type"])
            bar_group.extend(current_bar)

        output_ids = [self.token2id[x] for x in bar_group]

        return output_ids

    def processREMI(self, event_ids):
        # to generate the mask matrix of Mode?
        # just slicing the token sequence?
        # return src and tgt tokens?
        #### still not use it

        src_mode_mask = []
        in_mode = False
        for idx in event_ids:
            if self.id2token[idx].startswith("Mode_Start"):
                in_mode = True
            elif self.id2token[idx].startswith("Mode_End"):
                in_mode = False
            src_mode_mask.append(int(in_mode))

            
        for i in range(1, len(src_mode_mask)):
            src_mode_mask[i] = src_mode_mask[i-1]*src_mode_mask[i] + src_mode_mask[i]

        return {
            "remi_tokens": event_ids, 
            "mode_mask":src_mode_mask
            }

    def REMI2MIDI(self, event_ids, save_path, verbose=False):
        # create empty midi obj
        new_mido_obj = mid_parser.MidiFile()
        # set default bpm
        new_mido_obj.ticks_per_beat = 120

        # create tracks
        music_tracks = {}
        music_tracks["Melody"] = ct.Instrument(program=0, is_drum=False, name='Melody')
        music_tracks["Mode track"] = ct.Instrument(program=0, is_drum=False, name='Mode track')
 

        # all our generated music are 4/4
        new_mido_obj.time_signature_changes.append(miditoolkit.TimeSignature(4,4,0))

        ticks_per_step = new_mido_obj.ticks_per_beat / self.q_beat

        # convert tokens from id to string
        events = []
        for x in event_ids:
            events.append(self.id2token[x])
        
        # parsing tokens
        last_tick = 0
        current_bar_anchor = 0
        current_theme_boundary = []
        motif_label_segs = []
        idx = 0
        first_bar = True
        while(idx < len(events)):
            if events[idx] == "Bar":
                if first_bar:
                    current_bar_anchor = 0
                    first_bar = False
                else:
                    current_bar_anchor += new_mido_obj.ticks_per_beat * 4
                idx += 1
            elif events[idx].startswith("Position"):
                pos = int(events[idx].split('_')[1])
                last_tick = pos * ticks_per_step + current_bar_anchor
                idx += 1
            elif events[idx].startswith("Tempo"):
                tmp = pos = int(events[idx].split('_')[1])
                new_mido_obj.tempo_changes.append(ct.TempoChange(
                    tempo=int(tmp),
                    time=int(last_tick)
                ))
                idx += 1
            elif events[idx].startswith("Note"):
                # print(events[idx], events[idx+1], events[idx+2])
                if events[idx].startswith("Note-On"):
                    if idx+2 < len(events) and events[idx+1].startswith("Note-Duration") and events[idx+2].startswith("Note-Velocity"):
                        # assert events[idx].startswith("Note-On")
                        # assert events[idx+1].startswith("Note-Duration")
                        # assert events[idx+2].startswith("Note-Velocity")
                        
                        note_type = events[idx].split("_")[0].split("-")[2]
                        
                        new_note = miditoolkit.Note(
                                velocity=int(events[idx+2].split("_")[1]),
                                pitch=int(events[idx].split("_")[1]),
                                start=int(last_tick),
                                end=int(int(events[idx+1].split('_')[1]) * ticks_per_step) + int(last_tick)
                            )
                        if note_type == 'Mode':
                            music_tracks['Mode track'].notes.append(new_note)
                        music_tracks['Melody'].notes.append(new_note)
                        idx += 3
                    else:
                        idx += 1
                else:
                    idx += 1
                
            elif events[idx].startswith("Mode_Start"):
                if verbose:
                    print(f'Mode start at {last_tick}')
                idx += 1
            elif events[idx].startswith("Mode_End"):
                if verbose:
                    print(f'Mode end at {last_tick}')
                idx += 1
        # add tracks to midi file
        new_mido_obj.instruments.extend([music_tracks[ins] for ins in music_tracks])

        # if verbose:
        #     print("Saving midi to ({})".format(save_path))
        print("Saving midi to ({})".format(save_path))

        # save to disk
        new_mido_obj.dump(save_path)


# remi_vocab = Vocab()
# print(remi_vocab.n_tokens)
# # print(remi_vocab.token2id)
# # print(remi_vocab.token_type_base)
#output_ids = remi_vocab.midi2REMI('/home/jiatao/program/MelodyGLM/MDP/data/melody_tonal_downbeat_Tmode/Chinese Music/train/115_01_Tonal_downbeats_with_modes.mid', Mode_label=True,verbose=True)
# print(len(remi_vocab.processREMI(output_ids)))
# remi_vocab.REMI2MIDI(output_ids, "/home/jiatao/program/Mamba/mode_midi/310_Generated.mid", verbose=True)

### origin REMI Event###
## Bar, Position_i, [Tempo_i], Note-on_i, Note_Duration_i, Note-Velocity_i, Note-on_i+1...... ###