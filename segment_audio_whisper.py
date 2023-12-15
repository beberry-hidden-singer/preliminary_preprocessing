#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: karljeon44
# Date: 12/14/23 7:07 PM
import argparse
import json
import logging
import os
import time

from pydub import AudioSegment

logger = logging.getLogger(__name__)

argparser = argparse.ArgumentParser()
argparser.add_argument('-a', '--audio', required=True, help='fpath to transcribed audio')
argparser.add_argument('-w', '--whisper', required=True, help='fpath to whisper transcription json')
argparser.add_argument('-o', '--output', required=True, help='path to output dir')
argparser.add_argument('-sr', '--sample_rate', type=int, default=16000, help='sample rate')

def main():
  # 0. setup
  begin = time.time()

  args = argparser.parse_args()
  assert os.path.exists(args.audio)
  assert os.path.exists(args.whisper)
  os.makedirs(args.output, exist_ok=True)

  with open(args.whisper) as f:
    whisper = json.load(f)

  audio = AudioSegment.from_wav(args.audio)
  audio = audio.set_frame_rate(args.sample_rate)

  metadata = list()
  metadata_fpath = os.path.join(args.output, 'metadata.txt')

  segment_fpath_template = os.path.join(args.output, f"{os.path.splitext(os.path.basename(args.audio))[0]}_%d.wav")
  for i, segment in enumerate(whisper['segments']):
    start, end, text = segment['start'], segment['end'], segment['text']
    text = text.strip()
    if len(text) == 0:
      continue

    segment_fpath = segment_fpath_template %i
    audio_segment = audio[start*1000:end*1000]
    audio_segment.export(segment_fpath, format="wav")

    # add to metadata
    """audio1|This is my sentence.|This is my sentence.
       audio2|1469 and 1470|fourteen sixty-nine and fourteen seventy
       audio3|It'll be $16 sir.|It'll be sixteen dollars sir."""

    metadata.append("|".join([
      os.path.splitext(os.path.basename(segment_fpath))[0], text
    ]))

  with open(metadata_fpath, 'w') as f:
    f.write('\n'.join(metadata))

  duration = time.time() - begin
  print(f"Execution Time: {int(duration//60)}m {duration%60:.2f}s")


if __name__ == '__main__':
  main()
