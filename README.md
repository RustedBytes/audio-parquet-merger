# Audio Parquet Merger

## Data Viewer audio on HF

Insert the following header to `README.md` file on Hugging Face if you want to see audio HTML tag to listen to audios in the Data Viewer.

```
---
dataset_info:
  features:
  - name: audio_1
    dtype: audio
  - name: audio_2
    dtype: audio
  - name: duration
    dtype: float64
  - name: transcription
    dtype: string
tags:
  - audio
  - speech-processing
---
```

PS: maybe I'll rewrite it in Rust in the future. For now, it's okay for my scientific research...
