
| ![GIF](https://drive.google.com/thumbnail?id=1t_jAaTgN4nYujkNR-KLLbyXP4BF9NGBX&sz=h480) | ![GIF](https://drive.google.com/thumbnail?id=16HzxSyK7SL4ZYO_MVJ4z_Shg6G2DhMlD&sz=h480) |
|:---:|:---:|
# speaker-enhance

This is a Sieve pipeline that enhances a talking head video by:
* Enhancing and cleaning up the audio with the [audio-enhance](https://sievedata.com/functions/sieve/audio-enhance) Sieve function.
* Replacing/blurring the background with the [background-removal](https://sievedata.com/functions/sieve/background-removal) Sieve function.
* Correcting eye contact with the [eye-contact-correction](https://sievedata.com/functions/sieve/eye-contact-correction) Sieve function.

The pipeline completes jobs quickly by running each step (audio enhancement, background removal, eye contact correction) in parallel.

You can try it here: [https://sievedata.com/functions/sieve/speaker-enhance](https://sievedata.com/functions/sieve/speaker-enhance)

Or see [Calling `speaker-enhance` via the Sieve SDK](#calling-speaker-enhance-via-the-sieve-sdk) to learn how to call the function via the Sieve Python SDK.

## Options

* `background_img`: A background image to use for the background replacement. Overrides `background_color_rgb`.
* `background_color_rgb`: A comma-separated string representing the RGB color to use for the background replacement.
* `blur_background`: If true, blurs the background.
    * If `background_img` is provided, blurs the background image.
    * Otherwise, blurs the background of the input video.
* `blur_strength`: Larger values make the background blurrier. A value of 0 means no blurring. Defaults to 19.

## Calling `speaker-enhance` via the Sieve SDK
You can install `sieve` via pip with `pip install sievedata`.
Be sure to set `SIEVE_API_KEY` to your Sieve API key. 
You can find your API key at [https://www.sievedata.com/dashboard/settings](https://www.sievedata.com/dashboard/settings).

```python
import sieve

# get the speaker-enhance function
speaker_enhance = sieve.function.get("sieve/speaker-enhance")

# get input video, background image, and options
video = sieve.File("path/to/video.mp4")
background_img = sieve.File("path/to/background.png")
blur_background = True

# create a corrected video with the new background
out = speaker_enhance.run(video, background_img=background_img)

# create a corrected video with the original background blurred
out = speaker_enhance.run(video, blur_background=True)

# create a corrected video with a custom background color
out = speaker_enhance.run(video, background_color_rgb="255,255,255")
```
