# # word-level
# ffmpeg -i which_people_make_these_words_blank.mp4 -vf "subtitles=which_people_make_these_words.srt:force_style='Fontsize=24'" -c:a copy which_people_make_these_words.mp4
# character-level
wav=generated_speech/Which-people-make-strong-these-strong-words-170341.wav
srt=generated_speech/Which-people-make-strong-these-strong-words-170341.srt
ffmpeg -i $wav -f lavfi -i color=c=black:s=1920x1080 -shortest -c:v libx264 -c:a aac -b:a 192k -pix_fmt yuv420p which_people_make_these_words_blank.mp4
ffmpeg -i which_people_make_these_words_blank.mp4 -vf "subtitles=$srt:force_style='Fontsize=24'" -c:a copy which_people_make_these_words.mp4
rm which_people_make_these_words_blank.mp4
