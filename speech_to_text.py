# https://stackoverflow.com/questions/32005310/speech-recognition-python-code-not-working
# pip install SpeechRecognition
# pip install PyAudio


import time

import speech_recognition as sr


def recognize_speech_from_mic(recognizer, microphone):
    """Transcribe speech from recorded from `microphone`.

    Returns a dictionary with three keys:
    "success": a boolean indicating whether or not the API request was
           successful
    "error":   `None` if no error occured, otherwise a string containing
           an error message if the API could not be reached or
           speech was unrecognizable
    "transcription": `None` if speech could not be transcribed,
           otherwise a string containing the transcribed text
    """
    # check that recognizer and microphone arguments are appropriate type
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")

    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")

    # adjust the recognizer sensitivity to ambient noise and record audio
    # from the microphone
    with microphone as source:
    	print('Say Something')
    	recognizer.adjust_for_ambient_noise(source)
    	audio = recognizer.listen(source)

    # set up the response object
    response = {
        "success": True,
        "error": None,
        "transcription": None
    }

    try:
        response["transcription"] = recognizer.recognize_google(audio)
    except sr.RequestError:
        # API was unreachable or unresponsive
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        # speech was unintelligible
        response["error"] = "Unable to recognize speech"

    return response['transcription']


if __name__ == "__main__":
	# Test Here
	recognizer = sr.Recognizer()
	microphone = sr.Microphone()
	while True:
		print('You Said :-',recognize_speech_from_mic(recognizer, microphone))
