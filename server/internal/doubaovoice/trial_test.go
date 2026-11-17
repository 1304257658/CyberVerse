package doubaovoice

import "testing"

func TestPickPlayableTrialURLPrefersShortURL(t *testing.T) {
	info := TrialInfo{
		TrialURL:      "https://lf3-static.bytednsdoc.com/demo-long.mp3",
		ShortTrialURL: "https://lf3-static.bytednsdoc.com/demo-short.mp3",
	}
	got, ok := PickPlayableTrialURL(info)
	if !ok || got != info.ShortTrialURL {
		t.Fatalf("expected short trial url, got %q ok=%v", got, ok)
	}
}

func TestPickPlayableTrialURLRejectsPlaceholderAndSynthesis(t *testing.T) {
	for _, info := range []TrialInfo{
		{TrialURL: "https://www.volcengine.com/"},
		{TrialURL: "https://openspeech.bytedance.com/api/v3/tts/bidirection"},
		{TrialURL: "http://insecure.example.com/a.mp3"},
	} {
		if _, ok := PickPlayableTrialURL(info); ok {
			t.Fatalf("expected reject %+v", info)
		}
	}
}