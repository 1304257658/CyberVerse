package mediapeer

import "testing"

func TestVP8EncodeParamsDefaultsWhenUnset(t *testing.T) {
	_, _, deadline, cpu := resolveVP8Encode(0, VP8EncodeParams{})
	if deadline != "realtime" || cpu != 8 {
		t.Fatalf("defaults deadline=%q cpu=%d", deadline, cpu)
	}
}

func TestVP8EncodeParamsCustom(t *testing.T) {
	bitrate, fallback, deadline, cpu := resolveVP8Encode(0, VP8EncodeParams{
		Deadline:            "good",
		CPUUsed:             4,
		FallbackBitrateKbps: 1500,
	})
	if bitrate != 1500 || fallback != 1500 || deadline != "good" || cpu != 4 {
		t.Fatalf("got bitrate=%d fallback=%d deadline=%q cpu=%d", bitrate, fallback, deadline, cpu)
	}
}