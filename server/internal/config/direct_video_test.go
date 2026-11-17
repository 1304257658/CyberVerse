package config

import "testing"

func TestDirectVideoDefaultsMatchOfficialHardcoded(t *testing.T) {
	cfg, err := Load(writeTestConfig(t, "pipeline: {}\n"))
	if err != nil {
		t.Fatal(err)
	}
	official := OfficialDirectVideoDefaults()
	got := cfg.Pipeline.DirectVideo
	if got != official {
		t.Fatalf("official defaults mismatch:\n got=%+v\nwant=%+v", got, official)
	}
	if cfg.Recording.CRF != 23 {
		t.Fatalf("recording crf=%d want 23", cfg.Recording.CRF)
	}
}

func TestDirectVideoCustomValuesPreserved(t *testing.T) {
	cfg, err := Load(writeTestConfig(t, `
pipeline:
  direct_video:
    max_bitrate_kbps: 4000
    gcc_safety_percent: 90
    vp8_deadline: good
    vp8_cpu_used: 4
    gcc_max_bitrate_bps: 8000000
recording:
  crf: 18
`))
	if err != nil {
		t.Fatal(err)
	}
	dv := cfg.Pipeline.DirectVideo
	if dv.MaxBitrateKbps != 4000 || dv.GCCSafetyPercent != 90 {
		t.Fatalf("unexpected bitrate caps: %+v", dv)
	}
	if dv.VP8Deadline != "good" || dv.VP8CPUUsed != 4 {
		t.Fatalf("unexpected vp8 encode: %+v", dv)
	}
	if dv.GCCMaxBitrateBps != 8_000_000 {
		t.Fatalf("gcc max=%d", dv.GCCMaxBitrateBps)
	}
	if cfg.Recording.CRF != 18 {
		t.Fatalf("crf=%d", cfg.Recording.CRF)
	}
}