package recording

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func argsContainPair(args []string, key, val string) bool {
	for i := 0; i < len(args)-1; i++ {
		if args[i] == key && args[i+1] == val {
			return true
		}
	}
	return false
}

func TestFfmpegLibx264EncodeArgsDisableBFramesAndCFR(t *testing.T) {
	args := ffmpegLibx264EncodeArgs(23)
	if !argsContainPair(args, "-bf", "0") {
		t.Fatalf("expected -bf 0 in libx264 args, got %v", args)
	}
	if !argsContainPair(args, "-fps_mode", "cfr") {
		t.Fatalf("expected -fps_mode cfr in libx264 args, got %v", args)
	}
	if !slices.Contains(args, "libx264") {
		t.Fatalf("expected libx264 encoder in args, got %v", args)
	}
}

func TestFfmpegRawRGB24InputArgsGenPTS(t *testing.T) {
	args := ffmpegRawRGB24InputArgs("64x64", 25)
	if !argsContainPair(args, "-fflags", "+genpts") {
		t.Fatalf("expected -fflags +genpts on raw input, got %v", args)
	}
}

func TestFfmpegFinishMuxArgsShortest(t *testing.T) {
	args := ffmpegFinishMuxArgs("/tmp/v.mp4", "/tmp/a.wav", "/tmp/out.mp4")
	if !slices.Contains(args, "-shortest") {
		t.Fatalf("expected -shortest in mux args, got %v", args)
	}
	if !argsContainPair(args, "-movflags", "+faststart") {
		t.Fatalf("expected +faststart in mux args, got %v", args)
	}
}

func TestEncodeRGB24SyntheticNoBFramesAndTimestamps(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not in PATH")
	}
	if _, err := exec.LookPath("ffprobe"); err != nil {
		t.Skip("ffprobe not in PATH")
	}

	const (
		width  = 64
		height = 64
		fps    = 25
		frames = 10
	)
	frameBytes := width * height * 3
	var rgbChunks [][]byte
	for f := 0; f < frames; f++ {
		chunk := make([]byte, frameBytes)
		val := byte(f * 20)
		for i := 0; i < frameBytes; i += 3 {
			chunk[i] = val
			chunk[i+1] = 255 - val
			chunk[i+2] = 128
		}
		rgbChunks = append(rgbChunks, chunk)
	}

	outPath := filepath.Join(t.TempDir(), "synthetic.mp4")
	if err := EncodeRGB24ToMP4(outPath, width, height, fps, rgbChunks, nil, 0, 28); err != nil {
		t.Fatalf("EncodeRGB24ToMP4: %v", err)
	}

	probeOut, err := exec.Command("ffprobe",
		"-hide_banner",
		"-loglevel", "error",
		"-select_streams", "v:0",
		"-show_frames",
		"-show_entries", "frame=pict_type,pts_time,dts_time,pkt_pts_time,pkt_dts_time",
		"-of", "json",
		outPath,
	).CombinedOutput()
	if err != nil {
		t.Fatalf("ffprobe: %v\n%s", err, probeOut)
	}

	var parsed struct {
		Frames []struct {
			PictType   string `json:"pict_type"`
			PTSTime    string `json:"pts_time"`
			DTSTime    string `json:"dts_time"`
			PktPTSTime string `json:"pkt_pts_time"`
			PktDTSTime string `json:"pkt_dts_time"`
		} `json:"frames"`
	}
	if err := json.Unmarshal(probeOut, &parsed); err != nil {
		t.Fatalf("parse ffprobe json: %v", err)
	}
	if len(parsed.Frames) == 0 {
		t.Fatal("ffprobe returned no video frames")
	}
	for i, fr := range parsed.Frames {
		if fr.PictType == "B" {
			t.Fatalf("frame %d is B-frame (pict_type=%s)", i, fr.PictType)
		}
		pts := fr.PTSTime
		if pts == "" {
			pts = fr.PktPTSTime
		}
		dts := fr.DTSTime
		if dts == "" {
			dts = fr.PktDTSTime
		}
		if pts == "" {
			t.Fatalf("frame %d missing PTS", i)
		}
		if dts == "" {
			t.Fatalf("frame %d missing DTS", i)
		}
	}
}

func TestFinishMuxSyntheticShortest(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not in PATH")
	}

	dir := t.TempDir()
	videoPath := filepath.Join(dir, "v.mp4")
	audioPath := filepath.Join(dir, "a.wav")
	outPath := filepath.Join(dir, "mux.mp4")

	// 2s color video (no audio).
	videoCmd := exec.Command("ffmpeg", "-y",
		"-f", "lavfi", "-i", "color=c=red:s=64x64:r=25:d=2",
		"-c:v", "libx264", "-bf", "0", "-fps_mode", "cfr", "-pix_fmt", "yuv420p",
		videoPath,
	)
	if out, err := videoCmd.CombinedOutput(); err != nil {
		t.Fatalf("video setup: %v\n%s", err, out)
	}

	// ~0.5s silent WAV.
	audioCmd := exec.Command("ffmpeg", "-y",
		"-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
		"-t", "0.5",
		audioPath,
	)
	if out, err := audioCmd.CombinedOutput(); err != nil {
		t.Fatalf("audio setup: %v\n%s", err, out)
	}

	muxCmd := exec.Command("ffmpeg", ffmpegFinishMuxArgs(videoPath, audioPath, outPath)...)
	var stderr bytes.Buffer
	muxCmd.Stderr = &stderr
	if err := muxCmd.Run(); err != nil {
		t.Fatalf("mux: %v\n%s", err, stderr.String())
	}

	probeOut, err := exec.Command("ffprobe",
		"-hide_banner", "-loglevel", "error",
		"-show_entries", "format=duration",
		"-of", "default=noprint_wrappers=1:nokey=0",
		outPath,
	).CombinedOutput()
	if err != nil {
		t.Fatalf("ffprobe duration: %v\n%s", err, probeOut)
	}
	durLine := strings.TrimSpace(string(probeOut))
	if !strings.HasPrefix(durLine, "duration=") {
		t.Fatalf("unexpected ffprobe output: %q", durLine)
	}
	var dur float64
	if _, err := fmt.Sscanf(durLine, "duration=%f", &dur); err != nil {
		t.Fatalf("parse duration: %v", err)
	}
	if dur > 0.75 {
		t.Fatalf("expected mux output trimmed near audio length (~0.5s), got duration=%.3f", dur)
	}
}