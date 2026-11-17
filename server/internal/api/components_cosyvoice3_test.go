package api

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/cyberverse/server/internal/config"
)

func TestLocalCosyVoice3ComponentExposesConfiguredVoice(t *testing.T) {
	root := t.TempDir()
	modelDir := filepath.Join(root, "infra", "config", "tts_models")
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "cosyvoice3.yaml"), []byte(`
cosyvoice3:
  plugin_class: inference.plugins.tts.cosyvoice3_plugin.CosyVoice3TTSPlugin
  model: Fun-CosyVoice3-0.5B-2512
  voice: cyberverse_default
`), 0644); err != nil {
		t.Fatal(err)
	}
	configDir := filepath.Join(root, "config")
	if err := os.MkdirAll(configDir, 0755); err != nil {
		t.Fatal(err)
	}
	configPath := filepath.Join(configDir, "cyberverse.yaml")
	if err := os.WriteFile(configPath, []byte(`
inference:
  tts:
    default: cosyvoice3
`), 0644); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.Load(configPath)
	if err != nil {
		t.Fatal(err)
	}
	r := &Router{configPath: configPath, cfg: cfg}

	options, err := r.configuredComponentOptions("tts")
	if err != nil {
		t.Fatal(err)
	}
	if len(options) != 1 {
		t.Fatalf("expected one tts option, got %+v", options)
	}
	opt := options[0]
	if opt.ID != "cosyvoice3" || opt.TTSKind != "local_cosyvoice3" {
		t.Fatalf("unexpected option: %+v", opt)
	}
	if opt.DefaultVoice != "cyberverse_default" || len(opt.Voices) != 1 || opt.Voices[0].Value != "cyberverse_default" {
		t.Fatalf("unexpected voices: %+v", opt.Voices)
	}
	if !r.isAllowedTTSVoice("cosyvoice3", "cyberverse_default") {
		t.Fatal("expected cyberverse_default to be allowed")
	}
	if r.isAllowedTTSVoice("cosyvoice3", "longanyang") {
		t.Fatal("cloud builtin voice should not be allowed for local provider")
	}
}