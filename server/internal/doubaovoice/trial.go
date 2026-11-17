package doubaovoice

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"net/url"
	"strings"
)

type TrialInfo struct {
	VoiceType     string `json:"voice_type"`
	Name          string `json:"name,omitempty"`
	TrialURL      string `json:"trial_url"`
	ShortTrialURL string `json:"short_trial_url,omitempty"`
	SampleText    string `json:"sample_text,omitempty"`
	Source        string `json:"source,omitempty"`
}

// PickPlayableTrialURL chooses the official CDN trial clip URL (never a synthesis endpoint).
func PickPlayableTrialURL(info TrialInfo) (string, bool) {
	for _, candidate := range []string{info.ShortTrialURL, info.TrialURL} {
		if urlOK(candidate) {
			return strings.TrimSpace(candidate), true
		}
	}
	return "", false
}

func urlOK(raw string) bool {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return false
	}
	u, err := url.Parse(raw)
	if err != nil || u.Host == "" {
		return false
	}
	if u.Scheme != "https" {
		if u.Scheme != "http" || (!strings.HasPrefix(u.Host, "127.0.0.1:") && !strings.HasPrefix(u.Host, "localhost:")) {
			return false
		}
	}
	host := strings.ToLower(u.Host)
	if host == "www.volcengine.com" || host == "volcengine.com" {
		return false
	}
	if strings.Contains(raw, "openspeech.bytedance.com") {
		return false
	}
	return true
}

//go:embed trial_urls_seed.json
var trialSeedJSON []byte

func loadTrialSeed() map[string]TrialInfo {
	out := map[string]TrialInfo{}
	if len(trialSeedJSON) == 0 {
		return out
	}
	_ = json.Unmarshal(trialSeedJSON, &out)
	for voice, info := range out {
		info.VoiceType = voice
		info.Source = "seed"
		if play, ok := PickPlayableTrialURL(info); ok {
			info.TrialURL = play
		}
		out[voice] = info
	}
	return out
}

func normalizeListSpeakerRow(voiceType string, row map[string]any) (TrialInfo, error) {
	info := TrialInfo{VoiceType: voiceType, Source: "list_speakers"}
	if name, _ := row["Name"].(string); name != "" {
		info.Name = name
	}
	if trial, _ := row["TrialURL"].(string); trial != "" {
		info.TrialURL = trial
	}
	if short, _ := row["ShortTrialURL"].(string); short != "" {
		info.ShortTrialURL = short
	}
	if langs, ok := row["Languages"].([]any); ok {
		for _, langRow := range langs {
			m, _ := langRow.(map[string]any)
			if text, _ := m["Text"].(string); text != "" {
				info.SampleText = text
				break
			}
		}
	}
	if _, ok := PickPlayableTrialURL(info); !ok {
		return info, fmt.Errorf("voice %s has no playable official trial URL", voiceType)
	}
	return info, nil
}