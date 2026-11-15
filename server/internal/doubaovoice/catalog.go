package doubaovoice

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

const (
	listSpeakersAction  = "ListSpeakers"
	listSpeakersVersion = "2025-05-20"
	speechService       = "speech_saas_prod"
	speechRegion        = "cn-beijing"
	seedResourceID      = "seed-tts-2.0"
	defaultCacheTTL     = 7 * 24 * time.Hour
)

// Catalog resolves official Doubao TTS 2.0 trial clips (ListSpeakers / seed file). It never calls synthesis.
type Catalog struct {
	client *http.Client
	creds  credentials

	mu        sync.RWMutex
	byVoice   map[string]TrialInfo
	fetchedAt map[string]time.Time
	seed      map[string]TrialInfo
	ttl       time.Duration
}

func NewCatalogFromEnv() *Catalog {
	c := &Catalog{
		client: &http.Client{Timeout: 15 * time.Second},
		creds: credentials{
			accessKeyID: firstNonEmpty(
				os.Getenv("VOLC_ACCESS_KEY"),
				os.Getenv("DOUBAO_VOLC_ACCESS_KEY"),
			),
			secretAccessKey: firstNonEmpty(
				os.Getenv("VOLC_SECRET_KEY"),
				os.Getenv("DOUBAO_VOLC_SECRET_KEY"),
			),
			service: speechService,
			region:  speechRegion,
		},
		byVoice:   map[string]TrialInfo{},
		fetchedAt: map[string]time.Time{},
		seed:      loadTrialSeed(),
		ttl:       defaultCacheTTL,
	}
	return c
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func (c *Catalog) HasListSpeakersCredentials() bool {
	return c != nil && c.creds.accessKeyID != "" && c.creds.secretAccessKey != ""
}

func (c *Catalog) Lookup(ctx context.Context, voiceType string) (TrialInfo, error) {
	voiceType = strings.TrimSpace(voiceType)
	if voiceType == "" {
		return TrialInfo{}, fmt.Errorf("voice_type is required")
	}
	if !IsOfficialSeedTTS20Voice(voiceType) {
		return TrialInfo{}, fmt.Errorf("voice %s is not an official seed-tts-2.0 voice", voiceType)
	}

	if info, ok := c.cached(voiceType); ok {
		return info, nil
	}
	if info, ok := c.seed[voiceType]; ok {
		if _, playable := PickPlayableTrialURL(info); playable {
			c.store(voiceType, info)
			return info, nil
		}
	}
	if !c.HasListSpeakersCredentials() {
		return TrialInfo{}, fmt.Errorf("no official trial URL cached for %s (set VOLC_ACCESS_KEY/VOLC_SECRET_KEY for ListSpeakers sync)", voiceType)
	}
	info, err := c.fetchListSpeaker(ctx, voiceType)
	if err != nil {
		return TrialInfo{}, err
	}
	c.store(voiceType, info)
	return info, nil
}

func (c *Catalog) cached(voiceType string) (TrialInfo, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	info, ok := c.byVoice[voiceType]
	if !ok {
		return TrialInfo{}, false
	}
	if exp, ok := c.fetchedAt[voiceType]; ok && time.Since(exp) > c.ttl {
		return TrialInfo{}, false
	}
	if _, playable := PickPlayableTrialURL(info); !playable {
		return TrialInfo{}, false
	}
	return info, true
}

func (c *Catalog) store(voiceType string, info TrialInfo) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.byVoice[voiceType] = info
	c.fetchedAt[voiceType] = time.Now().UTC()
}

// StoreTrialForTest seeds cache in unit tests.
func (c *Catalog) StoreTrialForTest(voiceType string, info TrialInfo) {
	if c == nil {
		return
	}
	if c.byVoice == nil {
		c.byVoice = map[string]TrialInfo{}
	}
	if c.fetchedAt == nil {
		c.fetchedAt = map[string]time.Time{}
	}
	c.store(voiceType, info)
}

func (c *Catalog) fetchListSpeaker(ctx context.Context, voiceType string) (TrialInfo, error) {
	body, err := json.Marshal(map[string]any{
		"ResourceIDs": []string{seedResourceID},
		"VoiceTypes":  []string{voiceType},
		"Page":        1,
		"Limit":       5,
	})
	if err != nil {
		return TrialInfo{}, err
	}
	req, err := newSignedPost(c.creds, listSpeakersAction, listSpeakersVersion, body)
	if err != nil {
		return TrialInfo{}, err
	}
	req = req.WithContext(ctx)

	resp, err := c.client.Do(req)
	if err != nil {
		return TrialInfo{}, err
	}
	defer resp.Body.Close()
	payload, err := io.ReadAll(resp.Body)
	if err != nil {
		return TrialInfo{}, err
	}
	if resp.StatusCode >= 400 {
		return TrialInfo{}, fmt.Errorf("ListSpeakers failed with status %d", resp.StatusCode)
	}

	var parsed struct {
		Result struct {
			Speakers []map[string]any `json:"Speakers"`
		} `json:"Result"`
	}
	if err := json.Unmarshal(payload, &parsed); err != nil {
		return TrialInfo{}, err
	}
	for _, row := range parsed.Result.Speakers {
		rowVoice, _ := row["VoiceType"].(string)
		if strings.TrimSpace(rowVoice) == voiceType {
			return normalizeListSpeakerRow(voiceType, row)
		}
	}
	return TrialInfo{}, fmt.Errorf("ListSpeakers returned no speaker for %s", voiceType)
}

// ProxyTrialAudio streams an official trial clip. The upstream URL must already be validated.
func ProxyTrialAudio(ctx context.Context, client *http.Client, trialURL string) (*http.Response, error) {
	if client == nil {
		client = &http.Client{Timeout: 30 * time.Second}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, trialURL, nil)
	if err != nil {
		return nil, err
	}
	return client.Do(req)
}