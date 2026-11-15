package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/cyberverse/server/internal/doubaovoice"
)

func TestDoubaoVoicePreviewMetaUsesSeedWithoutSynthesis(t *testing.T) {
	audioSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "audio/mpeg")
		_, _ = w.Write([]byte("fake-audio"))
	}))
	t.Cleanup(audioSrv.Close)

	catalog := doubaovoice.NewCatalogFromEnv()
	catalog.StoreTrialForTest("zh_female_xiaohe_uranus_bigtts", doubaovoice.TrialInfo{
		VoiceType:     "zh_female_xiaohe_uranus_bigtts",
		TrialURL:      audioSrv.URL + "/trial.mp3",
		SampleText:    "hello",
		Source:        "seed",
	})

	r := newTestRouter()
	r.doubaoVoiceCatalog = catalog

	req := httptest.NewRequest(http.MethodGet, "/api/v1/tts/doubao/voices/zh_female_xiaohe_uranus_bigtts/preview", nil)
	w := httptest.NewRecorder()
	r.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d %s", w.Code, w.Body.String())
	}
	var resp map[string]any
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if resp["play_url"] == "" {
		t.Fatal("expected play_url")
	}

	req2 := httptest.NewRequest(http.MethodGet, "/api/v1/tts/doubao/voices/zh_female_xiaohe_uranus_bigtts/preview/audio", nil)
	w2 := httptest.NewRecorder()
	r.Handler().ServeHTTP(w2, req2)
	if w2.Code != http.StatusOK {
		t.Fatalf("expected audio 200, got %d", w2.Code)
	}
	if ct := w2.Header().Get("Content-Type"); ct == "" {
		t.Fatal("expected content type")
	}
}