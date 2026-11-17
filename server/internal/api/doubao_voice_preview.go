package api

import (
	"context"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/cyberverse/server/internal/doubaovoice"
)

func (r *Router) ensureDoubaoVoiceCatalog() *doubaovoice.Catalog {
	if r.doubaoVoiceCatalog == nil {
		r.doubaoVoiceCatalog = doubaovoice.NewCatalogFromEnv()
	}
	return r.doubaoVoiceCatalog
}

func (r *Router) handleDoubaoVoicePreviewMeta(w http.ResponseWriter, req *http.Request) {
	voiceType := strings.TrimSpace(req.PathValue("voice_type"))
	if voiceType == "" {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "voice_type is required"})
		return
	}
	if !doubaovoice.IsOfficialSeedTTS20Voice(voiceType) {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{
			Error: "official Doubao TTS 2.0 preview is only available for built-in voices",
		})
		return
	}

	ctx, cancel := context.WithTimeout(req.Context(), 12*time.Second)
	defer cancel()

	info, err := r.ensureDoubaoVoiceCatalog().Lookup(ctx, voiceType)
	if err != nil {
		writeJSON(w, http.StatusNotFound, ErrorResponse{Error: err.Error()})
		return
	}
	playURL, ok := doubaovoice.PickPlayableTrialURL(info)
	if !ok {
		writeJSON(w, http.StatusNotFound, ErrorResponse{Error: "no official trial audio for voice"})
		return
	}
	_ = playURL
	writeJSON(w, http.StatusOK, map[string]any{
		"voice_type":  voiceType,
		"name":        info.Name,
		"sample_text": info.SampleText,
		"source":      info.Source,
		"play_url":    "/api/v1/tts/doubao/voices/" + voiceType + "/preview/audio",
	})
}

func (r *Router) handleDoubaoVoicePreviewAudio(w http.ResponseWriter, req *http.Request) {
	voiceType := strings.TrimSpace(req.PathValue("voice_type"))
	if voiceType == "" {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "voice_type is required"})
		return
	}
	if !doubaovoice.IsOfficialSeedTTS20Voice(voiceType) {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "unsupported voice_type"})
		return
	}

	ctx, cancel := context.WithTimeout(req.Context(), 30*time.Second)
	defer cancel()

	info, err := r.ensureDoubaoVoiceCatalog().Lookup(ctx, voiceType)
	if err != nil {
		writeJSON(w, http.StatusNotFound, ErrorResponse{Error: err.Error()})
		return
	}
	trialURL, ok := doubaovoice.PickPlayableTrialURL(info)
	if !ok {
		writeJSON(w, http.StatusNotFound, ErrorResponse{Error: "no official trial audio for voice"})
		return
	}

	upstream, err := doubaovoice.ProxyTrialAudio(ctx, nil, trialURL)
	if err != nil {
		writeJSON(w, http.StatusBadGateway, ErrorResponse{Error: "failed to fetch official trial audio"})
		return
	}
	defer upstream.Body.Close()
	if upstream.StatusCode >= 400 {
		writeJSON(w, http.StatusBadGateway, ErrorResponse{Error: "official trial audio unavailable"})
		return
	}

	if ct := upstream.Header.Get("Content-Type"); ct != "" {
		w.Header().Set("Content-Type", ct)
	} else {
		w.Header().Set("Content-Type", "audio/mpeg")
	}
	w.Header().Set("Cache-Control", "public, max-age=86400")
	w.WriteHeader(http.StatusOK)
	_, _ = io.Copy(w, upstream.Body)
}